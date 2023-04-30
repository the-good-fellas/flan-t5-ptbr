import logging
import math
import time
from itertools import chain
from typing import Any, Callable, Dict

import datasets
import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import struct
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForTokenClassification,
)

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any

logger = logging.getLogger(__name__)


def create_train_state(
  model: FlaxAutoModelForTokenClassification,
  num_labels: int,
  optimizer: any
) -> train_state.TrainState:
  """Create initial training state."""

  class TrainState(train_state.TrainState):
    """Train state with an Optax optimizer.

    The two functions below differ depending on whether the task is classification
    or regression.

    Args:
      logits_fn: Applied to last layer to obtain the logits.
      loss_fn: Function to compute the loss.
    """

    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)

  def cross_entropy_loss(logits, labels):
    xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
    return jnp.mean(xentropy)

  return TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=optimizer,
    logits_fn=lambda logits: logits.argmax(-1),
    loss_fn=cross_entropy_loss,
  )


def train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
  """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
  steps_per_epoch = len(dataset) // batch_size
  perms = jax.random.permutation(rng, len(dataset))
  perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
  perms = perms.reshape((steps_per_epoch, batch_size))

  for perm in perms:
    batch = dataset[perm]
    batch = {k: np.array(v) for k, v in batch.items()}
    batch = shard(batch)

    yield batch


def eval_data_collator(dataset: Dataset, batch_size: int):
  """Returns batches of size `batch_size` from `eval dataset`. Sharding handled by
  `pad_shard_unpad` in the eval loop."""
  batch_idx = np.arange(len(dataset))

  steps_per_epoch = math.ceil(len(dataset) / batch_size)
  batch_idx = np.array_split(batch_idx, steps_per_epoch)

  for idx in batch_idx:
    batch = dataset[idx]
    batch = {k: np.array(v) for k, v in batch.items()}

    yield batch


def start_train_flax_ner(args):
  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )

  # Setup logging, we only want one process per machine to log things on the screen.
  logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

  create_repo(args.hub_model_id, exist_ok=True, private=True)
  repo = Repository(args.output_dir, clone_from=args.hub_model_id)

  raw_datasets = load_dataset(
    args.dataset_id,
    args.dataset_subset,
    use_auth_token=True
  )

  column_names = raw_datasets["train"].column_names
  features = raw_datasets["train"].features
  text_column_name = "tokens"

  label_column_name = column_names[2]

  label_list = features[label_column_name].feature.names
  label_to_id = {l: i for i, l in enumerate(label_list)}
  id_to_label = {i: l for l, i in label_to_id.items()}

  num_labels = len(label_list)

  # Load pretrained model and tokenizer
  config = AutoConfig.from_pretrained(
    args.lm_name,
    num_labels=num_labels,
    label2id=label_to_id,
    id2label=id_to_label,
    finetuning_task='ner',
    use_auth_token=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_config,
    use_auth_token=True,
    add_prefix_space=True
  )

  model = FlaxAutoModelForTokenClassification.from_pretrained(
    args.lm_name,
    config=config,
    use_auth_token=True,
  )

  # Preprocessing the datasets
  # Tokenize all texts and align the labels with them.

  def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
      if word_id != current_word:
        # Start of a new word!
        current_word = word_id
        label = -100 if word_id is None else labels[word_id]
        new_labels.append(label)
      elif word_id is None:
        # Special token
        new_labels.append(-100)
      else:
        # Same word as previous token
        label = labels[word_id]
        # If the label is B-XXX we change it to I-XXX
        if label % 2 == 1:
          label += 1
        new_labels.append(label)

    return new_labels

  def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
      examples[text_column_name],
      truncation=False,
      is_split_into_words=True,
      padding='max_length',
      max_length=256
    )
    all_labels = examples["ner_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
      word_ids = tokenized_inputs.word_ids(i)
      new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

  processed_raw_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    load_from_cache_file=not args.overwrite_cache,
    remove_columns=raw_datasets["train"].column_names,
    desc="Running tokenizer on dataset",
  )

  train_dataset = processed_raw_datasets["train"]
  eval_dataset = processed_raw_datasets["validation"]

  num_epochs = int(args.epochs)
  rng = jax.random.PRNGKey(42)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  train_batch_size = args.batch_size * jax.local_device_count()
  per_device_eval_batch_size = int(args.per_device_eval_batch_size)
  eval_batch_size = args.per_device_eval_batch_size * jax.local_device_count()

  @jax.jit
  def linear_warmup_and_sqrt_decay(global_step):
    """Linear warmup and then an inverse square root decay of learning rate."""
    linear_ratio = args.lr / args.warmup_steps
    decay_ratio = jnp.power(args.warmup_steps * 1.0, 0.5) * args.lr
    return jnp.minimum(linear_ratio * global_step,
                       decay_ratio * jnp.power(global_step, -0.5))

  linear_decay_lr_schedule_fn = optax.join_schedules(
    schedules=[linear_warmup_and_sqrt_decay], boundaries=[args.warmup_steps]
  )

  optimizer = optax.adafactor(
    learning_rate=linear_decay_lr_schedule_fn
  )

  state = create_train_state(model, num_labels=num_labels, optimizer=optimizer)

  # define step functions
  def train_step(
    state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey
  ):
    """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    targets = batch.pop("labels")

    def loss_fn(params):
      logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
      loss = state.loss_fn(logits, targets)
      return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean({"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch")
    return new_state, metrics, new_dropout_rng

  p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

  def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.logits_fn(logits)

  p_eval_step = jax.pmap(eval_step, axis_name="batch")

  metric = evaluate.load("seqeval")

  def get_labels(y_pred, y_true):
    # Transform predictions and references tensos to numpy arrays

    # Remove ignored index (special tokens)
    true_predictions = [
      [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
      for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
      [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
      for pred, gold_label in zip(y_pred, y_true)
    ]

    true_labels = [[tag if not tag.startswith('B-') else 'I-' + tag[2:] for tag in sublist] for sublist in true_labels]

    return true_predictions, true_labels

  def compute_metrics():
    results = metric.compute()
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
      if isinstance(value, dict):
        for n, v in value.items():
          final_results[f"{key}_{n}"] = v
      else:
        final_results[key] = value
    return final_results

  logger.info(f"===== Starting training ({num_epochs} epochs) =====")
  train_time = 0

  # make sure weights are replicated on each device
  state = replicate(state)

  step_per_epoch = len(train_dataset) // train_batch_size
  total_steps = step_per_epoch * num_epochs
  epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

  cur_step = 0
  for epoch in epochs:
    train_start = time.time()
    train_metrics = []

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    # train
    for step, batch in enumerate(
      tqdm(
        train_data_collator(input_rng, train_dataset, train_batch_size),
        total=step_per_epoch,
        desc="Training...",
        position=1,
      )
    ):
      state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
      train_metrics.append(train_metric)
      cur_step = (epoch * step_per_epoch) + (step + 1)

      if cur_step % args.logging_steps == 0 and cur_step > 0:
        # Save metrics
        train_metric = unreplicate(train_metric)
        train_time += time.time() - train_start

        train_metrics = []

      if cur_step % args.eval_steps == 0 and cur_step > 0:
        eval_metrics = {}
        # evaluate
        for batch in tqdm(
          eval_data_collator(eval_dataset, eval_batch_size),
          total=math.ceil(len(eval_dataset) / eval_batch_size),
          desc="Evaluating ...",
          position=2,
        ):
          labels = batch.pop("labels")
          predictions = pad_shard_unpad(p_eval_step)(
            state, batch, min_device_batch=per_device_eval_batch_size
          )
          predictions = np.array(predictions)
          labels[np.array(chain(*batch["attention_mask"])) == 0] = -100
          preds, refs = get_labels(predictions, labels)
          metric.add_batch(
            predictions=preds,
            references=refs,
          )

        eval_metrics = compute_metrics()

      if (cur_step % args.save_steps == 0 and cur_step > 0) or (cur_step == total_steps):
        if jax.process_index() == 0:
          params = jax.device_get(unreplicate(state.params))
          model.save_pretrained(args.output_dir, params=params)
          tokenizer.save_pretrained(args.output_dir)
          repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)

  if jax.process_index() == 0:
    params = jax.device_get(unreplicate(state.params))
    model.save_pretrained(args.output_dir, params=params)
    tokenizer.save_pretrained(args.output_dir)
    repo.push_to_hub(commit_message=f"Saving last step: {cur_step}", blocking=False)
