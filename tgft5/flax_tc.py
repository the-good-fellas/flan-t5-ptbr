import logging
import math
import time
from itertools import chain
from typing import Any, Callable, Dict
from sklearn.metrics import classification_report

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
from flax.training.common_utils import onehot, shard, get_metrics
from huggingface_hub import Repository, create_repo
from tqdm import tqdm
import wandb

from transformers import (
  AutoConfig,
  AutoTokenizer,
  FlaxAutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any


def create_train_state(
  model: FlaxAutoModelForSequenceClassification,
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


def start_train_flax_tc(args):
  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )

  accuracy_metric = evaluate.load("accuracy")
  f1_metric = evaluate.load("f1")
  precision_metric = evaluate.load("precision")
  recall_metric = evaluate.load("recall")

  current_labels = []
  current_preds = []

  def add_batch(preds, refs):
    accuracy_metric.add_batch(predictions=preds, references=refs)
    f1_metric.add_batch(predictions=preds, references=refs)
    precision_metric.add_batch(predictions=preds, references=refs)
    recall_metric.add_batch(predictions=preds, references=refs)

    current_preds.extend(preds)
    current_labels.extend(refs)

  # Setup logging, we only want one process per machine to log things on the screen.
  logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

  create_repo(args.hub_model_id, exist_ok=True, private=True)
  repo = Repository(args.output_dir, clone_from=args.hub_model_id)

  raw_datasets = load_dataset(
    args.dataset_id,
    args.dataset_subset,
    use_auth_token=True
  )

  label_list = raw_datasets["train"].unique("label")
  label_list.sort()  # Let's sort it for determinism
  num_labels = len(label_list)

  sent_feature = raw_datasets["train"].features["label"]
  label_names = sent_feature.names

  id2label = {i: label for i, label in enumerate(label_names)}
  label2id = {v: k for k, v in id2label.items()}

  # Load pretrained model and tokenizer
  config = AutoConfig.from_pretrained(
    args.lm_name,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    finetuning_task='text-classification',
    use_auth_token=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_config,
    use_auth_token=True,
    add_prefix_space=True
  )

  model = FlaxAutoModelForSequenceClassification.from_pretrained(
    args.lm_name,
    config=config,
    use_auth_token=True,
  )

  def preprocess_function(examples):
    result = tokenizer(examples["sentence"],
                       truncation=True,
                       padding='max_length',
                       max_length=256
                       )

    if "label" in examples:
      if label2id is not None:
        # Map labels to IDs (not necessary for GLUE tasks)
        # result["labels"] = [label2id[l] for l in examples["label"]]
        result["labels"] = examples["label"]
      else:
        # In all cases, rename the column to labels because the model will expect that.
        result["labels"] = examples["label"]
    return result

  processed_datasets = raw_datasets.map(
    preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
  )

  train_dataset = processed_datasets["train"]
  eval_dataset = processed_datasets["validation"]

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

  def compute_metrics():
    # predictions, labels = eval_pred
    # predictions = np.argmax(predictions, axis=1)
    #
    perclass_report = classification_report(current_preds, current_labels, target_names=label_names, output_dict=True)
    for label_report in perclass_report:
      if label_report in label_names:
        w_run.log({f'precision_{label_report}': perclass_report[label_report]['precision']})
        w_run.log({f'recall_{label_report}': perclass_report[label_report]['recall']})
        w_run.log({f'f1_{label_report}': perclass_report[label_report]['f1-score']})
        w_run.log({f'support_{label_report}': perclass_report[label_report]['support']})

    accuracy_overall = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_overall = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    precision_overall = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall_overall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')

    w_run.log(accuracy_overall)
    w_run.log(f1_overall)
    w_run.log(precision_overall)
    w_run.log(recall_overall)

    current_preds.clear()
    current_labels.clear()

  logger.info(f"===== Starting training ({num_epochs} epochs) =====")
  train_time = 0

  # make sure weights are replicated on each device
  state = replicate(state)

  step_per_epoch = len(train_dataset) // train_batch_size
  total_steps = step_per_epoch * num_epochs
  epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

  w_run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    id=args.wandb_run_id
  )

  w_run.log({'num_epochs': num_epochs})
  w_run.log({'num_train_steps': total_steps})
  w_run.log({"learning_rate": args.lr})
  w_run.log({"batch_size": args.batch_size})
  w_run.log({"effective_batch_size": train_batch_size})

  cur_step = 0
  for epoch in epochs:
    w_run.log({'current_epoch': epoch + 1})
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
        train_metrics = get_metrics(train_metrics)
        train_metrics = jax.tree_map(jnp.mean, train_metrics)
        train_time += time.time() - train_start
        # W&B
        for key, val in train_metrics.items():
          tag = f"train_{key}"
          w_run.log({tag: val}, step=cur_step)

        w_run.log({'train_time': train_time}, step=cur_step)
        w_run.log({'cur_step': cur_step})
        train_metrics = []

      if cur_step % args.eval_steps == 0 and cur_step > 0:
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

          add_batch(
            preds=predictions,
            refs=labels,
          )

        compute_metrics()

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

  w_run.finish()
