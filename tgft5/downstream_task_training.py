import logging
import math
import time
from functools import partial
from typing import Callable
import json
import os
import wandb

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from datasets import Dataset, load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.serialization import to_bytes
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository, create_repo
from tqdm import tqdm
from transformers import (
  FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
  AutoConfig,
  AutoTokenizer,
  FlaxT5ForConditionalGeneration,
  FlaxAutoModelForCausalLM
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TrainState(train_state.TrainState):
  dropout_rng: jnp.ndarray

  def replicate(self):
    return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
  """
  Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`,
  the final batch may be incomplete, and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
  """
  if shuffle:
    batch_idx = jax.random.permutation(rng, len(dataset))
    batch_idx = np.asarray(batch_idx)
  else:
    batch_idx = np.arange(len(dataset))

  if drop_last:
    steps_per_epoch = len(dataset) // batch_size
    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
  else:
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

  for idx in batch_idx:
    batch = dataset[idx]
    batch = {k: np.array(v) for k, v in batch.items()}

    yield batch


def create_learning_rate_fn(
  train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
  """Returns a linear warmup, linear_decay learning rate function."""
  steps_per_epoch = train_ds_size // train_batch_size
  num_train_steps = steps_per_epoch * num_train_epochs
  warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
  decay_fn = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
  )
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
  return schedule_fn


def save_checkpoint(model,
                    save_dir,
                    tokenizer,
                    state,
                    cur_step: int,
                    repo,
                    with_opt: bool = True,
                    push_to_hub: bool = False):
  state = jax_utils.unreplicate(state)
  if with_opt:
    logger.info(f'Saving optimizer and training state in {save_dir}...')
    with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
      f.write(to_bytes(state.opt_state))
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
      json.dump({"step": state.step.item()}, f)
  logger.info(f'Saving model in {save_dir} {"and pushing it to HF Hub" if push_to_hub else ""}')
  tokenizer.save_pretrained(save_dir)
  model.save_pretrained(save_dir, params=state.params)
  if cur_step == -1:
    message = f"Saving weights. Last push"
  else:
    message = f"Saving weights of step {cur_step}"

  repo.push_to_hub(commit_message=message, blocking=cur_step == -1)


def start_task_training(args):
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )

  # Setup logging, we only want one process per machine to log things on the screen.
  logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

  if jax.process_index() == 0:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_error()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

  # Set the verbosity to info of the Transformers logger (on main process only):
  logger.info(f"Training/evaluation parameters {args}")

  create_repo(args.hub_model_id, exist_ok=True, use_auth_token=True)
  repo = Repository(args.output_dir, clone_from=args.hub_model_id, use_auth_token=True)

  if args.dataset_subset is not None:
    dataset = load_dataset(
      args.dataset_id,
      args.dataset_subset,
      use_auth_token=True,
    )
  else:
    dataset = load_dataset(
      args.dataset_id,
      use_auth_token=True,
    )

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_config, use_auth_token=True)
  config = AutoConfig.from_pretrained(args.lm_name, use_auth_token=True)
  config.vocab_size = len(tokenizer)

  logger.info(f'loading weights from {args.lm_name}')
  # model = FlaxT5ForConditionalGeneration.from_pretrained(
  #   args.lm_name,
  #   seed=42,
  #   dtype=getattr(jnp, args.dtype),
  #   use_auth_token=True
  # )

  model = FlaxAutoModelForCausalLM.from_pretrained(
    args.lm_name,
    seed=42,
    dtype=getattr(jnp, args.dtype),
    use_auth_token=True
  )



  # if model.config.decoder_start_token_id is None:
  #   raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

  input_column = args.input_column
  target_column = args.target_column

  # Temporarily set max_target_length for training.
  max_target_length = args.max_target_length

  # In Flax, for seq2seq models we need to pass `decoder_input_ids`
  # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
  # for that dynamically import the `shift_tokens_right` function from the model file
  # model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
  # shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

  # Setting padding="max_length" as we need fixed length inputs for jitted functions
  def preprocess_function(examples):
    inputs = examples[input_column]
    targets = examples[target_column]
    model_inputs = tokenizer(
      inputs, max_length=args.max_target_length, padding="max_length", truncation=True, return_tensors="np"
    )

    # Setup the tokenizer for targets
    labels = tokenizer(
      text_target=targets,
      max_length=max_target_length,
      padding="max_length",
      truncation=True,
      return_tensors="np",
    )

    # model_inputs["labels"] = labels["input_ids"]
    # decoder_input_ids = shift_tokens_right_fn(
    #   labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
    # )
    # model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)
    #
    # # We need decoder_attention_mask so we can ignore pad tokens from loss
    # model_inputs["decoder_attention_mask"] = labels["attention_mask"]

    return model_inputs

  train_dataset = dataset["train"]
  train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=args.group_text_batch_size,
    num_proc=args.preprocessing_num_workers,
    remove_columns=args.column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on train dataset",
  )

  eval_dataset = dataset["validation"]
  eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=args.group_text_batch_size,
    num_proc=args.preprocessing_num_workers,
    remove_columns=args.column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on validation dataset",
  )

  # Metric
  # metric = evaluate.load(args.metric)

  # def postprocess_text(preds, labels):
  #   preds = [pred.strip() for pred in preds]
  #   labels = [label.strip() for label in labels]
  #
  #   # rougeLSum expects newline after each sentence
  #   preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  #   labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
  #
  #   return preds, labels

  # def compute_metrics(preds, labels):
  #   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
  #   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  #
  #   # Some simple post-processing
  #   decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
  #
  #   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  #   result = {k: round(v * 100, 4) for k, v in result.items()}
  #   prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
  #   result["gen_len"] = np.mean(prediction_lens)
  #   return result

  # Initialize our training
  rng = jax.random.PRNGKey(42)
  rng, dropout_rng = jax.random.split(rng)

  # Store some constant
  num_epochs = int(args.epochs)
  train_batch_size = int(args.batch_size) * jax.device_count()
  per_device_eval_batch_size = int(args.per_device_eval_batch_size)
  eval_batch_size = per_device_eval_batch_size * jax.device_count()
  steps_per_epoch = len(train_dataset) // train_batch_size
  total_train_steps = steps_per_epoch * num_epochs

  # Create learning rate schedule
  # linear_decay_lr_schedule_fn = create_learning_rate_fn(
  #   len(train_dataset),
  #   train_batch_size,
  #   args.epochs,
  #   args.warmup_steps,
  #   args.lr,
  # )

  # Define the inverse square root decay schedule
  # from https://github.com/deepmind/ithaca/blob/main/ithaca/util/optim.py
  @jax.jit
  def linear_warmup_and_sqrt_decay(global_step):
    """Linear warmup and then an inverse square root decay of learning rate."""
    linear_ratio = args.lr / args.warmup_steps
    decay_ratio = jnp.power(args.warmup_steps * 1.0, 0.5) * args.lr
    return jnp.minimum(linear_ratio * global_step,
                       decay_ratio * jnp.power(global_step, -0.5))

  decay_fn = linear_warmup_and_sqrt_decay

  linear_decay_lr_schedule_fn = optax.join_schedules(
    schedules=[decay_fn], boundaries=[args.warmup_steps]
  )

  # We use Optax's "masking" functionality to not apply weight decay
  # to bias and LayerNorm scale parameters. decay_mask_fn returns a
  # mask boolean with the same structure as the parameters.
  # The mask is True for parameters that should be decayed.
  def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
      layer[-2:]
      for layer_norm_name in layer_norm_candidates
      for layer in flat_params.keys()
      if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

  # create optimizer
  if args.adafactor:
    optimizer = optax.adafactor(
      learning_rate=linear_decay_lr_schedule_fn
    )
  else:
    optimizer = optax.adamw(
      learning_rate=linear_decay_lr_schedule_fn,
      b1=args.adam_beta1,
      b2=args.adam_beta2,
      eps=args.adam_epsilon,
      weight_decay=args.weight_decay,
      mask=decay_mask_fn,
    )

  # Setup train state
  state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

  # label smoothed cross entropy
  def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
    """
    The label smoothing implementation is adapted from Flax's official example:
    https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
    """
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing_factor
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
      confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = optax.softmax_cross_entropy(logits, soft_labels)
    loss = loss - normalizing_constant

    # ignore padded tokens from loss
    loss = loss * padding_mask
    loss = loss.sum()
    num_labels = padding_mask.sum()
    return loss, num_labels

  # Define gradient update step fn
  def train_step(state, batch, label_smoothing_factor=0.0):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params):
      labels = batch.pop("labels")
      logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
      loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
      return loss, num_labels

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, num_labels), grad = grad_fn(state.params)
    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    # true grad = total grad / total samples
    grad = jax.lax.psum(grad, "batch")
    grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
    new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

    metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
    return new_state, metrics

  # Define eval fn
  def eval_step(params, batch, label_smoothing_factor=0.0):
    labels = batch.pop("labels")
    logits = model(**batch, params=params, train=False)[0]

    loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    metrics = {"loss": loss}
    return metrics

  # Define generation function
  max_length = (
    args.val_max_target_length if args.val_max_target_length is not None else model.config.max_length
  )
  num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
  gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

  def generate_step(params, batch):
    model.params = params
    output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
    return output_ids.sequences

  # Create parallel version of the train and eval step
  p_train_step = jax.pmap(
    partial(train_step, label_smoothing_factor=args.label_smoothing_factor), "batch", donate_argnums=(0,)
  )
  p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=args.label_smoothing_factor), "batch")
  p_generate_step = jax.pmap(generate_step, "batch")

  # Replicate the train state on each device
  state = state.replicate()

  w_run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    id=args.wandb_run_id
  )

  w_run.log({'num_epochs': num_epochs})
  w_run.log({'num_train_steps': total_train_steps})
  w_run.log({"learning_rate": args.lr})
  w_run.log({"batch_size": args.batch_size})
  w_run.log({"effective_batch_size": train_batch_size})

  train_time = 0
  epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

  last_step = 0
  for epoch in epochs:
    w_run.log({'current_epoch': epoch + 1})
    # ======================== Training ================================
    train_start = time.time()

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)
    train_metrics = []

    # Generate an epoch by shuffling sampling indices from the train dataset
    train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
    steps_per_epoch = len(train_dataset) // train_batch_size
    # train
    for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
      cur_step = epoch * (len(train_dataset) // train_batch_size) + step
      batch = next(train_loader)
      batch = shard(batch)
      del batch['input']
      del batch['target']
      state, train_metric = p_train_step(state, batch)
      train_metrics.append(train_metric)

      last_step = cur_step
      if cur_step % args.logging_steps == 0 and cur_step > 0:
        train_metrics = get_metrics(train_metrics)
        train_metrics = jax.tree_map(jnp.mean, train_metrics)

        train_time += time.time() - train_start

        # W&B
        for key, val in train_metrics.items():
          tag = f"train_{key}"
          w_run.log({tag: val})

        w_run.log({'train_time': train_time})
        w_run.log({'cur_step': cur_step})

        train_metrics = []

    # ======================== Evaluating ==============================
    eval_metrics = []
    eval_preds = []
    eval_labels = []

    eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
    eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
    for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
      # Model forward
      batch = next(eval_loader)
      # labels = batch["labels"]

      del batch['input']
      del batch['target']

      metrics = pad_shard_unpad(p_eval_step, static_return=True)(
        state.params, batch, min_device_batch=per_device_eval_batch_size
      )
      eval_metrics.append(metrics)

      # generation
      # generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch)
      # eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
      # eval_labels.extend(labels)

    # normalize eval metrics
    eval_metrics = get_metrics(eval_metrics)
    eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

    # W&B
    for key, val in eval_metrics.items():
      tag = f"eval_{key}"
      w_run.log({tag: val.item()})

    if jax.process_index() == 0:
      save_checkpoint(model,
                      args.output_dir,
                      tokenizer,
                      state,
                      last_step,
                      repo,
                      with_opt=True,
                      push_to_hub=True
                      )

    # compute ROUGE metrics
    # rouge_metrics = compute_metrics(eval_preds, eval_labels)
    # eval_metrics.update(rouge_metrics)
    # rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])

  if jax.process_index() == 0:
    save_checkpoint(model,
                    args.output_dir,
                    tokenizer,
                    state,
                    last_step,
                    repo,
                    with_opt=False,
                    push_to_hub=True
                    )
