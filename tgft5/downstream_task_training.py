import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import datasets
import evaluate
import jax
import jax.numpy as jnp
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import optax
from datasets import Dataset, load_dataset
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository, create_repo
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM
)
from transformers.utils import get_full_repo_name, is_offline_mode, send_example_telemetry


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
    transformers.utils.logging.set_verbosity_info()
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
  model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
    args.lm_name,
    seed=42,
    dtype=getattr(jnp, args.dtype),
    use_auth_token=True
  )

  if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

  prefix = args.source_prefix if args.source_prefix is not None else ""

  input_column = args.input_column
  target_column = args.target_column

  # Temporarily set max_target_length for training.
  max_target_length = args.max_target_length

  # In Flax, for seq2seq models we need to pass `decoder_input_ids`
  # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
  # for that dynamically import the `shift_tokens_right` function from the model file
  model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
  shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

  # Setting padding="max_length" as we need fixed length inputs for jitted functions
  def preprocess_function(examples):
    inputs = examples[input_column]
    targets = examples[target_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
      inputs, max_length=args.max_source_length, padding="max_length", truncation=True, return_tensors="np"
    )

    # Setup the tokenizer for targets
    labels = tokenizer(
      text_target=targets,
      max_length=max_target_length,
      padding="max_length",
      truncation=True,
      return_tensors="np",
    )

    model_inputs["labels"] = labels["input_ids"]
    decoder_input_ids = shift_tokens_right_fn(
      labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
    )
    model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

    # We need decoder_attention_mask so we can ignore pad tokens from loss
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]

    return model_inputs

  train_dataset = dataset["train"]
  train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=args.column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on train dataset",
  )

  eval_dataset = dataset["validation"]
  eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=args.column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on validation dataset",
  )

  # Metric
  metric = evaluate.load(args.metric)

  def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

  def compute_metrics(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

  # Initialize our training
  rng = jax.random.PRNGKey(args.seed)
  rng, dropout_rng = jax.random.split(rng)

  # Store some constant
  num_epochs = int(args.epochs)
  train_batch_size = int(args.batch_size) * jax.device_count()
  per_device_eval_batch_size = int(args.per_device_eval_batch_size)
  eval_batch_size = per_device_eval_batch_size * jax.device_count()
  steps_per_epoch = len(train_dataset) // train_batch_size
  total_train_steps = steps_per_epoch * num_epochs

  # Create learning rate schedule
  linear_decay_lr_schedule_fn = create_learning_rate_fn(
    len(train_dataset),
    train_batch_size,
    args.epochs,
    args.warmup_steps,
    args.lr,
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

  # create adam optimizer
  adamw = optax.adamw(
    learning_rate=linear_decay_lr_schedule_fn,
    b1=args.adam_beta1,
    b2=args.adam_beta2,
    eps=args.adam_epsilon,
    weight_decay=args.weight_decay,
    mask=decay_mask_fn,
  )

  # Setup train state
  state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)

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

  logger.info("***** Running training *****")
  logger.info(f"  Num examples = {len(train_dataset)}")
  logger.info(f"  Num Epochs = {num_epochs}")
  logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
  logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
  logger.info(f"  Total optimization steps = {total_train_steps}")

  train_time = 0
  epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)

  for epoch in epochs:
    # ======================== Training ================================
    train_start = time.time()

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)
    train_metrics = []

    # Generate an epoch by shuffling sampling indices from the train dataset
    train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
    steps_per_epoch = len(train_dataset) // train_batch_size
    # train
    for _ in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
      batch = next(train_loader)
      batch = shard(batch)
      state, train_metric = p_train_step(state, batch)
      train_metrics.append(train_metric)

    train_time += time.time() - train_start

    train_metric = unreplicate(train_metric)

    # ======================== Evaluating ==============================
    eval_metrics = []
    eval_preds = []
    eval_labels = []

    eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
    eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
    for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
      # Model forward
      batch = next(eval_loader)
      labels = batch["labels"]

      metrics = pad_shard_unpad(p_eval_step, static_return=True)(
        state.params, batch, min_device_batch=per_device_eval_batch_size
      )
      eval_metrics.append(metrics)

      # generation
      generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch)
      eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
      eval_labels.extend(labels)

    # normalize eval metrics
    eval_metrics = get_metrics(eval_metrics)
    eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

    # compute ROUGE metrics
    rouge_metrics = compute_metrics(eval_preds, eval_labels)
    eval_metrics.update(rouge_metrics)
    rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])


