from tgft5.data_collator import FlaxDataCollatorForMaskedLanguageModeling
from flax.training.common_utils import get_metrics, onehot, shard
from flax.serialization import to_bytes, from_bytes
from huggingface_hub import Repository, create_repo
from datasets import load_dataset, DownloadConfig
from flax.training import train_state
from flax import jax_utils
from pathlib import Path
import jax.numpy as jnp
from tqdm import tqdm
from jax import jit
import numpy as np
import logging
import wandb
import optax
import json
import math
import time
import jax
import os

from transformers import (
    FlaxAutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    set_seed
)

logger = logging.getLogger(__name__)


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


def start_roberta_training(args):
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level="NOTSET",
    datefmt="[%X]",
  )
  # jax.distributed.initialize()

  assets_path = Path(args.assets)
  os.makedirs(assets_path, exist_ok=True)
  os.makedirs(args.output_dir, exist_ok=True)

  create_repo(args.hub_model_id, exist_ok=True, private=True)
  repo = Repository(args.output_dir, clone_from=args.hub_model_id)
  repo.git_pull()

  set_seed(42)

  tokenizer_name = args.tokenizer_config

  if args.from_pretrained:
    tokenizer_name = args.lm_name

  logger.debug(f'initializing tokezier from {tokenizer_name}')

  datasets = load_dataset(
    args.dataset_id,
    args.dataset_subset,
    use_auth_token=True,
    download_config=DownloadConfig(delete_extracted=True)
  )

  if "validation" not in datasets.keys():
    datasets["validation"] = load_dataset(
      args.dataset_id,
      args.dataset_subset,
      split=f"train[:{args.validation_split_count}]"
    )

    datasets["train"] = load_dataset(
      args.dataset_id,
      args.dataset_subset,
      split=f"train[{args.validation_split_count}:]"
    )
  else:
    datasets["validation"] = load_dataset(
      args.dataset_id,
      args.dataset_subset,
      split=f"validation[:{args.validation_split_count}]"
    )
    datasets["train"] = load_dataset(
      args.dataset_id,
      args.dataset_subset,
      split="train",
    )

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, revision=args.revision)
  config = AutoConfig.from_pretrained(args.lm_name, use_auth_token=True, revision=args.revision)

  column_names = datasets["train"].column_names
  text_column_name = "text" if "text" in column_names else column_names[0]

  max_length = min(args.max_length, tokenizer.model_max_length)

  def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

  tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not args.overwrite_cache,
  )

  def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_length) * max_length
    result = {
      k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
      for k, t in concatenated_examples.items()
    }
    return result

  logger.info(f"Start group_texts")

  tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=args.group_text_batch_size,
    num_proc=args.preprocessing_num_workers,
    load_from_cache_file=not args.overwrite_cache,
  )

  # Initialize our training
  rng = jax.random.PRNGKey(42)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  if args.from_pretrained:
    model = None
  else:
    model = FlaxAutoModelForMaskedLM.from_config(config, seed=42, dtype=jnp.dtype(args.dtype))

  # Store some constant
  num_epochs = int(args.epochs)
  train_batch_size = int(args.batch_size) * jax.device_count()
  eval_batch_size = int(args.per_device_eval_batch_size) * jax.device_count()

  # should change if using gradient acc?
  num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs

  w_run = wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    id=args.wandb_run_id
  )

  w_run.log({'num_epochs': num_epochs})
  w_run.log({'num_train_steps': num_train_steps})
  w_run.log({"learning_rate": args.lr})
  w_run.log({"batch_size": args.batch_size})
  w_run.log({"effective_batch_size": train_batch_size})

  linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=args.lr, end_value=0,
                                                      transition_steps=num_train_steps)

  adamw = optax.adamw(
    learning_rate=linear_decay_lr_schedule_fn,
    b1=args.adam_beta1,
    b2=args.adam_beta2,
    eps=args.adam_epsilon,
    weight_decay=args.weight_decay,
  )

  state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

  data_collator = FlaxDataCollatorForMaskedLanguageModeling(mlm_probability=0.15)

  def generate_batch_splits(num_samples, batch_size, rng=None):
    samples_idx = jax.numpy.arange(num_samples)

    # if random seed is provided, then shuffle the dataset
    if input_rng is not None:
      samples_idx = jax.random.permutation(input_rng, samples_idx)

    samples_to_remove = num_samples % batch_size

    # throw away incomplete batch
    if samples_to_remove != 0:
      samples_idx = samples_idx[:-samples_to_remove]

    batch_idx = np.split(samples_idx, num_samples // batch_size)
    return batch_idx

  @jit
  def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
      labels = batch.pop("labels")

      logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

      # compute loss, ignore padded input tokens
      label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
      loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

      # take average
      loss = loss.sum() / label_mask.sum()

      return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
      {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

  parallel_train_step = jax.pmap(train_step, "batch")

  @jit
  def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

    # compute accuracy
    accuracy = jax.numpy.equal(jax.numpy.argmax(logits, axis=-1), labels) * label_mask

    # summarize metrics
    metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
    metrics = jax.lax.psum(metrics, axis_name="batch")

    return metrics

  parallel_eval_step = jax.pmap(eval_step, "batch")

  state = jax_utils.replicate(state)

  def process_eval_metrics(metrics):
    metrics = get_metrics(metrics)
    metrics = jax.tree_map(jax.numpy.sum, metrics)
    normalizer = metrics.pop("normalizer")
    metrics = jax.tree_map(lambda x: x / normalizer, metrics)
    return metrics

  for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    w_run.log({'current_epoch': epoch + 1})
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_batch_idx = generate_batch_splits(len(tokenized_datasets["train"]), train_batch_size, rng=input_rng)

    with tqdm(total=len(train_batch_idx), desc="Training...", leave=False) as progress_bar_train:
      for batch_idx in train_batch_idx:
        model_inputs = data_collator(tokenized_datasets["train"][batch_idx], tokenizer=tokenizer, pad_to_multiple_of=16)

        # Model forward
        model_inputs = shard(model_inputs.data)
        state, train_metric, dropout_rngs = parallel_train_step(state, model_inputs, dropout_rngs)

        progress_bar_train.update(1)

      for key, val in train_metric.items():
        tag = f"train_{key}"
        w_run.log({tag: val.mean()})

      progress_bar_train.write(
        f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
      )

    # -- Eval --
    eval_batch_idx = generate_batch_splits(len(tokenized_datasets["validation"]), eval_batch_size)
    eval_metrics = []

    with tqdm(total=len(eval_batch_idx), desc="Evaluation...", leave=False) as progress_bar_eval:
      for batch_idx in eval_batch_idx:
        model_inputs = data_collator(tokenized_datasets["validation"][batch_idx], tokenizer=tokenizer)

        # Model forward
        model_inputs = shard(model_inputs.data)
        eval_metric = parallel_eval_step(state.params, model_inputs)
        eval_metrics.append(eval_metric)

        progress_bar_eval.update(1)

      eval_metrics_dict = process_eval_metrics(eval_metrics)
      for key, val in eval_metrics_dict.items():
        tag = f"eval_{key}"
        w_run.log({tag: val.item().mean()})

      progress_bar_eval.write(
        f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics_dict['loss']}, Acc: {eval_metrics_dict['accuracy']})"
      )

  if jax.process_index() == 0:
    save_checkpoint(model,
                    args.output_dir,
                    tokenizer,
                    state,
                    -1,
                    repo,
                    with_opt=False,
                    push_to_hub=True
                    )

  w_run.finish()