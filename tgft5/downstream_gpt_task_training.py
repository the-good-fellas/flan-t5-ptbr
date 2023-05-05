from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from datasets import load_dataset, DownloadConfig, Dataset
from flax.serialization import to_bytes, from_bytes
from huggingface_hub import Repository, create_repo
from flax.jax_utils import pad_shard_unpad
from flax import jax_utils, traverse_util
from flax.training import train_state
from typing import Callable
from copy import deepcopy
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

from tgft5.consts import (
  END_KEY,
  INSTRUCTION_KEY,
  RESPONSE_KEY_NL
)

from tgft5.preprocess_dataset import process_training_dataset

from transformers import (
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    set_seed
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TrainState(train_state.TrainState):
  dropout_rng: jnp.ndarray

  def replicate(self):
    return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


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


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
  """
  Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
  and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
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


def restore_checkpoint(load_dir, state):
  logger.info(f"Restoring checkpoint from {load_dir}")
  with open(os.path.join(load_dir, "flax_model.msgpack"), "rb") as f:
    params = from_bytes(state.params, f.read())
  with open(os.path.join(load_dir, "opt_state.msgpack"), "rb") as f:
    opt_state = from_bytes(state.opt_state, f.read())
  with open(os.path.join(load_dir, "training_state.json"), "r") as f:
    training_state = json.load(f)
  step = training_state["step"]
  logger.info(f"Checkpoint restored at step {step}")
  return state.replace(step=step, params=params, opt_state=opt_state), step


def start_gpt_task_training(args):
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level="NOTSET",
    datefmt="[%X]",
  )

  logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
  # if jax.process_index() == 0:
  #   datasets.utils.logging.set_verbosity_warning()
  #   transformers.utils.logging.set_verbosity_info()
  # else:
  #   datasets.utils.logging.set_verbosity_error()
  #   transformers.utils.logging.set_verbosity_error()

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

  if args.use_l2_regurarization:
    logger.debug('training with L2 Regularization')

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, revision=args.revision)

  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

  datasets = process_training_dataset(dataset=args.dataset_id, tokenizer=tokenizer)
  # Main data processing function that will concatenate all texts from our dataset and generate chunks
  # of expanded_inputs_length.

  response_token_ids = tokenizer.encode(RESPONSE_KEY_NL)

  def process_texts(examples):
    examples["labels"] = deepcopy(examples["input_ids"])

    # Find the position of the response token in each input sequence and modify the labels accordingly
    for i, input_ids in enumerate(examples["labels"]):
      response_token_ids_start_idx = input_ids.index(response_token_ids[0])
      response_token_ids_end_idx = response_token_ids_start_idx + 1

      labels = examples["labels"][i]
      labels[:response_token_ids_end_idx] = [-100] * response_token_ids_end_idx

    return examples

  logger.info(f"Start process_texts")
  lm_datasets = datasets.map(
    process_texts,
    batched=True,
    batch_size=args.group_text_batch_size,
    num_proc=args.preprocessing_num_workers,
    load_from_cache_file=False,
  )

  train_dataset = lm_datasets["train"]
  eval_dataset = lm_datasets["validation"]

  # Initialize our training
  rng = jax.random.PRNGKey(42)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  with jax.default_device(jax.devices("cpu")[0]):
    logger.info(f'loading weights from {args.lm_name}')
    model = FlaxAutoModelForCausalLM.from_pretrained(
      args.lm_name,
      seed=42,
      dtype=getattr(jnp, args.dtype),
      use_auth_token=True,
      revision=args.revision
    )

  def resize_token_embeddings(model, new_size, rnd_key):
    if model.config.vocab_size == new_size:
      return
    model.config.vocab_size = new_size
    params = model.params
    # params = unfreeze(params)
    old_embeddings = params['transformer']['wte']['embedding']
    old_size = old_embeddings.shape[0]
    dim = old_embeddings.shape[1]
    initializer = jax.nn.initializers.normal(stddev=model.config.initializer_range)
    new_embeddings = initializer(rnd_key, (new_size, dim))
    new_embeddings = new_embeddings.at[:old_size].set(old_embeddings)
    params['transformer']['wte']['embedding'] = new_embeddings
    # params = freeze(params)
    model.params = params

  resize_token_embeddings(model, 50_260, rng)

  # Store some constant
  num_epochs = int(args.epochs)
  train_batch_size = int(args.batch_size) * jax.device_count()
  eval_batch_size = int(args.per_device_eval_batch_size) * jax.device_count()

  # should change if using gradient acc?
  num_train_steps = len(datasets["train"]) // train_batch_size * num_epochs

  # Create learning rate schedule
  # linear_decay_lr_schedule_fn = create_learning_rate_fn(
  #   len(tokenized_datasets),
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
    layer_norm_named_params = set(
      [
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
      ]
    )
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

  # create optimizer
  if args.adafactor:
    if args.apply_grad_clipping:
      optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip_value),
        optax.adafactor(
          learning_rate=linear_decay_lr_schedule_fn
        )
      )
    else:
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

  if args.gradient_accumulation_steps > 1:
    optimizer = optax.MultiSteps(optimizer, args.gradient_accumulation_steps)
  grad_accum_steps = args.gradient_accumulation_steps

  # state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rngs)
  state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

  if args.resume_from_checkpoint:
    state, resume_step = restore_checkpoint(args.output_dir, state)
  else:
    resume_step = 0

  if args.skip_steps != 0:
    resume_step = args.skip_steps

  # Define gradient update step fn
  @jit
  def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
      labels = batch.pop("labels")
      logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

      loss = optax.softmax_cross_entropy(logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()
      return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
      {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

  # Create parallel version of the train step
  logger.info(f'initializing training. Devices: {jax.device_count()}')
  p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,), devices=jax.devices())

  # Define eval fn
  @jit
  def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    loss = optax.softmax_cross_entropy(logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])).mean()

    # summarize metrics
    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics

  p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

  # Replicate the train state on each device
  state = jax_utils.replicate(state)

  train_time = 0
  train_metrics = []
  epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)

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

  for epoch in epochs:
    w_run.log({'current_epoch': epoch + 1})
    # ======================== Training ================================
    train_start = time.time()

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    # Generate an epoch by shuffling sampling indices from the train dataset
    train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
    steps_per_epoch = len(train_dataset) // train_batch_size

    # train
    for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
      batch = next(train_loader)
      batch = shard(batch)
      state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
      train_metrics.append(train_metric)

      cur_step = epoch * (len(train_dataset) // train_batch_size) + step

      if cur_step % args.logging_steps * grad_accum_steps == 0 and cur_step > 0:
        # Save metrics
        # train_metric = jax_utils.unreplicate(train_metric)
        train_time += time.time() - train_start
        train_metrics = get_metrics(train_metrics)
        train_metrics = jax.tree_map(jnp.mean, train_metrics)

        # W&B
        for key, val in train_metrics.items():
          tag = f"train_{key}"
          w_run.log({tag: val}, step=cur_step)

        w_run.log({'train_time': train_time}, step=cur_step)
        w_run.log({'cur_step': cur_step})

        train_metrics = []

      if cur_step % args.eval_steps * grad_accum_steps == 0 and cur_step > 0:
        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)

        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
          # Model forward
          batch = next(eval_loader)
          metrics = pad_shard_unpad(p_eval_step, static_return=True)(
            state.params, batch, min_device_batch=args.per_device_eval_batch_size
          )
          eval_metrics.append(metrics)

        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # W&B
        for key, val in eval_metrics.items():
          tag = f"eval_{key}"
          w_run.log({tag: val.item()})

        try:
          eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
        except OverflowError:
          eval_metrics["perplexity"] = float("inf")

      if cur_step % args.save_steps * grad_accum_steps == 0 and cur_step > 0:
        if jax.process_index() == 0:
          save_checkpoint(
            model,
            args.output_dir,
            tokenizer,
            state,
            cur_step,
            repo,
            with_opt=True,
            push_to_hub=True
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
