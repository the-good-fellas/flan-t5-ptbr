from flax.training.common_utils import get_metrics, onehot, shard
from tgft5.data_collator import FlaxDataCollatorForT5MLM
from flax.serialization import to_bytes, from_bytes
from huggingface_hub import Repository, create_repo
from datasets import load_dataset, DownloadConfig
from flax import jax_utils, traverse_util
from flax.training import train_state
from jax.tree_util import tree_leaves
from itertools import chain
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
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    FlaxT5ForConditionalGeneration,
    T5Config,
    set_seed
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
  """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
  Training parameters to avoid padding with random_spans_noise_mask.
  When training a model with random_spans_noise_mask, we would like to set the other
  training hyperparmeters in a way that avoids padding.
  This function helps us compute these hyperparameters.
  We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
  and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
  This function tells us the required number of tokens in the raw example (for split_tokens())
  as well as the length of the encoded targets. Note that this function assumes
  the inputs and targets will have EOS appended and includes that in the reported length.
  Args:
      inputs_length: an integer - desired length of the tokenized inputs sequence
      noise_density: a float
      mean_noise_span_length: a float
  Returns:
      tokens_length: length of original text in tokens
      targets_length: an integer - length in tokens of encoded targets sequence
  """

  def _tokens_length_to_inputs_length_targets_length(tokens_length):
    num_noise_tokens = int(round(tokens_length * noise_density))
    num_nonnoise_tokens = tokens_length - num_noise_tokens
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # inputs contain all nonnoise tokens, sentinels for all noise spans
    # and one EOS token.
    _input_length = num_nonnoise_tokens + num_noise_spans + 1
    _output_length = num_noise_tokens + num_noise_spans + 1
    return _input_length, _output_length

  tokens_length = inputs_length

  while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
    tokens_length += 1

  inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

  # minor hack to get the targets length to be equal to inputs length
  # which is more likely to have been set to a nice round number.
  if noise_density == 0.5 and targets_length > inputs_length:
    tokens_length -= 1
    targets_length -= 1
  return tokens_length, targets_length


def generate_batch_splits(samples_idx, batch_size: int, drop_last=True) -> np.ndarray:
  """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
  the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
  num_samples = len(samples_idx)
  if drop_last:
    samples_to_remove = num_samples % batch_size
    if samples_to_remove != 0:
      samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    samples_idx = samples_idx.reshape((sections_split, batch_size))
  else:
    sections_split = math.ceil(num_samples / batch_size)
    samples_idx = np.array_split(samples_idx, sections_split)
  return samples_idx


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


def start_t5_training(args):
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

  if args.use_l2_regurarization:
    logger.debug('training with L2 Regularization')

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
  config = T5Config.from_pretrained(args.lm_name, use_auth_token=True, revision=args.revision)

  column_names = datasets["train"].column_names
  text_column_name = "text" if "text" in column_names else column_names[0]

  max_length = min(args.max_length, tokenizer.model_max_length)

  def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_attention_mask=False)

  tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not args.overwrite_cache,
  )

  expanded_inputs_length, targets_length = compute_input_and_target_lengths(
    inputs_length=max_length,
    noise_density=0.15,
    mean_noise_span_length=3.0,
  )

  # Main data processing function that will concatenate all texts from our dataset and generate chunks
  # of expanded_inputs_length.
  def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
      total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    # Split by chunks of max_len.
    result = {
      k: [t[i: i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
      for k, t in concatenated_examples.items()
    }
    return result

  # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
  # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
  # might be slower to preprocess.
  #
  # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
  # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
  logger.info(f"Start group_texts")
  tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=500,
    num_proc=args.preprocessing_num_workers,
    load_from_cache_file=not args.overwrite_cache,
  )

  # Initialize our training
  rng = jax.random.PRNGKey(42)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())

  with jax.default_device(jax.devices("cpu")[0]):
    if args.from_pretrained:
      logger.info(f'loading weights from {args.lm_name}')
      model = FlaxT5ForConditionalGeneration.from_pretrained(
        args.lm_name,
        seed=42,
        dtype=getattr(jnp, args.dtype),
        use_auth_token=True,
        revision=args.revision
      )
    else:
      logger.warning('creating model from scratch')
      model = FlaxT5ForConditionalGeneration(
        config,
        seed=42,
        dtype=getattr(jnp, args.dtype),
        _do_init=True
      )

  #   gradient_checkpointing=True

  data_collator = FlaxDataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=max_length,
    target_length=targets_length,
    pad_token_id=model.config.pad_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
  )

  # Store some constant
  num_epochs = int(args.epochs)
  train_batch_size = int(args.batch_size) * jax.device_count() * args.gradient_accumulation_steps
  eval_batch_size = int(args.per_device_eval_batch_size) * jax.device_count()

  # should change if using gradient acc?
  num_train_steps = len(tokenized_datasets["train"]) // train_batch_size * num_epochs * args.gradient_accumulation_steps

  # Create learning rate schedule
  warmup_fn = optax.linear_schedule(
    init_value=args.lr_init, end_value=args.lr, transition_steps=args.warmup_steps
  )

  # decay_fn = optax.linear_schedule(
  #   init_value=args.lr,
  #   end_value=0,
  #   transition_steps=num_train_steps - args.warmup_steps
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
      weight_decay=args.weight_decay,
      mask=decay_mask_fn,
    )

  # Setup train state

  if args.gradient_accumulation_steps > 1:
    optimizer = optax.MultiSteps(optimizer, args.gradient_accumulation_steps)

  grad_accum_steps = args.gradient_accumulation_steps

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
    labels = batch.pop("labels")

    def loss_fn(params):
      logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

      # compute softmax cross entropy loss
      softmax_xent_loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()

      if args.use_l2_regurarization:
        # compute L2 regularization loss
        l2_reg_loss = args.l2_regularization_weight * sum(jnp.sum(jnp.square(p)) for p in tree_leaves(params))

        # combine losses
        loss_step = softmax_xent_loss + l2_reg_loss

        return loss_step
      else:
        return softmax_xent_loss

    grad_fn = jax.value_and_grad(loss_fn)

    # initialize the gradients
    grad = jax.grad(loss_fn)(state.params)

    for i in range(1, grad_accum_steps):
      # calculate the gradients
      loss, new_grad = grad_fn(state.params)

      # accumulate the gradients
      grad = jax.tree_map(lambda x, y: x + y, grad, new_grad)
    #
    # take the mean of the accumulated gradients
    grad = jax.tree_map(lambda x: x / grad_accum_steps, grad)

    # loss, grad = grad_fn(state.params)
    # grad = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
      {
        "loss": loss,
        "learning_rate": linear_decay_lr_schedule_fn(state.step // grad_accum_steps)
      },
      axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

  # Create parallel version of the train step
  logger.info(f'initializing training. Devices: {jax.device_count()}')
  p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,), devices=jax.devices())

  # Define eval fn
  def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    # compute log-probabilities
    log_probs = jax.nn.log_softmax(logits)

    # compute negative log-likelihood
    neg_log_likelihood = -jnp.mean(jnp.sum(log_probs * onehot(labels, logits.shape[-1]), axis=-1))

    # compute accuracy
    accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

    # summarize metrics
    metrics = {"loss": neg_log_likelihood,
               "accuracy": accuracy.mean(),
               "ppl": jnp.exp(neg_log_likelihood)
               }
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    return metrics

  p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

  # Replicate the train state on each device
  state = jax_utils.replicate(state)

  train_time = 0
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
    train_metrics = []

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)
    # Generate an epoch by shuffling sampling indices from the train dataset
    num_train_samples = len(tokenized_datasets["train"])
    # Avoid using jax.numpy here in case of TPU training
    # train_samples_idx = np.random.permutation(np.arange(num_train_samples))
    # train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

    # IF THE DATASET IS TOO LONG, WE ONLY PROCEED SEQUENTIALLY WITHOUT SHUFFLING
    samples_to_remove = num_train_samples % (train_batch_size // grad_accum_steps)
    samples_idx = np.arange(num_train_samples)
    if samples_to_remove != 0:
      samples_idx = samples_idx[:-samples_to_remove]
    steps = num_train_samples // (train_batch_size // grad_accum_steps)

    # Gather the indexes for creating the batch and do a training step
    # for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
    #   samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
    for step in tqdm(range(steps), desc="Training...", position=1):
      cur_step = epoch * (num_train_samples // train_batch_size) + step
      if cur_step < resume_step:
        continue

      batch_idx = [x for x in range(step * train_batch_size, (step + 1) * train_batch_size)]
      samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]

      try:
        model_inputs = data_collator(samples)
      except ValueError as ve:
        logger.error(f'problematic batch {batch_idx} on step {cur_step} skipping')
        continue

      # local_host_model_inputs = {
      #   key: np.split(model_inputs.data[key], num_of_hosts, axis=0)[current_host_idx]
      #   for key, value in model_inputs.data.items()
      # }

      # Model forward
      model_inputs = shard(model_inputs.data)
      state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
      train_metrics.append(train_metric)

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
        # epochs.write(
        #   f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate:"
        #   f" {train_metric['learning_rate'].mean()})"
        # )

        train_metrics = []

      if cur_step % args.eval_steps * grad_accum_steps == 0 and cur_step > 0:
        # ======================== Evaluating ==============================
        num_eval_samples = len(tokenized_datasets["validation"])
        eval_samples_idx = jnp.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

        eval_metrics = []
        for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
          samples = [tokenized_datasets["validation"][int(idx)] for idx in batch_idx]
          model_inputs = data_collator(samples)

          # Model forward
          model_inputs = shard(model_inputs.data)
          metrics = p_eval_step(state.params, model_inputs)
          eval_metrics.append(metrics)

        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # W&B
        for key, val in eval_metrics.items():
          tag = f"eval_{key}"
          w_run.log({tag: val.item()})

        # Update progress bar
        # epochs.write(f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})")

      if cur_step % args.save_steps * grad_accum_steps == 0 and cur_step > 0:
        # save checkpoint after each epoch and push checkpoint to the hub
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
