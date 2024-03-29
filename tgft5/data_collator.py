from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from typing import List, Dict
import numpy as np
import flax

from transformers import (
  BatchEncoding,
  PreTrainedTokenizerBase
)


@flax.struct.dataclass
class FlaxDataCollatorForMaskedLanguageModeling:
  mlm_probability: float = 0.15

  def __call__(self, examples, tokenizer, pad_to_multiple_of=16):
    batch = tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)

    special_tokens_mask = batch.pop("special_tokens_mask", None)
    batch["input_ids"], batch["labels"] = self.mask_tokens(
        batch["input_ids"], special_tokens_mask, tokenizer
    )

    return batch

  def mask_tokens(self, inputs, special_tokens_mask, tokenizer):
    labels = inputs.copy()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.full(labels.shape, self.mlm_probability)
    special_tokens_mask = special_tokens_mask.astype("bool")

    probability_matrix[special_tokens_mask] = 0.0
    masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
    indices_random &= masked_indices & ~indices_replaced
    random_words = np.random.randint(tokenizer.vocab_size, size=labels.shape, dtype="i4")
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


@flax.struct.dataclass
class FlaxDataCollatorForT5MLM:
  """
  Data collator used for T5 span-masked language modeling.
  It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of
  fixed length.
  For more information on how T5 span-masked language modeling works, one can take a look
  at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
  or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
  Args:
      tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
          The tokenizer used for encoding the data.
      noise_density (:obj:`float`):
          The probability with which to (randomly) mask tokens in the input.
      mean_noise_span_length (:obj:`float`):
          The average span length of the masked tokens.
      input_length (:obj:`int`):
          The expected input length after masking.
      target_length (:obj:`int`):
          The expected target length after masking.
      pad_token_id: (:obj:`int`):
          The pad token id of the model
      decoder_start_token_id: (:obj:`int):
          The decoder start token id of the model
  """

  tokenizer: PreTrainedTokenizerBase
  noise_density: float
  mean_noise_span_length: float
  input_length: int
  target_length: int
  pad_token_id: int
  decoder_start_token_id: int

  def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
    # convert list to dict and tensorize input
    batch = BatchEncoding(
      {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
    )

    input_ids = batch["input_ids"]
    batch_size, expandend_input_length = input_ids.shape

    mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
    labels_mask = ~mask_indices

    input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
    labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

    batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
    batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

    if batch["input_ids"].shape[-1] != self.input_length:
      raise ValueError(
        f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
        f" should be {self.input_length}."
      )

    if batch["labels"].shape[-1] != self.target_length:
      raise ValueError(
        f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
        f" {self.target_length}."
      )

    # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)`
    # and `self.tokenizer.batch_decode(labels)` here...
    batch["decoder_input_ids"] = shift_tokens_right(
      batch["labels"], self.pad_token_id, self.decoder_start_token_id
    )

    return batch

  def create_sentinel_ids(self, mask_indices):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids

  def filter_input_ids(self, input_ids, sentinel_ids):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
      [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
    )
    return input_ids

  def random_spans_noise_mask(self, length):
    """This function is copy of `random_spans_helper
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * self.noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
      """Partition a sequence of items randomly into non-empty segments.
      Args:
          num_items: an integer scalar > 0
          num_segments: an integer scalar in [1, num_items]
      Returns:
          a Tensor with shape [num_segments] containing positive integers that add
          up to num_items
      """
      mask_indices = np.arange(num_items - 1) < (num_segments - 1)
      np.random.shuffle(mask_indices)
      first_in_segment = np.pad(mask_indices, [[1, 0]])
      segment_id = np.cumsum(first_in_segment)
      # count length of sub segments assuming that list is sorted
      _, segment_length = np.unique(segment_id, return_counts=True)
      return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
      np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]
