from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial
from .consts import (
  PROMPT_WITH_INPUT_FORMAT,
  PROMPT_NO_INPUT_FORMAT
)


def load_training_dataset(args):
  tokenizer = AutoTokenizer.from_pretrained('thegoodfellas/tgf-bpe-tokenizer', use_auth_token=True)
  
  datasets = load_dataset(
    args.dataset_id,
    use_auth_token=True
  )

  def _add_text(example):
    instruction = example["instruction"]
    response = example["response"]
    context = example.get("context")
    
    if context:
      example["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
    else:
      example["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
    return example

  def preprocess_batch(batch: dict[str, list], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

  datasets = datasets.map(_add_text)

  _preprocessing_function = partial(preprocess_batch, max_length=1024, tokenizer=tokenizer)
  datasets = datasets.map(
    _preprocessing_function,
    batched=True,
    remove_columns=["instruction", "context", "response", "text", "category"],
  )
  datasets
  return datasets



