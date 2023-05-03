from transformers import AutoTokenizer
from datasets import load_dataset

datasets = load_dataset(
  'ds',
  use_auth_token=True
)


tokenizer = AutoTokenizer.from_pretrained('thegoodfellas/tgf-bpe-tokenizer', use_auth_token=True)


def _add_text(examples):
  'aqui'
  return examples


tokenized_datasets = datasets.map(
  _add_text,
  batched=True
)
