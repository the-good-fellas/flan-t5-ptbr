from transformers import RobertaConfig, PreTrainedTokenizerFast
from huggingface_hub import HfApi, create_repo
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import os


def create_bpe_tk(args):
  os.makedirs(args.output_dir, exist_ok=True)
  create_repo(args.tokenizer_config, exist_ok=True, private=True)
  # load dataset
  dataset = load_dataset(args.dataset_id, split="train")

  vocab_size = args.vocab_size
  api = HfApi()

  # Instantiate tokenizer
  tokenizer = ByteLevelBPETokenizer()

  def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
      yield dataset[i: i + batch_size]["text"]

  # Customized training
  tokenizer.train_from_iterator(batch_iterator(), vocab_size=vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "<txcla>"
  ])

  bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

  # Save files to disk
  bpe_tokenizer.save_pretrained(args.output_dir)

  config = RobertaConfig.from_pretrained(args.lm_name, vocab_size=bpe_tokenizer.get_vocab_size())
  config.save_pretrained(args.output_dir)

  api.upload_folder(folder_path=args.output_dir, repo_id=args.tokenizer_config,
                    commit_message=f"trained from {args.dataset_id}")
