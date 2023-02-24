import argparse


class TGFArgs:
  def __init__(self):
    parser = argparse.ArgumentParser(description='TGF')
    parser.add_argument('-mode', default="t5", type=str)
    parser.add_argument('--assets', default="/root/data/thegoodfellas/assets", type=str)
    parser.add_argument('--hub_model_id', default='tgf-flan-t5-xl-ptbr', type=str)
    parser.add_argument('--output_dir', default="/root/data/thegoodfellas/checkpoints", type=str)
    parser.add_argument('--dataset_id', default="thegoodfellas/brwac", type=str)
    parser.add_argument('--tokenizer_config', default="thegoodfellas/tgf-sp-unigram-tokenizer-ptbr", type=str)
    parser.add_argument('--lm_name', default="google/flan-t5-xl", type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--preprocessing_num_workers', default=10, type=int)
    parser.add_argument('--dtype', default="bfloat16", type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--warmup_steps', default=10_000, type=int)
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--save_steps', default=80_000, type=int)
    parser.add_argument('--wandb_project', default="flan_t5_ptbr", type=str)
    parser.add_argument('--wandb_entity', default="thegoodfellas", type=str)
    parser.add_argument('--wandb_run_id', default="tgf-dummy", type=str)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)

    self.opts = parser.parse_args()

  def get_params(self):
    return self.opts
