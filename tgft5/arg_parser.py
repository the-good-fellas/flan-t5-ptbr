import argparse


class TGFArgs:
  def __init__(self):
    parser = argparse.ArgumentParser(description='TGF')
    parser.add_argument('-mode', default="t5", type=str)
    parser.add_argument('--assets', default="/root/data/thegoodfellas/assets", type=str)
    parser.add_argument('--hub_model_id', default='thegoodfellas/tgf-flan-t5-base-ptbr', type=str)
    parser.add_argument('--output_dir', default="/root/data/thegoodfellas/checkpoints", type=str)
    parser.add_argument('--dataset_id', default="thegoodfellas/brwac", type=str)
    parser.add_argument('--dataset_subset', type=str)
    parser.add_argument('--tokenizer_config', default="thegoodfellas/tgf-sp-unigram-tokenizer-ptbr", type=str)
    parser.add_argument('--lm_name', default="google/flan-t5-xl", type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--preprocessing_num_workers', default=10, type=int)
    parser.add_argument('--dtype', default="float32", type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--lr_init', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.999, type=float)
    parser.add_argument('--l2_regularization_weight', default=1e-3, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--warmup_steps', default=10_000, type=int)
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--save_steps', default=80_000, type=int)
    parser.add_argument('--eval_steps', default=10_000, type=int)
    parser.add_argument('--validation_split_count', default=50, type=int)
    parser.add_argument('--wandb_project', default="flan_t5_ptbr", type=str)
    parser.add_argument('--wandb_entity', default="thegoodfellas", type=str)
    parser.add_argument('--wandb_run_id', default="tgf-dummy", type=str)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--from_pretrained', action='store_true', default=False)
    parser.add_argument('--use_l2_regurarization', action='store_true', default=False)
    parser.add_argument('--add_new_tokens', action='store_true', default=False)
    parser.add_argument('--adafactor', action='store_true', default=False)
    parser.add_argument('--skip_steps', default=0, type=int)
    parser.add_argument('--revision', default='main', type=str)

    self.opts = parser.parse_args()

  def get_params(self):
    return self.opts
