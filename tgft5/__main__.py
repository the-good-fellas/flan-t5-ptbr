from tgft5.downstream_gpt_task_training import start_gpt_task_training
from tgft5.downstream_task_training import start_task_training
from tgft5.roberta_training import start_roberta_training
from tgft5.run_gpt_training import start_gpt_training
from tgft5.create_t5_tokenizer import create_t5_tk
from tgft5.run_training import start_t5_training
from tgft5.arg_parser import TGFArgs
import logging

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

ignore_modelues_logs = ["jax", "urllib3"]

for _ in ignore_modelues_logs:
  logging.getLogger(_).setLevel(logging.CRITICAL)

if __name__ == '__main__':
  args = TGFArgs().get_params()
  if args.mode == 't5':
    start_t5_training(args)
  if args.mode == 'gpt':
    start_gpt_training(args)
  if args.mode == 'downstream':
    start_task_training(args)
  if args.mode == 'downstream-gpt':
    start_gpt_task_training(args)
  if args.mode == 'roberta':
    start_roberta_training(args)
  if args.mode == 'create_t5_tk':
    create_t5_tk(args)

