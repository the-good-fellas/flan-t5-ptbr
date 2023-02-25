# flan-t5-xl-ptbr

nohup python3 -m tgft5 -mode t5 \
--hub_model_id thegoodfellas/tgf-flan-t5-base-ptbr \
--dataset_id thegoodfellas/brwac \
--lm_name google/flan-t5-base \
--wandb_run_id tgf-flan-t5-base-tpuv2-8-brwac_1 \
--save_steps 10_000 \
--warmup_steps 2_000 \
--batch_size 32 &

## Oscar
nohup python3 -m tgft5 -mode t5 \
--hub_model_id thegoodfellas/tgf-flan-t5-base-ptbr \
--dataset_id oscar \
--dataset_subset unshuffled_deduplicated_pt \
--lm_name thegoodfellas/tgf-flan-t5-base-ptbr \
--wandb_run_id tgf-flan-t5-base-tpuv2-8-oscar \
--save_steps 10_000 \
--warmup_steps 2_000 \
--batch_size 32 \
--from_pretrained