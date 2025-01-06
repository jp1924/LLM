export WANDB_PROJECT=""
export WANDB_RUN_GROUP=''
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1,2,3 --master_port=8532 \
    '/root/workspace/main.py' \
    --output_dir='' \
    --cache_dir='/root/.cache/' \
    --run_name='' \
    --model_name_or_path='' \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --dataset_repo_ls \
        '' \
    --data_name_map='{"": ""}' \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=12 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=3 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to='wandb' \
    --learning_rate=2e-5 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --eval_strategy='no' \
    --eval_steps=1 \
    --save_strategy='epoch' \
    --save_steps=500 \
    --logging_strategy='steps' \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --data_max_length=3072 \
    --ddp_timeout=18000000 \
    --profiling=false \
    --do_data_main_process_first=true \
    --use_liger_kernel=true \
    --torch_compile=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --chat_template="" \
    --response_template="" \
    --padding_side='right' \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --attn_implementation='flash_attention_2' \
    --remove_unused_columns=true \
    --do_packing=true \
    --packing_max_elem=20 \
    --packing_shuffle=true \
    --group_by_length=false \
    --torch_empty_cache_steps=100 \
    --deepspeed='/root/workspace/config/ZeRO_3_act_check.json'
