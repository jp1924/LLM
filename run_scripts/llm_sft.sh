export WANDB_PROJECT="LLM"
export WANDB_RUN_GROUP='packing-test'
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCHDYNAMO_DISABLE="OFF"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export OMP_NUM_THREADS=2

# accelerate launch --config_file="/root/workspace/config/fsdp.yaml" \
deepspeed --include=localhost:0,1,2,3 --master_port=8532 \
    '/root/workspace/src/train.py' \
    --output_dir="/root/output_dir/Trillion-7B-preview/OpenOrcaGuguKo" \
    --cache_dir="/root/.cache/.[Trillion-7B-preview]preprocess/OpenOrcaGuguKo" \
    --run_name="sft" \
    --model_name_or_path='trillionlabs/Trillion-7B-preview' \
    --preprocessing_batched=true \
    --preprocessing_num_workers=4 \
    --preprocessing_batch_size=1000 \
    --data_preprocessor_type="sft" \
    --dataset_repo_ls \
        jp1924/OpenOrcaGuguKo \
    --data_name_map='{"jp1924/OpenOrcaGuguKo": "SFT"}' \
    --data_truncate_map='{"jp1924/OpenOrcaGuguKo": {"train": "100000"}}' \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=60 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=2 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to='none' \
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
    --tf32=false \
    --data_max_length=2024 \
    --ddp_timeout=18000000 \
    --do_data_main_process_first=true \
    --use_liger_kernel=true \
    --torch_compile=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --attn_implementation='flash_attention_2' \
    --remove_unused_columns=true \
    --packing=true \
    --packing_max_elem=20 \
    --include_num_input_tokens_seen=true \
    --include_tokens_per_second=true \
    --deepspeed='/root/workspace/config/ZeRO_3_act_check.json'
