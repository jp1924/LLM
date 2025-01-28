export WANDB_PROJECT="NIA95"
export WANDB_RUN_GROUP='[gemma2-9b]pretrain'

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCH_USE_CUDA_DSA="1"
export CUDA_LAUNCH_BLOCKING="1"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


deepspeed --num_gpus=4 \
    /root/workspace/src/main.py \
    --output_dir="/root/output_dir/gemma-2-9b/national-corpus/pretrain" \
    --cache_dir="/root/.cache/.[google/gemma-2-9b]preprocess/CorpusForLLMNationalRecordsAndArchives" \
    --run_name="[jp]llm-train" \
    --model_name_or_path="google/gemma-2-9b" \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --data_preprocessor_type="pretrain" \
    --dataset_repo_ls \
        jp1924/CorpusForLLMNationalRecordsAndArchives \
    --data_name_map='{"jp1924/CorpusForLLMNationalRecordsAndArchives": "CORPUS"}' \
    --train_dataset_prefix="train" \
    --per_device_train_batch_size=42 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=10 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --report_to='wandb' \
    --learning_rate=2e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --eval_strategy=no \
    --eval_steps=1 \
    --save_strategy=steps \
    --save_steps=500 \
    --logging_strategy="steps" \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --data_max_length=2048 \
    --ddp_timeout=18000000 \
    --profiling=false \
    --do_data_main_process_first=true \
    --use_liger_kernel=true \
    --torch_compile=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --padding_side="right" \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --attn_implementation="flash_attention_2" \
    --remove_unused_columns=true \
    --do_packing=true \
    --packing_max_elem=20 \
    --packing_shuffle=true \
    --group_by_length=false \
    --torch_empty_cache_steps=20 \
    --include_num_input_tokens_seen=true \
    --include_tokens_per_second=true \
    --deepspeed="/root/workspace/config/ZeRO_3_act_check.json"