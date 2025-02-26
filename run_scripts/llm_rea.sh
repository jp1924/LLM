export WANDB_PROJECT="LLM"
export WANDB_RUN_GROUP='reasoning-sft'
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCHDYNAMO_DISABLE="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
export OMP_NUM_THREADS=2

# think 토큰이 들어간 상태로 해야할 거임.
accelerate launch --config_file="/root/workspace/config/fsdp.yaml" \
    "/root/workspace/src/main.py" \
    --output_dir="/root/output_dir/ko-gemma-2-9b-it/R1/reasoning-sft" \
    --cache_dir="/root/.cache/.[ko-gemma-2-9b-it]preprocess/reasoning-sft" \
    --run_name="R1" \
    --model_name_or_path="/root/output_dir/ko-gemma-2-9b-it/R1" \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --data_preprocessor_type="reasoning_sft" \
    --dataset_repo_ls \
        "llami-team/Korean-OpenThoughts-114k-Normalized" \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=1 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to='wandb' \
    --learning_rate=2e-05 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.1 \
    --weight_decay=0 \
    --eval_strategy='no' \
    --eval_steps=1 \
    --save_strategy='epoch' \
    --save_steps=1 \
    --logging_strategy='steps' \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --data_max_length=26558 \
    --ddp_timeout=18000000 \
    --profiling=false \
    --do_data_main_process_first=true \
    --use_liger_kernel=true \
    --torch_compile=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --padding_side=right \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --chat_template="{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' }}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{% if role == 'user' %}{{ role + '\\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'text' %}{{ content.text | trim }}{% endif %}{% endfor %}{% else %}{{ message['content'] | trim }}{% endif %}{% elif role == 'model' %}{{ role + '\\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'think' %}{{ '<Think>' }}{{ content.text | trim }}{{ '</Think>' }}{% elif content.type == 'text' %}{{ content.text | trim }}{% endif %}{% endfor %}{% else %}{{ message['content'] | trim }}{% endif %}{% endif %}{{ '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}" \
    --attn_implementation='flash_attention_2' \
    --remove_unused_columns=false \
    --do_packing=true \
    --padding_side='right' \
    --packing_max_elem=20 \
    --packing_shuffle=true \
    --group_by_length=false \
    --torch_empty_cache_steps=100 \
    --optim='lomo' \
    --response_template='[106, 2516, 108]' \
    --instruction_template='[106, 1645, 108]' \
    --include_tokens_per_second=true \
    --include_num_input_tokens_seen=true
