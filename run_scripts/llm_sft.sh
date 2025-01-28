export WANDB_PROJECT="NIA95"
export WANDB_RUN_GROUP='[gemma2-9b]sft'
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1,2,3 --master_port=8532 \
    '/root/workspace/src/main.py' \
    --output_dir="/root/output_dir/gemma-2-9b/national-corpus/sft" \
    --cache_dir="/root/.cache/.[gemma-2-9b]preprocess/CorpusForLLMNationalRecordsAndArchives" \
    --run_name="[jp]llm-train" \
    --model_name_or_path='/root/output_dir/gemma-2-9b/national-corpus/pretrained_model' \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --data_preprocessor_type="sft" \
    --dataset_repo_ls \
        jp1924/CorpusForLLMNationalRecordsAndArchives \
    --data_name_map='{"jp1924/CorpusForLLMNationalRecordsAndArchives": "SFT"}' \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=38 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=1 \
    --num_train_epochs=2 \
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
    --chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not continue_final_message is defined %}{% set continue_final_message = false %}{% endif %}{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' }}{% if message.role == 'user' %}{{ '### User:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'system' %}{{ '### System:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'passage' %}{{ '### Passage:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'assistant' %}{{ '### Assistant:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ image_token }}{% elif content.type == 'text' %}{{ content.text }}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{% endif %}{% if not (continue_final_message and loop.last) %}{{ '<end_of_turn>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>' }}{{ '### Assistant:\n' }}{% elif not continue_final_message %}{{ eos_token }}{% endif %}" \
    --data_max_length=2048 \
    --ddp_timeout=18000000 \
    --profiling=false \
    --response_template='[106, 6176, 18145, 235292]' \
    --instruction_template='[106, 6176, 4926, 235292]' \
    --do_data_main_process_first=true \
    --use_liger_kernel=true \
    --torch_compile=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
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
    --include_num_input_tokens_seen=true \
    --include_tokens_per_second=true \
    --deepspeed='/root/workspace/config/ZeRO_3_act_check.json'

