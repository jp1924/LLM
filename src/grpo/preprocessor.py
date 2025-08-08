import json
import time
from typing import Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

from transformers import ProcessorMixin, TrainingArguments
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def grpo_preprocessor(example, tokenizer: ProcessorMixin, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

        system = getattr(args, "system_prompt", "")
        conversations = [{"role": "system", "content": system}] if system else []
        for chat in row_dataset["conversations"]:
            try:
                # NOTE: VML의 경우 [{"role": "user", "content": '[{"type": "image"}, {"type": "text", "text": "test"}]'}]의 형태로 되어 있다 보니, json decoding이 필요함.
                #       그냥 raw string으로 되어 있다면, json.loads를 하지 않아도 되기 때문에, 예외처리로 처리함.
                chat["content"] = json.loads(chat["content"])
            except json.JSONDecodeError:
                pass

            conversations.append(chat)

        finish_data = {"prompt": conversations, "answer": row_dataset["answer"]}
        if "image" in row_dataset:
            finish_data["image"] = row_dataset["image"]

        process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def processing_datasets(
    train_args: TrainingArguments, tokenizer: ProcessorMixin
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    def process_dataset(dataset, dataset_key, repo_name, truncate_map):
        if dataset_key in truncate_map:
            truncate_size = truncate_map[dataset_key]
            dataset_size = len(dataset)
            dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
            if dataset_size <= truncate_size and train_args.distributed_state.is_local_main_process:
                logger.info(
                    f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                )

        if dataset_key in train_args.dataset_prefix["train"] and train_args.do_train:
            train_dataset_ls.append(dataset)

        if dataset_key in train_args.dataset_prefix["valid"] and train_args.do_eval:
            valid_dataset_ls.append(dataset)

        if dataset_key in train_args.dataset_prefix["test"] and train_args.do_predict:
            test_dataset_ls.append(dataset)

    def concat(datasets_ls, dataset_type):
        if datasets_ls:
            dataset = concatenate_datasets(datasets_ls)
            dataset.set_format("pt")
            if train_args.distributed_state.is_local_main_process:
                logger.info(f"{dataset_type}_dataset:\n{dataset}")
            return dataset
        return None

    start_time = time.time()
    train_dataset_ls, valid_dataset_ls, test_dataset_ls = [], [], []
    for repo_name in train_args.dataset_repo_ls:
        if train_args.distributed_state.is_local_main_process:
            logger.info(f"load-{repo_name}")

        data_name = train_args.data_name_map.get(repo_name, None)
        truncate_map = train_args.data_truncate_map.get(repo_name, {})
        datasets = load_dataset(repo_name, data_name)

        prefix_ls = (
            train_args.dataset_prefix["train"] + train_args.dataset_prefix["valid"] + train_args.dataset_prefix["test"]
        )
        for prefix in list(datasets.keys()):
            if prefix in prefix_ls:
                continue
            datasets.pop(prefix, None)

        datasets = datasets.map(
            grpo_preprocessor,
            num_proc=train_args.preprocessing_num_workers,
            batch_size=train_args.preprocessing_batch_size,
            remove_columns=set(sum(datasets.column_names.values(), [])),
            fn_kwargs={"tokenizer": tokenizer, "args": train_args},
            desc=f"preprocess-{repo_name}",
            keep_in_memory=True,
            batched=True,
        )

        for dataset_key in datasets:
            process_dataset(
                datasets[dataset_key],
                dataset_key,
                repo_name,
                truncate_map,
            )

    train_dataset = concat(train_dataset_ls, "train")
    valid_dataset = concat(valid_dataset_ls, "valid")
    test_dataset = concat(test_dataset_ls, "test")

    if train_args.distributed_state.is_local_main_process:
        logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

    return train_dataset, valid_dataset, test_dataset
