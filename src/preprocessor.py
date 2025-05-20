import time
from typing import Callable, Optional, Tuple

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def sft_processor(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        for chat_turn in row_dataset["conversations"]:
            chat_turn["content"] = chat_turn["content"].strip()

        inst, text, labels = (
            row_dataset["conversations"][:-1],
            row_dataset["conversations"],
            row_dataset["conversations"][-1]["content"],
        )
        # chat format encoding
        inst: np.ndarray = tokenizer.apply_chat_template(inst, apply_conversation_template=True, return_tensors="np")
        text: np.ndarray = tokenizer.apply_chat_template(text, return_tensors="np")
        labels: np.ndarray = tokenizer.encode(labels, return_tensors="np", add_special_tokens=False)
        inst, text, labels = inst.squeeze(), text.squeeze(), labels.squeeze()

        # verify special token
        if np.isin(text, tokenizer.eos_token_id).any():
            text = np.append(text, tokenizer.eos_token_id)
        if np.isin(text, tokenizer.bos_token_id).any():
            text = np.insert(text, 0, tokenizer.bos_token_id)
            inst = np.insert(inst, 0, tokenizer.bos_token_id)

        finish_data = {}
        if with_split in args.train_dataset_prefix:
            labels = text.copy()
            labels[: len(inst)] = -100

            finish_data["input_ids"] = text
            finish_data["labels"] = labels
            finish_data[args.length_column_name] = len(text)

        elif with_split in args.valid_dataset_prefix + args.test_dataset_prefix:
            finish_data["input_ids"] = inst
            finish_data["labels"] = labels
            finish_data[args.length_column_name] = len(inst)

        process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def pretrain_processor(example, _, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        sentence = row_dataset["sentence_ls" if "sentence_ls" in row_dataset else "sentence"]
        outputs = tokenizer(sentence, return_length=True, return_tensors="np")
        input_ids_ls, length_ls = outputs.input_ids, outputs.length

        for input_ids, length in zip(input_ids_ls, length_ls):
            if np.isin(input_ids, tokenizer.eos_token_id).any():
                input_ids = np.append(input_ids, tokenizer.eos_token_id)
            if np.isin(input_ids, tokenizer.bos_token_id).any():
                input_ids = np.insert(input_ids, 0, tokenizer.bos_token_id)

            finish_data = {"input_ids": input_ids, args.length_column_name: length}
            process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def tnt_dual_script(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        conversations = [
            {"role": "spelling", "content": row_dataset["spelling"]},
            {"role": "phonetic", "content": row_dataset["phonetic"]},
            {"role": "dual_script", "content": row_dataset["sentence"]},
        ]
        inst, text, labels = (conversations[:-1], conversations, conversations[-1]["content"])
        # chat format encoding
        inst: np.ndarray = tokenizer.apply_chat_template(inst, apply_conversation_template=True, return_tensors="np")
        text: np.ndarray = tokenizer.apply_chat_template(text, return_tensors="np")
        labels: np.ndarray = tokenizer.encode(labels, return_tensors="np", add_special_tokens=False)
        inst, text, labels = inst.squeeze(), text.squeeze(), labels.squeeze()

        # verify special token
        if np.isin(text, tokenizer.eos_token_id).any():
            text = np.append(text, tokenizer.eos_token_id)
        if np.isin(text, tokenizer.bos_token_id).any():
            text = np.insert(text, 0, tokenizer.bos_token_id)
            inst = np.insert(inst, 0, tokenizer.bos_token_id)

        finish_data = {}
        if with_split in args.train_dataset_prefix:
            finish_data["input_ids"] = text
            finish_data["labels"] = text.copy()
            finish_data[args.length_column_name] = len(text)
        elif with_split in args.valid_dataset_prefix + args.test_dataset_prefix:
            finish_data["input_ids"] = inst
            finish_data["labels"] = labels
            finish_data[args.length_column_name] = len(inst)

        process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def s2p_tnt_processor(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        conversations = [
            {"role": "spelling", "content": row_dataset["spelling"]},
            {"role": "phonetic", "content": row_dataset["phonetic"]},
        ]
        inst, text, labels = (conversations[:-1], conversations, conversations[-1]["content"])
        # chat format encoding
        inst: np.ndarray = tokenizer.apply_chat_template(inst, apply_conversation_template=True, return_tensors="np")
        text: np.ndarray = tokenizer.apply_chat_template(text, return_tensors="np")
        labels: np.ndarray = tokenizer.encode(labels, return_tensors="np", add_special_tokens=False)
        inst, text, labels = inst.squeeze(), text.squeeze(), labels.squeeze()

        # verify special token
        if np.isin(text, tokenizer.eos_token_id).any():
            text = np.append(text, tokenizer.eos_token_id)
        if np.isin(text, tokenizer.bos_token_id).any():
            text = np.insert(text, 0, tokenizer.bos_token_id)
            inst = np.insert(inst, 0, tokenizer.bos_token_id)

        finish_data = {}
        if with_split in args.train_dataset_prefix:
            finish_data["input_ids"] = text
            finish_data["labels"] = text.copy()
            finish_data[args.length_column_name] = len(text)
        elif with_split in args.valid_dataset_prefix + args.test_dataset_prefix:
            finish_data["input_ids"] = inst
            finish_data["labels"] = labels
            finish_data[args.length_column_name] = len(inst)

        process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def p2s_tnt_processor(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()
    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416
        conversations = [
            {"role": "phonetic", "content": row_dataset["phonetic"]},
            {"role": "spelling", "content": row_dataset["spelling"]},
        ]
        inst, text, labels = (conversations[:-1], conversations, conversations[-1]["content"])
        # chat format encoding
        inst: np.ndarray = tokenizer.apply_chat_template(inst, apply_conversation_template=True, return_tensors="np")
        text: np.ndarray = tokenizer.apply_chat_template(text, return_tensors="np")
        labels: np.ndarray = tokenizer.encode(labels, return_tensors="np", add_special_tokens=False)
        inst, text, labels = inst.squeeze(), text.squeeze(), labels.squeeze()

        # verify special token
        if np.isin(text, tokenizer.eos_token_id).any():
            text = np.append(text, tokenizer.eos_token_id)
        if np.isin(text, tokenizer.bos_token_id).any():
            text = np.insert(text, 0, tokenizer.bos_token_id)
            inst = np.insert(inst, 0, tokenizer.bos_token_id)

        finish_data = {}
        if with_split in args.train_dataset_prefix:
            finish_data["input_ids"] = text
            finish_data["labels"] = text.copy()
            finish_data[args.length_column_name] = len(text)
        elif with_split in args.valid_dataset_prefix + args.test_dataset_prefix:
            finish_data["input_ids"] = inst
            finish_data["labels"] = labels
            finish_data[args.length_column_name] = len(inst)

        process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def all_tnt_processor(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    process_finish_ls = list()

    s2p_tnt_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    s2p_tnt = s2p_tnt_processor(example, with_split, tokenizer, args)
    s2p_tnt["input_ids"] = [np.insert(input_ids, 0, s2p_tnt_token) for input_ids in s2p_tnt["input_ids"]]
    for row_dataset in list(zip(*[s2p_tnt[key] for key in s2p_tnt])):
        row_dataset = {key: value for key, value in zip(s2p_tnt.keys(), row_dataset)}  # noqa: C416
        process_finish_ls.append(row_dataset)

    p2s_tnt_token = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    p2s_tnt = p2s_tnt_processor(example, with_split, tokenizer, args)
    p2s_tnt["input_ids"] = [np.insert(input_ids, 0, p2s_tnt_token) for input_ids in p2s_tnt["input_ids"]]
    for row_dataset in list(zip(*[p2s_tnt[key] for key in p2s_tnt])):
        row_dataset = {key: value for key, value in zip(p2s_tnt.keys(), row_dataset)}  # noqa: C416
        process_finish_ls.append(row_dataset)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


PROCESSOR_REGISTRY = {
    "sft": sft_processor,
    "pretrain": pretrain_processor,
    "tnt_dual": tnt_dual_script,
    "p2s_tnt": p2s_tnt_processor,
    "s2p_tnt": s2p_tnt_processor,
    "all_tnt": all_tnt_processor,
}


def range_histogram(data, num_bins=50, width=50):
    # 데이터의 최대값과 최소값 찾기
    min_val = min(data)
    max_val = max(data)

    # 구간 크기 계산
    bin_size = (max_val - min_val) / num_bins

    # 각 구간별 빈도수 계산
    bins = [0] * num_bins
    for value in data:
        bin_index = min(int((value - min_val) / bin_size), num_bins - 1)
        bins[bin_index] += 1

    # 최대 빈도수 찾기
    max_freq = max(bins)

    # 히스토그램 출력
    logger.info(f"\nHistogram (total {len(data)} items, {num_bins} bins)")
    logger.info("-" * 80)
    logger.info(f"Range{' ' * 18}Count  Distribution")
    logger.info("-" * 80)

    for i in range(num_bins):
        start = min_val + (i * bin_size)
        end = min_val + ((i + 1) * bin_size)
        bar_length = int((bins[i] / max_freq) * width)
        bar = "█" * bar_length

        # 구간과 빈도수, 막대 출력
        logger.info(f"{start:8.0f}-{end:8.0f}: {bins[i]:6d} |{bar}")

    logger.info("-" * 80)
    logger.info("\nStatistics:")
    logger.info(f"데이터 개수: {len(data)}")
    logger.info(f"최소값: {min_val:.0f}")
    logger.info(f"최대값: {max_val:.0f}")
    logger.info(f"평균값: {sum(data) / len(data):.2f}")
    logger.info(f"구간 크기: {bin_size:.2f}")


def processing_datasets(
    train_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    func: Callable,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    def process_dataset(dataset, dataset_key, repo_name, truncate_map, filter_cache_file_name):
        original_size = len(dataset)

        if dataset_key in truncate_map:
            truncate_size = truncate_map[dataset_key]
            dataset_size = len(dataset)
            dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
            if dataset_size <= truncate_size and train_args.is_world_process_zero:
                logger.info(
                    f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                )

        # if train_args.is_world_process_zero:
        #     range_histogram(dataset["length"], 100, 50)

        if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
            dataset = dataset.filter(
                lambda length_ls: [length <= train_args.data_max_length for length in length_ls],  # type: ignore
                num_proc=train_args.preprocessing_num_workers,
                input_columns=[train_args.length_column_name],
                cache_file_name=filter_cache_file_name[dataset_key],
                batched=train_args.preprocessing_batched,
                batch_size=train_args.preprocessing_batch_size,
                desc=f"length-filtering-{repo_name}/{dataset_key}",
            )
            train_dataset_ls.append(dataset)

        if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
            valid_dataset_ls.append(dataset)

        if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
            test_dataset_ls.append(dataset)

        if train_args.is_world_process_zero:
            length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
            length_ls = [int(length) for length in length_ls]
            logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")
            logger.info(f"{repo_name}/{dataset_key}-size: {original_size} -> {len(dataset)}")

    def concat(datasets_ls, dataset_type):
        if datasets_ls:
            dataset = concatenate_datasets(datasets_ls)
            dataset.set_format("pt")
            if train_args.is_world_process_zero:
                logger.info(f"{dataset_type}_dataset:\n{dataset}")
            return dataset
        return None

    start_time = time.time()
    train_dataset_ls, valid_dataset_ls, test_dataset_ls = [], [], []
    for repo_name in train_args.dataset_repo_ls:
        if train_args.is_world_process_zero:
            logger.info(f"load-{repo_name}")

        data_name = train_args.data_name_map.get(repo_name, None)
        truncate_map = train_args.data_truncate_map.get(repo_name, {})
        datasets = load_dataset(repo_name, data_name)

        prefix_ls = train_args.train_dataset_prefix + train_args.valid_dataset_prefix + train_args.test_dataset_prefix
        for prefix in list(datasets.keys()):
            if prefix in prefix_ls:
                continue
            datasets.pop(prefix, None)

        map_cache_file_name, filter_cache_file_name = None, None
        if train_args.cache_dir is not None:
            name = repo_name.split("/")[-1]
            name = f"{name}-{data_name}" if data_name else name

            map_cache_file_name = {
                x: train_args.cache_dir.joinpath(f"map_{name}-{x}_preprocessor.arrow").as_posix() for x in datasets
            }
            filter_cache_file_name = {
                x: train_args.cache_dir.joinpath(
                    f"filter_{f'{truncate_map[x]}-' if x in truncate_map else ''}{train_args.data_max_length}_{name}-{x}_preprocessor.arrow"
                ).as_posix()
                for x in datasets
            }

        datasets = datasets.map(
            func,
            num_proc=train_args.preprocessing_num_workers,
            load_from_cache_file=True,
            with_split=True,
            batched=train_args.preprocessing_batched,
            cache_file_names=map_cache_file_name,
            batch_size=train_args.preprocessing_batch_size,
            remove_columns=set(sum(datasets.column_names.values(), [])),
            desc=f"preprocess-{repo_name}",
            fn_kwargs={"tokenizer": tokenizer, "args": train_args},
        )

        for dataset_key in datasets:
            process_dataset(
                datasets[dataset_key],
                dataset_key,
                repo_name,
                truncate_map,
                filter_cache_file_name,
            )

    train_dataset = concat(train_dataset_ls, "train")
    valid_dataset = concat(valid_dataset_ls, "valid")
    test_dataset = concat(test_dataset_ls, "test")

    if train_args.is_world_process_zero and train_dataset:
        logger.info("train-datasets")
        range_histogram(train_dataset["length"], 100, 50)

    if train_args.is_world_process_zero:
        logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

    return train_dataset, valid_dataset, test_dataset
