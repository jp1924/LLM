import json
import os
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.config import HF_DATASETS_CACHE
from trl.data_utils import pack_dataset

from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def get_sentencepiece_offset(tokenizer, input_ids):
    """토큰 오프셋 계산"""
    tokens = tokenizer.batch_decode([[token_id] for token_id in input_ids])
    special_tokens_set = set(tokenizer.all_special_tokens) | set(tokenizer.added_tokens_encoder.keys())

    text_parts = []
    positions = []
    current_pos = 0

    for i, token in enumerate(tokens):
        token_len = len(token)
        positions.append(
            (
                i,  # idx
                input_ids[i],  # token
                current_pos,  # start
                current_pos + token_len,  # end
                token in special_tokens_set,  # is_special
            )
        )
        text_parts.append(token)
        current_pos += token_len

    return positions, "".join(text_parts)


def create_assistant_labels(tokenizer, conversations, images=None):
    """
    Assistant 응답 구간만 학습하도록 레이블 생성

    Args:
        tokenizer: HuggingFace tokenizer
        conversations: 대화 메시지 리스트
        images: 이미지 리스트 (optional)

    Returns:
        input_ids, labels, outputs
    """
    if len(conversations) % 2:
        raise ValueError("Conversations should have an even number of messages (user starts, assistant ends).")

    # Chat template 적용 및 토큰화
    outputs = tokenizer(
        **{
            "text": tokenizer.apply_chat_template(conversations, tokenize=False),
            **({"images": images} if images else {}),
        },
        return_attention_mask=False,
        add_special_tokens=False,
    )

    text = tokenizer.decode(outputs.input_ids)

    # BOS/EOS 토큰 추가
    if tokenizer.bos_token and not text.strip().startswith(tokenizer.bos_token):
        text = tokenizer.bos_token + text
    if tokenizer.eos_token and not text.strip().endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    # 최종 인코딩 및 오프셋 계산
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    outputs.update({"input_ids": input_ids})
    offset_ls, text = get_sentencepiece_offset(tokenizer, input_ids)

    # Assistant 응답만 필터링
    answer_ls = [chat["content"] for chat in conversations if chat["role"] == "assistant"]
    answer_ids_list = tokenizer(answer_ls, add_special_tokens=False).input_ids

    # batch_decode로 한 번에 처리
    all_answer_tokens = []
    for ids in answer_ids_list:
        tokens = tokenizer.batch_decode([[token_id] for token_id in ids])
        all_answer_tokens.append("".join(tokens))

    # 위치 계산
    answer_pos_ls = []
    for new_answer in all_answer_tokens:
        start_idx = text.find(new_answer)
        if start_idx == -1:
            raise ValueError(f"tokenizer is weird. cannot find assistant content: {new_answer} > {text}")
        answer_pos_ls.append((start_idx, start_idx + len(new_answer) + 1))

    # 레이블 생성
    labels = [-100] * len(input_ids)
    for start_idx, end_idx in answer_pos_ls:
        token_start_idx = None
        token_end_idx = None

        for idx, _, start, _, _ in offset_ls:
            if start >= start_idx and token_start_idx is None:
                token_start_idx = idx
            if start < end_idx:
                token_end_idx = idx
            elif token_start_idx is not None:
                break

        if token_start_idx is not None and token_end_idx is not None:
            labels[token_start_idx : token_end_idx + 1] = input_ids[token_start_idx : token_end_idx + 1]

    return labels, outputs


def sft_processor(example, with_split: str, tokenizer: PreTrainedTokenizer, args: TrainingArguments):
    """통합된 SFT 데이터 전처리 함수 (이미지 처리 포함)"""
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    process_finish_ls = list()

    for row_dataset in list(zip(*[example[key] for key in example])):
        row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

        images = row_dataset.get("images", None)
        for chat_turn in row_dataset["conversations"]:
            content = json.loads(chat_turn["text"]) if images else chat_turn["text"].strip()
            chat_turn["content"] = content

        labels, outputs = create_assistant_labels(tokenizer, row_dataset["conversations"], images=images)

        finish_data = {
            "labels": np.array(labels),
            args.length_column_name: len(outputs.input_ids),
            **outputs,
        }

        process_finish_ls.append(finish_data)

    # 결과 딕셔너리 생성
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
            if not np.isin(input_ids, tokenizer.eos_token_id).any():
                input_ids = np.append(input_ids, tokenizer.eos_token_id)
            if not np.isin(input_ids, tokenizer.bos_token_id).any():
                input_ids = np.insert(input_ids, 0, tokenizer.bos_token_id)

            finish_data = {"input_ids": input_ids, args.length_column_name: length}
            process_finish_ls.append(finish_data)

    return_dict = dict()
    for res in process_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


PROCESSOR_REGISTRY = {"sft": sft_processor, "pretrain": pretrain_processor}

#############################################################################################################


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
            if dataset_size <= truncate_size and train_args.distributed_state.is_local_main_process:
                logger.info(
                    f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                )

        if dataset_key in train_args.dataset_prefix["train"] and train_args.do_train:
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

        if dataset_key in train_args.dataset_prefix["valid"] and train_args.do_eval:
            valid_dataset_ls.append(dataset)

        if dataset_key in train_args.dataset_prefix["test"] and train_args.do_predict:
            test_dataset_ls.append(dataset)

        if train_args.distributed_state.is_local_main_process:
            length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
            length_ls = [int(length) for length in length_ls]
            logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")
            logger.info(f"{repo_name}/{dataset_key}-size: {original_size} -> {len(dataset)}")

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

    # NOTE: pretrain과 같이 샘플의 개수가 너무 많은 경우 histogram 뽑는데 시간이 너무 오래 걸림.
    if (
        train_args.distributed_state.is_local_main_process
        and train_dataset
        and train_args.data_preprocessor_type != "pretrain"
    ):
        logger.info("train-datasets")
        range_histogram(train_dataset["length"], 100, 50)

    if train_args.distributed_state.is_local_main_process:
        logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

    return train_dataset, valid_dataset, test_dataset
