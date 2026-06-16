import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from processing_utils import (
    _cache_exists,
    _columnar,
    _get_cache_dir,
    _iter_rows,
    _load_repo_datasets,
    _merge_datasets,
    _normalize_content,
    _role_of,
    _to_conversational_list,
    create_assistant_labels,
    has_trainable_assistant,
)
from trl.data_utils import pack_dataset

from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import logging as hf_logging


if TYPE_CHECKING:
    from main import SFTScriptArguments


_DO_FLAG = {"train": "do_train", "valid": "do_eval", "test": "do_predict"}


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def sft_processor(example, _, tokenizer: PreTrainedTokenizer, args: TrainingArguments) -> Dict[str, list]:
    """통합된 SFT 데이터 전처리 함수 (이미지 처리 포함)"""
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    rows = []
    for row in _iter_rows(example):
        conversation = _to_conversational_list(row)
        if not conversation:
            continue

        images = row.get("images", None)
        processed = [
            {"role": turn.get("role", "user"), "content": _normalize_content(turn.get("content", ""), bool(images))}
            for turn in conversation
        ]

        # 학습 가능한 assistant turn 이 없으면(빈 문자열 등) 건너뛴다.
        if not has_trainable_assistant(processed):
            logger.warning("assistant 응답이 비어 있어 샘플을 건너뜁니다.")
            continue

        # 일부 샘플은 zero-width/특수문자/비라틴 스크립트로 assistant 구간을 char offset 으로
        # 찾지 못해 ValueError 가 발생한다. 이런 degenerate 샘플은 건너뛴다.
        try:
            labels, outputs = create_assistant_labels(tokenizer, processed, images=images)
        except ValueError:
            logger.warning("assistant 구간을 찾지 못해 샘플을 건너뜁니다.")
            continue
        rows.append({"labels": labels, args.length_column_name: len(outputs.input_ids), **outputs})

    return _columnar(rows)


def pretrain_processor(example, _, tokenizer: PreTrainedTokenizer, args: TrainingArguments) -> Dict[str, list]:
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    rows = []
    for row in _iter_rows(example):
        sentence = row["sentence_ls" if "sentence_ls" in row else "sentence"]
        outputs = tokenizer(sentence, return_length=True, return_tensors="np")

        for input_ids, length in zip(outputs.input_ids, outputs.length):
            if not np.isin(input_ids, tokenizer.eos_token_id).any():
                input_ids = np.append(input_ids, tokenizer.eos_token_id)
            if not np.isin(input_ids, tokenizer.bos_token_id).any():
                input_ids = np.insert(input_ids, 0, tokenizer.bos_token_id)
            rows.append({"input_ids": input_ids, args.length_column_name: length})

    return _columnar(rows)


PROCESSOR_REGISTRY = {"sft": sft_processor, "pretrain": pretrain_processor}


def processing_datasets(
    train_args: "SFTScriptArguments",
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset | None, Dataset | None, Dataset | None]:
    def _pack(dataset: Dataset, train_args, desc: str) -> Dataset:
        dataset = dataset.remove_columns(train_args.length_column_name)
        dataset = pack_dataset(dataset, train_args.max_length, train_args.packing_strategy, {"desc": desc})
        dataset = dataset.rename_column("seq_lengths", train_args.length_column_name)
        dataset.set_format("pt")
        return dataset

    if train_args.dataset_type not in PROCESSOR_REGISTRY:
        raise ValueError(f"알 수 없는 데이터 프로세서 타입: {train_args.dataset_type}")

    func = PROCESSOR_REGISTRY[train_args.dataset_type]
    is_main = train_args.distributed_state.is_local_main_process
    start_time = time.time()

    buckets = {"train": [], "valid": [], "test": []}

    for repo_name in train_args.dataset_repo_ls:
        if is_main:
            logger.info(f"Loading {repo_name}")

        truncate_map = train_args.dataset_truncate_map.get(repo_name) or {}
        datasets, data_name = _load_repo_datasets(repo_name, train_args, truncate_map)

        prefix = train_args.dataset_prefix.get(repo_name, train_args.dataset_prefix.get("default", {}))
        wanted_splits = sum(prefix.values(), [])
        datasets = DatasetDict({k: v for k, v in datasets.items() if k in wanted_splits})

        map_cache, filter_cache = _get_cache_dir(train_args, repo_name, data_name, datasets.keys(), truncate_map)
        use_main_process_first = not _cache_exists(map_cache)

        if is_main:
            logger.info(f"{repo_name}: cache_exists={not use_main_process_first}")

        context = (
            train_args.main_process_first(desc=f"preprocessing-{repo_name}")
            if use_main_process_first
            else nullcontext()
        )
        with context:
            datasets = datasets.map(
                func,
                num_proc=train_args.dataset_num_proc,
                load_from_cache_file=True,
                with_split=True,
                batched=True,
                batch_size=train_args.dataset_batch_size,
                cache_file_names=map_cache,
                remove_columns=list(set(sum(datasets.column_names.values(), []))),
                desc=f"preprocess-{repo_name}",
                fn_kwargs={"tokenizer": tokenizer, "args": train_args},
            )

            for split_key, dataset in datasets.items():
                role = _role_of(prefix, split_key)
                if role is None:
                    continue

                if split_key in truncate_map and len(dataset) > truncate_map[split_key]:
                    dataset = dataset.shuffle(seed=train_args.seed).select(range(truncate_map[split_key]))

                if role == "train" and train_args.do_train:
                    dataset = dataset.filter(
                        lambda lengths: [length <= train_args.max_length for length in lengths],
                        num_proc=train_args.dataset_num_proc,
                        input_columns=[train_args.length_column_name],
                        cache_file_name=filter_cache.get(split_key),
                        load_from_cache_file=True,
                        batched=True,
                        batch_size=train_args.dataset_batch_size,
                        desc=f"filter-{repo_name}/{split_key}",
                    )

                if getattr(train_args, _DO_FLAG[role]):
                    buckets[role].append(dataset)

    train_dataset = _merge_datasets(buckets["train"], "train", train_args.packing)
    valid_dataset = _merge_datasets(buckets["valid"], "valid", train_args.packing)
    test_dataset = _merge_datasets(buckets["test"], "test", train_args.packing)

    if is_main:
        logger.info(f"load_dataset_time: {time.time() - start_time:.2f}s")

    if train_dataset is not None and train_args.packing:
        train_dataset = _pack(train_dataset, train_args, "Packing dataset")

    if valid_dataset is not None and train_args.packing and train_args.eval_packing:
        valid_dataset = _pack(valid_dataset, train_args, "Packing eval dataset")

    return train_dataset, valid_dataset, test_dataset
