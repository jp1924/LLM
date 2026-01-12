import json
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


def _get_cache_dir(train_args, repo_name, data_name, splits, truncate_map):
    """캐시 파일명 생성 헬퍼 함수"""
    model_name_or_path = (
        Path(train_args.model_name_or_path)
        if isinstance(train_args.model_name_or_path, str)
        else train_args.model_name_or_path
    )
    repo_name = Path(repo_name) if isinstance(repo_name, str) else repo_name
    run_name = train_args.run_name.replace("/", "_")

    cache_dir = HF_DATASETS_CACHE / "preprocess_cache" / model_name_or_path.name / run_name / repo_name.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # map 캐시: 전처리 결과 저장
    map_cache = {split: (cache_dir / f"map_{data_name}-{split}_preprocessor.arrow").as_posix() for split in splits}

    # filter 캐시: 필터링 결과 저장
    filter_cache = {}
    for split in splits:
        truncate_prefix = f"{truncate_map[split]}-" if split in truncate_map else ""
        filter_cache[split] = (
            cache_dir / f"filter_{truncate_prefix}{train_args.data_max_length}_{data_name}-{split}_preprocessor.arrow"
        ).as_posix()

    return map_cache, filter_cache


def check_dataset_cache_exists(
    map_cache: Dict[str, str],
    filter_cache: Optional[Dict[str, str]],
    train_split_keys: list,
    num_proc: Optional[int] = None,
) -> bool:
    def check_cache_files_exist(
        cache_file_path: str,
        num_proc: Optional[int] = None,
    ) -> bool:
        """
        캐시 파일들이 모두 존재하는지 확인

        Args:
            cache_file_path: 기본 캐시 파일 경로
            num_proc: 멀티프로세싱 worker 수

        Returns:
            모든 캐시 파일이 존재하면 True
        """
        # 단일 프로세스 또는 num_proc이 지정되지 않은 경우
        if num_proc is None or num_proc <= 1:
            return os.path.exists(cache_file_path)

        # 멀티프로세싱: 모든 worker의 캐시 파일 확인
        # datasets는 suffix 패턴을 사용: {base}_{rank:05d}_of_{num_proc:05d}.arrow
        base_name = os.path.splitext(cache_file_path)[0]
        for rank in range(num_proc):
            cache_file = f"{base_name}_{rank:05d}_of_{num_proc:05d}.arrow"
            if not os.path.exists(cache_file):
                return False

        return True

    """
    모든 데이터셋 split의 캐시가 존재하는지 확인

    Args:
        map_cache: map 캐시 파일 경로 딕셔너리
        filter_cache: filter 캐시 파일 경로 딕셔너리
        train_split_keys: train split 키 리스트
        num_proc: 멀티프로세싱 worker 수

    Returns:
        모든 split의 캐시가 존재하면 True
    """
    # map 캐시 확인
    for split_key, cache_path in map_cache.items():
        if not check_cache_files_exist(cache_path, num_proc):
            return False

    # filter 캐시 확인 (train split만)
    if filter_cache:
        for split_key, cache_path in filter_cache.items():
            if split_key in train_split_keys:
                # filter는 항상 단일 파일 (num_proc 고려 필요)
                if not check_cache_files_exist(cache_path, num_proc):
                    return False

    return True


def range_histogram(data, num_bins=50, width=50):
    """데이터 분포를 히스토그램으로 시각화"""
    if not data:
        return

    min_val, max_val = min(data), max(data)
    bin_size = (max_val - min_val) / num_bins

    # 빈도수 계산
    bins = [0] * num_bins
    for value in data:
        bin_index = min(int((value - min_val) / bin_size), num_bins - 1)
        bins[bin_index] += 1

    max_freq = max(bins)

    # 출력
    logger.info(f"\nHistogram (total {len(data)} items, {num_bins} bins)")
    logger.info("-" * 80)
    logger.info(f"Range{' ' * 18}Count  Distribution")
    logger.info("-" * 80)

    for i, count in enumerate(bins):
        start = min_val + (i * bin_size)
        end = min_val + ((i + 1) * bin_size)
        bar = "█" * int((count / max_freq) * width)
        logger.info(f"{start:8.0f}-{end:8.0f}: {count:6d} |{bar}")

    logger.info("-" * 80)
    logger.info("\nStatistics:")
    logger.info(f"데이터 개수: {len(data)}, 최소값: {min_val:.0f}, 최대값: {max_val:.0f}")
    logger.info(f"평균값: {sum(data) / len(data):.2f}, 구간 크기: {bin_size:.2f}")


def processing_datasets(
    train_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    func: Callable,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    if train_args.data_preprocessor_type not in PROCESSOR_REGISTRY:
        raise ValueError(f"알 수 없는 데이터 프로세서 타입: {train_args.data_preprocessor_type}")
    func = PROCESSOR_REGISTRY[train_args.data_preprocessor_type]

    start_time = time.time()
    is_main = train_args.distributed_state.is_local_main_process

    # 데이터셋 저장소
    datasets_by_split = {"train": [], "valid": [], "test": []}

    for repo_name in train_args.dataset_repo_ls:
        if is_main:
            logger.info(f"Loading {repo_name}")

        # 데이터셋 로드
        data_name = train_args.data_name_map.get(repo_name)
        truncate_map = train_args.data_truncate_map.get(repo_name, {}) or {}
        datasets = load_dataset(repo_name, data_name)

        # 필요한 split만 유지
        all_prefixes = sum(train_args.dataset_prefix.values(), [])
        datasets = DatasetDict({k: v for k, v in datasets.items() if k in all_prefixes})

        # 캐시 파일명 생성
        map_cache, filter_cache = _get_cache_dir(train_args, repo_name, data_name, datasets.keys(), truncate_map)

        # 캐시 존재 여부 확인 및 main_process_first 적용
        cache_exists = check_dataset_cache_exists(
            map_cache=map_cache,
            filter_cache=filter_cache,
            train_split_keys=train_args.dataset_prefix.get("train", []),
            num_proc=train_args.preprocessing_num_workers,
        )
        use_main_process_first = not cache_exists

        if is_main:
            logger.info(f"{repo_name}: cache_exists={cache_exists}, use_main_process_first={use_main_process_first}")

        with (
            train_args.main_process_first(desc=f"preprocessing-{repo_name}")
            if use_main_process_first
            else nullcontext()
        ):
        # 전처리 (토크나이징 등)
        datasets = datasets.map(
            func,
            num_proc=train_args.preprocessing_num_workers,
            load_from_cache_file=True,
            with_split=True,
            batched=True,
            batch_size=train_args.preprocessing_batch_size,
            cache_file_names=map_cache,
            remove_columns=set(sum(datasets.column_names.values(), [])),
            desc=f"preprocess-{repo_name}",
            fn_kwargs={"tokenizer": tokenizer, "args": train_args},
        )

        # 각 split 처리
        for split_key, dataset in datasets.items():
            original_size = len(dataset)

            # 크기 제한 (truncate)
            if split_key in truncate_map:
                truncate_size = truncate_map[split_key]
                if len(dataset) > truncate_size:
                    dataset = dataset.shuffle().select(range(truncate_size))
                elif is_main:
                    logger.info(f"{repo_name}/{split_key}: 크기 {len(dataset)} <= truncate {truncate_size}")

            # 길이 필터링 (train만)
            if split_key in train_args.dataset_prefix["train"] and train_args.do_train:
                dataset = dataset.filter(
                        lambda lengths: [l <= train_args.max_length for l in lengths],
                    num_proc=train_args.preprocessing_num_workers,
                    input_columns=[train_args.length_column_name],
                    cache_file_name=filter_cache[split_key] if filter_cache else None,
                        load_from_cache_file=True,
                    batched=True,
                    batch_size=train_args.preprocessing_batch_size,
                    desc=f"filter-{repo_name}/{split_key}",
            )

            # 로깅
            if is_main:
                top_lengths = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
                logger.info(f"{repo_name}/{split_key}-length: {[int(l) for l in top_lengths]}")
                logger.info(f"{repo_name}/{split_key}-size: {original_size} -> {len(dataset)}")

            # split별로 분류
            for split_type, prefixes in train_args.dataset_prefix.items():
                split_type = "predict" if split_type == "test" else split_type
                if split_key in prefixes:
                    do_flag = getattr(
train_args,
f"do_{split_type if split_type != 'valid' else 'eval'}",
)
                    if do_flag:
                        datasets_by_split[split_type].append(dataset)

    # 데이터셋 병합
    def merge_datasets(dataset_list, name):
        if not dataset_list:
            return None
        combined = concatenate_datasets(dataset_list)
        combined.set_format("pt")
        if is_main:
            logger.info(f"{name}_dataset:\n{combined}")
        return combined

    train_dataset = merge_datasets(datasets_by_split["train"], "train")
    valid_dataset = merge_datasets(datasets_by_split["valid"], "valid")
    test_dataset = merge_datasets(datasets_by_split["test"], "test")

    # # 히스토그램 (pretrain 제외)
    # if is_main and train_dataset and train_args.data_preprocessor_type != "pretrain":
    #     logger.info("train-datasets")
    #     range_histogram(train_dataset["length"], 100, 50)

    if is_main:
        logger.info(f"load_dataset_time: {time.time() - start_time:.2f}s")

    if train_dataset and train_args.packing:
        train_dataset = pack_dataset(
            train_dataset, train_args.max_length, train_args.packing_strategy, {"desc": "Packing dataset"}
        )

    return train_dataset, valid_dataset, test_dataset
