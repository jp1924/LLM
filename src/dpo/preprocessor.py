import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Tuple

from datasets import Dataset, DatasetDict
from processing_utils import (
    _cache_exists,
    _columnar,
    _get_cache_dir,
    _iter_rows,
    _load_repo_datasets,
    _merge_datasets,
    _role_of,
    create_assistant_labels,
    has_trainable_assistant,
)
from trl.data_utils import pack_dataset

from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import logging as hf_logging


if TYPE_CHECKING:
    from main import DPOScriptArguments


_DO_FLAG = {"train": "do_train", "valid": "do_eval", "test": "do_predict"}
PROMPT_KEYS = ("prompt", "question", "instruction", "input", "orig_prompt", "prompt_aug")
CHOSEN_KEYS = ("chosen", "chosen_response", "positive", "answer")
REJECT_KEYS = ("rejected", "rejected_response", "negative", "reject")
HISTORY_KEYS = ("history", "context", "conversations", "messages", "chosen_messages")

hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def _build_dpo_conversations(prompt, chosen, reject, history, images):
    """다양한 입력 포맷을 (chosen_conv, reject_conv) 두 개의 완성형 conversation 으로 정규화.

    지원 케이스:
      (A) chosen/reject 가 그 자체로 conversation ls 인 경우
          → 마지막 assistant turn 만 서로 다르고 앞 구간은 공통이라고 가정하고 그대로 사용.
      (B) prompt 가 conversation ls(이전 turn 포함, user 로 끝남) + chosen/reject 가 문자열
      (C) history(이전 turn ls) + 단일 prompt 문자열 + chosen/reject 문자열
      (D) 단일 prompt 문자열 + chosen/reject 문자열 (싱글턴)

    실패하면 (None, None) 반환.
    """

    def _user_message(text: str, images) -> dict:
        """images 유무에 맞춰 user 메시지 한 개 구성."""
        if images is not None:
            return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        return {"role": "user", "content": text}

    def _is_conversation(value) -> bool:
        """role/content 형식 메시지들의 list 인지 판별."""
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(isinstance(m, dict) and "role" in m and "content" in m for m in value)
        )

    # (A) chosen/reject 가 conversation ls 인 경우
    if _is_conversation(chosen) and _is_conversation(reject):
        return list(chosen), list(reject)

    # 여기서부터 chosen/reject 는 마지막 답변 문자열이어야 한다.
    if not isinstance(chosen, str) or not isinstance(reject, str):
        return None, None

    base = []
    # (B) prompt 가 conversation ls (이전 turn 들을 포함)
    if _is_conversation(prompt):
        base = list(prompt)
    else:
        # (C) 별도 history(이전 turn ls) 가 있으면 먼저 깔아준다.
        if _is_conversation(history):
            base = list(history)
        # (D) 마지막 user turn(=prompt 문자열) 추가
        if isinstance(prompt, str):
            base = base + [_user_message(prompt, images)]
        elif not base:
            return None, None

    chosen_conv = base + [{"role": "assistant", "content": chosen}]
    reject_conv = base + [{"role": "assistant", "content": reject}]
    return chosen_conv, reject_conv


def _split_prompt_completion(labels: list, input_ids: list) -> Tuple[list, list]:
    """labels 의 마지막 assistant 구간(=last answer)을 기준으로 (prompt_ids, completion_ids) 분리.

    create_assistant_labels 는 모든 assistant turn 을 라벨링하므로, 멀티턴에서는
    -100 이 아닌 '마지막 연속 구간' 만 completion 으로 떼어내야 한다.
    그 앞부분(이전 turn 전부 포함)은 chosen/reject 에서 공통인 prompt 가 된다.
    """
    last = next((i for i in range(len(labels) - 1, -1, -1) if labels[i] != -100), None)
    if last is None:
        # assistant 구간을 못 찾으면 전부 prompt 로 둔다(이상 케이스).
        return input_ids, []

    start = last
    while start > 0 and labels[start - 1] != -100:
        start -= 1
    return input_ids[:start], input_ids[start : last + 1]


def dpo_processor(example, _, tokenizer: PreTrainedTokenizer, args: TrainingArguments) -> Dict[str, list]:
    def _get_value(row: dict, keys: Tuple[str, ...]):
        """row 에서 keys 순서대로 처음 존재하는(None 이 아닌) 값을 반환한다."""
        for key in keys:
            value = row.get(key)
            if value is not None:
                return value
        return None

    tokenizer = getattr(tokenizer, "tokenizer")

    rows = []
    for row in _iter_rows(example):
        prompt = _get_value(row, PROMPT_KEYS)
        chosen = _get_value(row, CHOSEN_KEYS)
        reject = _get_value(row, REJECT_KEYS)
        history = _get_value(row, HISTORY_KEYS)  # 이전 turn 들의 conversation ls (optional)
        images = row.get("images")  # images 는 optional하다.

        if chosen is None or reject is None:
            continue

        # 다양한 입력 포맷을 두 개의 완성형 conversation(assistant 로 끝남)으로 정규화.
        chosen_conv, reject_conv = _build_dpo_conversations(prompt, chosen, reject, history, images)
        if chosen_conv is None:
            continue

        # chosen/reject 의 마지막 답변이 빈 문자열이면 라벨링할 구간이 없어
        # prompt 구간이 무너져 mismatch 로 이어지므로 미리 건너뛴다.
        if not has_trainable_assistant(chosen_conv) or not has_trainable_assistant(reject_conv):
            logger.warning("chosen/reject 답변이 비어 있어 샘플을 건너뜁니다.")
            continue

        # 일부 샘플은 zero-width/특수문자로 assistant 구간을 char offset 으로 찾지 못해
        # ValueError 가 발생한다. 이런 degenerate 샘플은 건너뛴다.
        try:
            chosen_labels, chosen_outputs = create_assistant_labels(tokenizer, chosen_conv, images)
            reject_labels, reject_outputs = create_assistant_labels(tokenizer, reject_conv, images)
        except ValueError:
            logger.warning("assistant 구간을 찾지 못해 샘플을 건너뜁니다.")
            continue
        chosen_prompt_ids, chosen_ids = _split_prompt_completion(chosen_labels, chosen_outputs.input_ids)
        reject_prompt_ids, reject_ids = _split_prompt_completion(reject_labels, reject_outputs.input_ids)

        # 멀티턴이어도 last answer 이전 구간들은 동일해야 한다.
        if chosen_prompt_ids != reject_prompt_ids:
            logger.warning("chosen/reject 의 prompt 구간이 일치하지 않아 샘플을 건너뜁니다.")
            continue

        prompt_ids = chosen_prompt_ids

        rows.append(
            {
                "input_ids": prompt_ids,
                "chosen": chosen_ids,
                "rejected": reject_ids,
                args.length_column_name: len(prompt_ids) + max(len(chosen_ids), len(reject_ids)),
            }
        )

    return _columnar(rows)


PROCESSOR_REGISTRY = {"dpo": dpo_processor}


def processing_datasets(
    train_args: "DPOScriptArguments",
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset | None, Dataset | None, Dataset | None]:
    def _pack(dataset: Dataset, desc: str) -> Dataset:
        """캐시된 map(packing 무관) 이후에 실행하는 packing 단계.

        1) prompt/chosen/reject → packed input_ids + position_ids + 실제 길이(P+C+R) 구성
        2) 한 그룹이 max_length 를 넘으면 제거(pack_dataset bfd truncate 가 pair 를 손상시키는 것 방지)
        3) trl.pack_dataset 로 여러 그룹을 max_length bin 으로 병합(length 기반 best-fit) → GPU 입력 효율↑
        input_ids/position_ids 가 함께 concat 되어 그룹 경계(position_id==0)가 보존된다.
        """

        def _build_packed(batch: Dict[str, list], length_column_name: str) -> Dict[str, list]:
            """prompt/chosen/reject 를 [prompt, chosen_tail, reject_tail] 단일 packed 행으로 변환.

            position_ids: prompt=0..P-1, chosen/reject 각각 P 부터 재시작(attention.py 캐너니컬 레이아웃).
            length: P+C+R(packed 행 전체 길이) → pack_dataset 이 이 길이로 bin을 max_length 에 근사시킨다.
            """
            out_ids, out_pos, out_len = [], [], []
            for prompt, chosen, reject in zip(batch["input_ids"], batch["chosen"], batch["rejected"]):
                prompt, chosen, reject = list(prompt), list(chosen), list(reject)
                P, C, R = len(prompt), len(chosen), len(reject)
                out_ids.append(prompt + chosen + reject)
                out_pos.append(list(range(P)) + list(range(P, P + C)) + list(range(P, P + R)))
                out_len.append(P + C + R)
            return {"input_ids": out_ids, "position_ids": out_pos, length_column_name: out_len}

        dataset = dataset.map(
            _build_packed,
            batched=True,
            batch_size=train_args.dataset_batch_size,
            num_proc=train_args.dataset_num_proc,
            remove_columns=dataset.column_names,
            fn_kwargs={"length_column_name": train_args.length_column_name},
            desc=f"{desc}: build packed rows",
        )
        dataset = dataset.filter(
            lambda lengths: [length <= train_args.max_length for length in lengths],
            input_columns=[train_args.length_column_name],
            num_proc=train_args.dataset_num_proc,
            batched=True,
            batch_size=train_args.dataset_batch_size,
            desc=f"{desc}: drop > max_length",
        )
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
        train_dataset = _pack(train_dataset, "Packing dataset")

    # collator 는 args.packing 하나로만 분기(train/eval 공용)하므로, packing 이면 valid 도 반드시
    # packed 해야 eval 시 _pack_collate 가 position_ids 를 찾을 수 있다(eval_packing 별도 토글 무의미).
    if valid_dataset is not None and train_args.packing:
        valid_dataset = _pack(valid_dataset, "Packing eval dataset")

    return train_dataset, valid_dataset, test_dataset
