import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE
from trl.data_utils import is_conversational


INPUT_KEYS = {"prompt", "question", "input", "instruction"}
OUTPUT_KEYS = {"completion", "answer", "label", "output", "response", "assistant"}
CONVERSATION_KEYS = {"conversations", "conversation", "messages"}


def get_sentencepiece_offset(tokenizer, input_ids) -> Tuple[list, str]:
    """slow(sentencepiece) tokenizer 용: 토큰을 디코딩해 char offset을 직접 계산한다.

    fast tokenizer는 `return_offsets_mapping`을 지원하지만, slow tokenizer는
    `char_to_token`/`return_offsets_mapping`이 `ValueError`를 던지므로 이 fallback이 필요하다.
    """
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


def _fix_trailing_labels(labels: list, input_ids: list) -> list:
    """마지막 턴의 종료 토큰(EOS 등)까지 학습하도록 뒤쪽 -100 꼬리를 복원한다."""
    if labels and labels[-1] == -100:
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] != -100:
                break
            labels[i] = input_ids[i]
    return labels


def _assistant_text(content: Any) -> str:
    """assistant 턴 content를 검색용 문자열로 변환 (멀티모달이면 text 파트만)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return str(content)


def _find_from(text: str, needle: str, cursor: int, tokenizer=None) -> Tuple[int, str]:
    """cursor 이후에서 needle 을 찾는다. 실패 시 encode→decode round-trip 문자열로 재시도.

    반환: (start_idx, 실제로 매칭된 문자열). 못 찾으면 (-1, needle).
    """
    start_idx = text.find(needle, cursor)
    if start_idx != -1:
        return start_idx, needle
    if tokenizer is not None:
        # 원본 문자열은 encode→decode 정규화(zero-width/special char 등) 때문에
        # decode 공간인 `text` 와 다를 수 있다. round-trip 한 문자열로 다시 찾는다.
        needle_rt = tokenizer.decode(tokenizer(needle, add_special_tokens=False).input_ids)
        rt_idx = text.find(needle_rt, cursor)
        if rt_idx != -1:
            return rt_idx, needle_rt
    return -1, needle


def _labels_via_offsets(enc, text: str, conversations: list, input_ids: list, tokenizer=None) -> list:
    """fast tokenizer: char_to_token 으로 assistant 구간 → labels.

    턴을 순서대로 순회하며 cursor 를 전진시켜, 각 턴의 content 를 이전 턴 이후에서만
    찾는다. 이렇게 하면 assistant 답변이 user prompt 의 문구를 그대로 인용해도
    prompt 구간의 이른 occurrence 에 잘못 매칭되지 않는다(prompt-echo 방지).
    """
    labels = [-100] * len(input_ids)
    cursor = 0
    for chat in conversations:
        content = _assistant_text(chat["content"])
        start_idx, content = _find_from(text, content, cursor, tokenizer)
        if start_idx == -1:
            # assistant 턴은 라벨링에 필수 → 못 찾으면 에러. user 턴은 anchor 실패해도 진행.
            if chat["role"] == "assistant":
                raise ValueError(f"tokenizer is weird. cannot find assistant content: {content} > {text}")
            continue
        end_idx = start_idx + len(content)
        cursor = end_idx  # 다음 턴 검색은 이 위치 이후부터 → prompt-echo 오매칭 방지

        if chat["role"] != "assistant":
            continue

        token_start = enc.char_to_token(start_idx)
        # 종료 토큰(턴 terminator)까지 포함해 stop을 학습. 마지막 글자면 종료 토큰이 없을 수 있음.
        token_end = enc.char_to_token(end_idx)
        if token_end is None:
            token_end = enc.char_to_token(end_idx - 1)

        if token_start is not None and token_end is not None:
            labels[token_start : token_end + 1] = input_ids[token_start : token_end + 1]

    return _fix_trailing_labels(labels, input_ids)


def strip_conversation(conversations: list) -> list:
    """conversation 의 모든 메시지 content 앞/뒤 공백을 제거한 새 list 를 반환한다."""

    def _strip_message_content(content: Any) -> Any:
        """메시지 content 의 앞/뒤 공백을 제거한다(멀티모달 list 면 text part 만 strip).

        대부분의 chat template(예: Gemma)은 각 turn content 의 앞/뒤 공백을
        렌더링 단계에서 strip 한 뒤 turn 종료 토큰을 붙인다. 따라서 원본 content 가
        공백으로 시작/끝나면 디코딩된 text 의 substring 으로 더 이상 존재하지 않아
        `_find_from`(char offset 매칭) 이 실패한다. 들어가기 전에 동일하게 strip 해서
        원본 content 와 디코딩 text 의 공백 불일치를 제거한다.
        """
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            out = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append({**part, "text": part["text"].strip()})
                else:
                    out.append(part)
            return out
        return content

    return [{**m, "content": _strip_message_content(m.get("content", ""))} for m in conversations]


def has_trainable_assistant(conversations: list) -> bool:
    """학습 가능한(빈 문자열이 아닌) assistant turn 이 하나라도 있는지 확인한다.

    chosen/reject/answer 가 빈 문자열이면 라벨링할 구간이 없어 boundary 가 무너지고
    prompt 구간 불일치 등 degenerate 결과를 낳으므로, 호출부에서 미리 건너뛰는 데 쓴다.
    """
    for m in conversations:
        if m.get("role") == "assistant" and _assistant_text(m.get("content", "")).strip():
            return True
    return False


def create_assistant_labels(tokenizer, conversations, images=None) -> Tuple[list, Any]:
    """Assistant 응답 구간만 학습하도록 레이블 생성.

    - fast tokenizer: `return_offsets_mapping` + `char_to_token` (네이티브, 빠름)
    - slow tokenizer: `get_sentencepiece_offset` 기반 수동 매칭 (기존 동작 보존)

    NOTE: chat template 의 공백 strip 동작과 어긋나 char offset 매칭이 실패하는 것을
    막기 위해, 렌더링 전에 모든 메시지 content 를 strip 한다(SFT/DPO 공통 적용).
    """

    def _labels_via_sentencepiece(tokenizer, input_ids: list, conversations: list) -> list:
        """slow tokenizer: 디코딩-공간 문자열 매칭으로 assistant 구간 → labels (기존 방식 유지)."""
        offset_ls, text = get_sentencepiece_offset(tokenizer, input_ids)

        answer_ls = [chat["content"] for chat in conversations if chat["role"] == "assistant"]
        answer_ids_list = tokenizer(answer_ls, add_special_tokens=False).input_ids
        all_answer_tokens = ["".join(tokenizer.batch_decode([[tid] for tid in ids])) for ids in answer_ids_list]

        answer_pos_ls = []
        for new_answer in all_answer_tokens:
            start_idx = text.find(new_answer)
            if start_idx == -1:
                raise ValueError(f"tokenizer is weird. cannot find assistant content: {new_answer} > {text}")
            answer_pos_ls.append((start_idx, start_idx + len(new_answer) + 1))

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

        return _fix_trailing_labels(labels, input_ids)

    if len(conversations) % 2:
        raise ValueError("Conversations should have an even number of messages (user starts, assistant ends).")

    conversations = strip_conversation(conversations)
    rendered = tokenizer.apply_chat_template(conversations, tokenize=False)
    outputs = tokenizer(
        **{"text": rendered, **({"images": images} if images else {})},
        return_attention_mask=False,
        add_special_tokens=False,
    )

    text = tokenizer.decode(outputs.input_ids)
    if tokenizer.bos_token and not text.strip().startswith(tokenizer.bos_token):
        text = tokenizer.bos_token + text
    if tokenizer.eos_token and not text.strip().endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    if tokenizer.is_fast and not images:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        outputs.update({"input_ids": input_ids})
        labels = _labels_via_offsets(enc, text, conversations, input_ids, tokenizer)
    else:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        outputs.update({"input_ids": input_ids})
        labels = _labels_via_sentencepiece(tokenizer, input_ids, conversations)

    return labels, outputs


def _part_text(part: Any) -> str:
    """멀티모달 part 하나에서 텍스트만 추출."""
    if isinstance(part, str):
        return part.strip()
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        return part["text"].strip()
    return str(part)


def _normalize_content(content: Any, has_images: bool) -> Any:
    """has_images=True 면 part 리스트(list[dict]), False 면 텍스트 문자열로 정규화."""
    if not has_images:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(t for t in (_part_text(p) for p in content) if t)
        if isinstance(content, dict):
            text = content.get("text")
            return text.strip() if isinstance(text, str) else str(content)
        return str(content)

    # has_images=True → list[dict]
    if isinstance(content, str):
        s = content.strip()
        if s[:1] in ("{", "["):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass
        return [{"type": "text", "text": s}]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [p if isinstance(p, dict) else {"type": "text", "text": str(p)} for p in content]
    return [{"type": "text", "text": str(content)}]


def _to_conversational_list(row: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    # 1) conversations/conversation/messages 컬럼 (role+content 형식)
    for k in CONVERSATION_KEYS:
        if k in row:
            val = row[k]
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], dict)
                and "role" in val[0]
                and "content" in val[0]
            ):
                return val

    # 2) TRL식 conversational 감지 (prompt/messages 등)
    if is_conversational(row):
        if "messages" in row and isinstance(row["messages"], list):
            return row["messages"]
        prompt = row.get("prompt")
        answer = row.get("completion") or row.get("chosen") or row.get("answer") or row.get("response")
        if isinstance(prompt, (str, list, dict)):
            messages = [{"role": "user", "content": prompt}]
            if answer is not None:
                messages.append({"role": "assistant", "content": answer})
            return messages

    # 3) 간단한 QA/INSTRUCTION pair 패턴
    input_text = None
    output_text = None
    for k in INPUT_KEYS:
        if k in row and isinstance(row[k], (str, list, dict)):
            input_text = row[k]
            break
    for k in OUTPUT_KEYS:
        if k in row and isinstance(row[k], (str, list, dict)):
            output_text = row[k]
            break
    if input_text is not None:
        messages = [{"role": "user", "content": input_text}]
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        return messages

    return None


def _iter_rows(example: Dict[str, list]):
    keys = list(example.keys())
    for values in zip(*(example[k] for k in keys)):
        yield dict(zip(keys, values))


def _columnar(rows: List[dict]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for row in rows:
        for key, value in row.items():
            out.setdefault(key, []).append(value)
    return out


def _load_repo_datasets(repo_name: str, train_args, truncate_map: Dict[str, int]) -> Tuple[DatasetDict, str]:
    """
    1. save_to_disk 로 저장된 로컬 데이터 (state.json / dataset_dict.json 존재)
    2. 로컬 파일 (data_files_map)
    3. HuggingFace Hub 단일 subset
    4. HuggingFace Hub 복수 subset (list) → subset별 truncate 후 split 단위 병합
    """
    truncate_map = truncate_map or {}
    path = Path(repo_name)

    if path.exists() and ((path / "state.json").exists() or (path / "dataset_dict.json").exists()):
        loaded = load_from_disk(repo_name)
        # save_to_disk 로 저장된 단일 Dataset(스플릿 없음)은 DatasetDict({"train": ...}) 로 감싼다.
        if isinstance(loaded, Dataset):
            loaded = DatasetDict({"train": loaded})
        return loaded, path.name

    if path.exists() and train_args.dataset_files_map:
        data_files = train_args.dataset_files_map.get(repo_name)
        return load_dataset(repo_name, data_files=data_files), path.name

    raw_names = train_args.dataset_name_map.get(repo_name)
    data_names = raw_names if isinstance(raw_names, list) else [raw_names]

    if len(data_names) == 1:
        datasets = load_dataset(repo_name, data_names[0])
        data_name = str(data_names[0]) if data_names[0] is not None else ""
        return datasets, data_name

    truncated = []
    for name in data_names:
        subset = load_dataset(repo_name, name)
        key = f"{repo_name}-{name}"
        if key in truncate_map:
            n = truncate_map[key]
            subset = DatasetDict(
                {
                    split: (d.shuffle(seed=train_args.seed).select(range(n)) if len(d) > n else d)
                    for split, d in subset.items()
                }
            )
        truncated.append(subset)

    all_splits = set().union(*[ds.keys() for ds in truncated])
    datasets = DatasetDict(
        {split: concatenate_datasets([ds[split] for ds in truncated if split in ds]) for split in all_splits}
    )
    return datasets, "+".join(str(n) for n in data_names)


def _get_cache_dir(train_args, repo_name, data_name, splits, truncate_map) -> Tuple[Dict[str, str], Dict[str, str]]:
    """map 캐시는 모델/repo에만 의존(max_length 미포함), filter 캐시는 max_length 포함.

    → max_length 만 바꿔 재실험할 때 비싼 토큰화(map)는 재사용하고 filter만 다시 돈다.
    명시적 캐시명을 쓰는 이유: train_args(accelerate 객체 포함)는 datasets fingerprint 해싱이
    불안정해 random fingerprint 로 떨어지면 매 실행 재전처리되기 때문.
    """
    truncate_map = truncate_map or {}
    model_name = Path(train_args.model_name_or_path).name
    repo = Path(repo_name).name
    run_name = train_args.run_name.replace("/", "_")

    cache_dir = HF_DATASETS_CACHE / "preprocess_cache" / model_name / run_name / repo
    cache_dir.mkdir(parents=True, exist_ok=True)

    map_cache = {split: (cache_dir / f"map_{data_name}-{split}_preprocessor.arrow").as_posix() for split in splits}

    filter_cache = {}
    for split in splits:
        prefix = f"{truncate_map[split]}-" if split in truncate_map else ""
        filter_cache[split] = (
            cache_dir / f"filter_{prefix}{train_args.max_length}_{data_name}-{split}_preprocessor.arrow"
        ).as_posix()

    return map_cache, filter_cache


def _cache_exists(cache_file_names: Dict[str, str]) -> bool:
    """datasets 내부(`glob.iglob(prefix*ext)`)와 동일한 방식으로 캐시(샤드 포함) 존재를 확인.

    수동으로 `_{rank:05d}_of_{num_proc:05d}` 를 세지 않으므로 num_proc 가 바뀌어도 견고하다.
    오직 '첫 실행만 main_process_first' 결정을 위해서만 사용된다 (캐시 로드는 datasets가 담당).
    """
    if not cache_file_names:
        return False
    for path in cache_file_names.values():
        base, ext = os.path.splitext(path)
        if not glob.glob(f"{base}*{ext}"):
            return False
    return True


def _role_of(prefix: Dict[str, list], split_key: str) -> str | None:
    for role, splits in prefix.items():
        if split_key in splits:
            return role
    return None


def _merge_datasets(dataset_list: List[Dataset], name: str, packing: bool) -> Dataset | None:
    if not dataset_list:
        return None
    combined = concatenate_datasets(dataset_list)
    if not packing:
        combined.set_format("pt")
    return combined
