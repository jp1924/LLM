import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import ProfileKwargs
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from torch.utils.data import DataLoader, RandomSampler, Sampler
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, is_main_process, seed_worker
from transformers.utils import is_datasets_available


@dataclass
class SFTTrainingArguments(Seq2SeqTrainingArguments):
    dataset_repo_ls: List[str] = field(
        default=None,
        metadata={"help": "The list of dataset repository names to use (via the datasets library)."},
    )
    data_max_length: int = field(
        default=2048,
        metadata={"help": "The maximum length of the data sequences."},
    )

    preprocessing_num_workers: int = field(
        default=5,
        metadata={"help": "The number of worker processes to use for data preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The batch size to use for data preprocessing."},
    )
    preprocessing_batched: bool = field(
        default=True,
        metadata={"help": "Whether to batch the data during preprocessing."},
    )
    train_dataset_prefix: List[str] = field(
        default="train",
        metadata={"help": "A prefix required to distinguish training splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": "A prefix required to distinguish validation splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": "A prefix required to distinguish test splits in the data loaded by load_dataset."},
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "A map to truncate part of the data. Example: {'repo_name': {'train': 3000, 'validation': 1500}}."
        },
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to config_name of the data. Example: {'repo_name': 'data_config_name'}."},
    )
    do_data_main_process_first: bool = field(
        default=False,
        metadata={"help": "Whether to run data preprocessing on the main process first."},
    )
    profiling: bool = field(
        default=False,
        metadata={"help": "Whether to enable profiling during training."},
    )
    sot_token: str = field(
        default="",
        metadata={"help": "The start of text token."},
    )
    eot_token: str = field(
        default="",
        metadata={"help": "The end of text token."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to store the cache files."},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The name or path of the pre-trained model."},
    )
    response_template: str = field(
        default=None,
        metadata={"help": "The template for generating responses."},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "The template for generating instructions."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "The attention implementation to use. Options: 'eager', 'flash_attention_2'."},
    )
    padding_side: str = field(
        default="right",
        metadata={"help": "The side on which to pad sequences. Options: 'left', 'right'."},
    )

    chat_template: str = field(
        default=None,
        metadata={"help": "The template for chat interactions."},
    )
    packing_max_elem: int = field(
        default=10,
        metadata={"help": "The maximum number of elements to pack together."},
    )
    do_packing: bool = field(
        default=False,
        metadata={"help": "Whether to enable packing of sequences."},
    )
    packing_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle sequences during packing."},
    )

    do_logic_kor_at_save: bool = field(
        default=False,
        metadata={"help": "Whether to enable logic_kor evaluation at each save."},
    )
    judge_model: str = field(
        default=None,
        metadata={"help": "The name or path of the judge model."},
    )
    judge_repeat_num: int = field(
        default=5,
        metadata={"help": "The number of times to repeat the evaluation."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.data_truncate_map = json.loads(self.data_truncate_map) if self.data_truncate_map else {}
        self.data_name_map = json.loads(self.data_name_map) if self.data_name_map else {}
        self.response_template = json.loads(self.response_template) if self.response_template else None
        self.instruction_template = json.loads(self.instruction_template) if self.instruction_template else None

        self.train_dataset_prefix = self.train_dataset_prefix if self.train_dataset_prefix else []
        self.valid_dataset_prefix = self.valid_dataset_prefix if self.valid_dataset_prefix else []
        self.test_dataset_prefix = self.test_dataset_prefix if self.test_dataset_prefix else []

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None

        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

        if self.group_by_length:
            logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")

        if self.do_logic_kor_at_save:
            if self.judge_model is None:
                raise ValueError("judge_model이 없음. 다시 설정하셈.")

            if os.getenv("OPENAI_API_KEY", None) is None:
                raise ValueError("OPENAI_API_KEY가 없음. 다시 설정하셈.")

            if self.report_to is not None and "wandb" not in self.report_to:
                raise ValueError("do_logic_kor_at_save에선 wandb만 지원함. 다시 설정하셈.")


class PackingSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
        do_shuffle: bool = False,
    ):
        self.dataset = dataset

        self.packing_strategies = self._get_packing_strategies(
            lengths=lengths,
            max_seq_len=max_seq_len,
            max_seq_per_pack=max_seq_per_pack,
        )

        self.do_shuffle = do_shuffle
        self.lengths = lengths

        self.packing_sample_ls = self._transform_length_to_indices(
            strategies_per_length=self.packing_strategies,
            lengths=lengths,
        )

    def _get_packing_strategies(
        self,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
    ) -> dict:
        def add_pack(
            pack: List[int],
            count: int,
            tmp: defaultdict,
            final: defaultdict,
            limit: int,
            offset: int,
        ) -> None:
            if len(pack) == limit or offset == 0:
                final[offset].append((count, pack))
            else:
                tmp[offset].append((count, pack))

        seq_lens, counts = np.unique(lengths, return_counts=True)
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        histogram[seq_lens - 1] = counts

        reversed_histogram = np.flip(histogram)

        tmp_strategies_per_length = defaultdict(list)
        strategies_per_length = defaultdict(list)

        for i in range(max_seq_len):
            n_sequences_to_bin = reversed_histogram[i]
            length_to_bin = max_seq_len - i
            offset = i + 1  # largest possible offset
            while n_sequences_to_bin > 0:
                if (length_to_bin + offset) in tmp_strategies_per_length:
                    # extract shortest pack that will get modified
                    n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                    new_pack = pack + [length_to_bin]
                    count = min(n_sequences_to_pack, n_sequences_to_bin)
                    if n_sequences_to_pack > n_sequences_to_bin:
                        # old pack gets reduced
                        n_sequences_to_pack -= n_sequences_to_bin
                        tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                        n_sequences_to_bin = 0
                    else:
                        n_sequences_to_bin -= n_sequences_to_pack
                    add_pack(
                        new_pack, count, tmp_strategies_per_length, strategies_per_length, max_seq_per_pack, offset
                    )
                    # clean up to speed up main key search
                    if not tmp_strategies_per_length[length_to_bin + offset]:
                        tmp_strategies_per_length.pop(length_to_bin + offset)
                else:
                    offset -= 1
                # Does not fit anywhere. Create new pack.
                if offset < 0:
                    add_pack(
                        [length_to_bin],
                        n_sequences_to_bin,
                        tmp_strategies_per_length,
                        strategies_per_length,
                        max_seq_per_pack,
                        i,
                    )
                    n_sequences_to_bin = 0
        # merge all strategies
        for key in tmp_strategies_per_length:
            strategies_per_length[key].extend(tmp_strategies_per_length[key])

        return strategies_per_length

    def _transform_length_to_indices(self, strategies_per_length: dict, lengths: List[int]) -> List[List[int]]:
        length_to_indices = {}
        length_array = np.array(lengths)
        unique_lengths = np.unique(length_array).tolist()

        for length in unique_lengths:
            dataset_idx_ls = np.where(length_array == length)[0].tolist()
            if self.do_shuffle:
                random.shuffle(dataset_idx_ls)
            length_to_indices[length] = dataset_idx_ls

        pack_strategies_ls = [
            pack
            for strategies in strategies_per_length.values()
            for strategies_num, pack_strategies in strategies
            for pack in ([pack_strategies] * strategies_num)
        ]

        packing_sample_ls = list()
        for pack_strategies in pack_strategies_ls:
            pack_size = len(pack_strategies)
            strategie_position = 0

            dataset_idx_ls = list()
            while strategie_position + 1 <= pack_size:
                length = pack_strategies[strategie_position]
                pack_length_ls = length_to_indices[length]
                dataset_idx_ls.append(pack_length_ls.pop())
                length_to_indices[length] = pack_length_ls
                strategie_position += 1

            packing_sample_ls.append(dataset_idx_ls)

        if self.do_shuffle:
            random.shuffle(packing_sample_ls)

        return packing_sample_ls

    def __iter__(self):
        if self.do_shuffle:
            packing_sample_ls = self._transform_length_to_indices(
                strategies_per_length=self.packing_strategies,
                lengths=self.lengths,
            )
        else:
            packing_sample_ls = self.packing_sample_ls

        return iter(packing_sample_ls)

    def __len__(self):
        return len(self.packing_sample_ls)


class PackingTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        def __packing_getitems__(train_dataset, keys: List[List[int]]) -> List:
            """Can be used to get a batch using a list of integers indices."""

            return_ls = list()
            for key in keys:
                batch = train_dataset.__getitem__(key)
                n_examples = len(batch[next(iter(batch))])

                return_ls.append([{col: array[i] for col, array in batch.items()} for i in range(n_examples)])
            return return_ls

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # NOTE: packing을 사용할 경우 packing에 알맞은 getitems를 사용하도록 합니다.
        if self.args.do_packing:
            # 래핑된 함수를 정의하여 self를 전달할 수 있도록 합니다.
            def getitems_wrapper(keys):
                return __packing_getitems__(train_dataset, keys)

            setattr(self.train_dataset, "__getitems__", getitems_wrapper)

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        self.args: SFTTrainingArguments

        if self.args.group_by_length and self.args.do_packing:
            raise ValueError("group_by_length and do_packing cannot be used together.")

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.do_packing:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None

            return PackingSampler(
                dataset=self.train_dataset,
                lengths=lengths,
                max_seq_len=self.args.data_max_length,
                max_seq_per_pack=self.args.packing_max_elem,
                do_shuffle=self.args.packing_shuffle,
            )

        else:
            return RandomSampler(self.train_dataset)


class DataPackingCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    def __init__(self, args: SFTTrainingArguments, dtype, **kwargs):
        super().__init__(**kwargs)
        self.pack_max_seq = args.data_max_length
        self.attn_implementation = args.attn_implementation
        self.args = args
        self.do_packing = args.do_packing
        self.dtype = dtype

    def _create_attention_mask(self, input_length_ls):
        total_length = sum(input_length_ls)
        attention_mask = torch.full((1, 1, total_length, total_length), torch.finfo(self.dtype).min)

        start_idx, end_idx = 0, 0
        for length in input_length_ls:
            end_idx += length
            one_tensor = torch.ones((length, length), dtype=torch.float32)
            mask = torch.tril(one_tensor, diagonal=0).to(dtype=torch.bool)
            attention_mask[0, 0, start_idx:end_idx, start_idx:end_idx][mask] = 0
            start_idx = end_idx

        return attention_mask

    def _process_features(self, features_ls):
        input_ids_ls, labels_ls, position_ids_ls, input_length_ls = list(), list(), list(), list()
        for features in features_ls:
            batch = super().torch_call([features])
            input_ids, labels = batch.input_ids[0], batch.labels[0]
            length = len(input_ids)

            labels_ls.append(labels)
            input_ids_ls.append(input_ids)
            input_length_ls.append(length)
            position_ids_ls.append(torch.arange(length))

        return input_ids_ls, labels_ls, position_ids_ls, input_length_ls

    def torch_call(self, features_ls):
        if not self.do_packing:
            input_ids_ls, labels_ls, position_ids_ls, input_length_ls = self._process_features(features_ls)
        elif isinstance(features_ls, list) and isinstance(features_ls[0], list):
            input_ids_ls, labels_ls, position_ids_ls, input_length_ls = list(), list(), list(), list()
            for packing_ls in features_ls:
                ids, labels, positions, lengths = self._process_features(packing_ls)
                input_ids_ls.extend(ids)
                labels_ls.extend(labels)
                position_ids_ls.extend(positions)
                input_length_ls.extend(lengths)

        attention_mask = self._create_attention_mask(input_length_ls)

        batch = {
            "labels": torch.concat(labels_ls)[None],
            "input_ids": torch.concat(input_ids_ls)[None],
            "position_ids": torch.concat(position_ids_ls)[None],
            "attention_mask": attention_mask,
        }

        return batch


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: SFTTrainingArguments) -> None:
    def preprocessor(example):
        finish_input_id_ls, finish_length_ls = list(), list()
        for conversations in example["conversations"]:
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                eot_token=train_args.eot_token,
                sot_token=train_args.sot_token,
            )
            is_multi_turn = len(conversations) > 2

            if tokenizer.bos_token not in text and not add_bos_token:
                text = f"{tokenizer.bos_token}{text}"
            elif not is_multi_turn and add_bos_token and text.startswith(tokenizer.bos_token):
                logger.warning_once(
                    "tokenizing하면서 bos토큰을 자동으로 추가해 주는데, chat_template 적용하면서 bos토큰이 자동으로 추가됨. 따라서 chat_template을 통해 추가된 bos토큰은 필터링 함."
                )
                text = text.replace(tokenizer.bos_token, "")
            elif is_multi_turn and add_bos_token and text.endswith(tokenizer.bos_token):
                raise ValueError("아직 구현 안함.")

            if tokenizer.eos_token not in text and not add_eos_token:
                text = f"{text}{tokenizer.eos_token}"
            elif not is_multi_turn and add_eos_token and text.endswith(tokenizer.eos_token):
                logger.warning_once(
                    "tokenizing하면서 eos토큰을 자동으로 추가해 주는데, chat_template 적용하면서 eos토큰이 자동으로 추가됨. 따라서 chat_template을 통해 추가된 eos토큰은 필터링 함."
                )
                text = text.replace(tokenizer.eos_token, "")
            elif is_multi_turn and add_eos_token and text.endswith(tokenizer.eos_token):
                raise ValueError("아직 구현 안함.")

            outputs = tokenizer(text, return_tensors="np", return_length=True)

            finish_input_id_ls.extend(outputs.input_ids)
            finish_length_ls.extend(outputs.length)

        return {
            "input_ids": finish_input_id_ls,
            train_args.length_column_name: finish_length_ls,
        }

    def processing_datasets() -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        def length_filter(length_ls: List[int]) -> List[bool]:
            return [length <= train_args.data_max_length for length in length_ls]

        def process_dataset(dataset, dataset_key, repo_name, truncate_map, filter_cache_file_name):
            original_size = len(dataset)
            if dataset_key in truncate_map:
                truncate_size = truncate_map[dataset_key]
                dataset_size = len(dataset)
                dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
                if dataset_size <= truncate_size and is_main_process(train_args.local_rank):
                    logger.info(
                        f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                    )

            if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                dataset = dataset.filter(
                    length_filter,
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

            if is_main_process(train_args.local_rank):
                length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
                length_ls = [int(length) for length in length_ls]
                logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")
                logger.info(f"{repo_name}/{dataset_key}-size: {original_size} -> {len(dataset)}")

        def concatenate_and_log(datasets_ls, dataset_type):
            if datasets_ls:
                dataset = concatenate_datasets(datasets_ls)
                dataset.set_format("pt")
                if is_main_process(train_args.local_rank):
                    logger.info(f"{dataset_type}_dataset:\n{dataset}")
                return dataset
            return None

        start_time = time.time()
        train_dataset_ls, valid_dataset_ls, test_dataset_ls = [], [], []
        for repo_name in train_args.dataset_repo_ls:
            if is_main_process(train_args.local_rank):
                logger.info(f"load-{repo_name}")

            data_name = train_args.data_name_map.get(repo_name, None)
            truncate_map = train_args.data_truncate_map.get(repo_name, {})
            datasets = load_dataset(repo_name, data_name)

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
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=map_cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=set(sum(datasets.column_names.values(), [])),
                desc=f"preprocess-{repo_name}",
            )

            for dataset_key in datasets:
                process_dataset(datasets[dataset_key], dataset_key, repo_name, truncate_map, filter_cache_file_name)

        train_dataset, valid_dataset, test_dataset = (
            concatenate_and_log(train_dataset_ls, "train"),
            concatenate_and_log(valid_dataset_ls, "valid"),
            concatenate_and_log(test_dataset_ls, "test"),
        )

        sample_dataset = train_dataset or valid_dataset or test_dataset
        if sample_dataset and is_main_process(train_args.local_rank):
            formated_instruct = tokenizer.decode(sample_dataset[0]["input_ids"], skip_special_tokens=False)
            logger.info(f"formated_instruct: {formated_instruct}")

            if train_args.response_template is not None:
                response_template = tokenizer.decode(train_args.response_template, skip_special_tokens=False)
                logger.info(f"response_template: {response_template}")
                if response_template not in formated_instruct:
                    raise ValueError("이거 response_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
            else:
                raise logger.error("response_template이 없음. 다시 설정하셈.")

            if train_args.instruction_template is not None:
                instruction_template = tokenizer.decode(train_args.instruction_template, skip_special_tokens=False)
                logger.info(f"instruction_template: {instruction_template}")
                if instruction_template not in formated_instruct:
                    raise ValueError(
                        "이거 instruction_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈"
                    )
            else:
                logger.warning("instruction_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")
        elif sample_dataset is None:
            logger.warning("train, valid, test데이터가 전혀 없는 상태인데 확인 한번 해봐.")

        end_time = time.time()
        if is_main_process(train_args.local_rank):
            logger.info(f"load_dataset_time: {end_time - start_time:.2f}")

        return train_dataset, valid_dataset, test_dataset

    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path, padding_side=train_args.padding_side)

    # NOTE: add bos, eos token이 있는녀석이 있고 없는 녀석이 있음.
    check_input_ids = tokenizer("안녕하세요").input_ids
    add_bos_token = check_input_ids[0] == tokenizer.bos_token_id
    if not add_bos_token and is_main_process(train_args.local_rank):
        logger.warning(
            "tokenizer에 add_bos_token이 False로 되어 있음. 전처리 시, bos토큰이 삽입되지 않을 가능성이 있음."
        )

    add_eos_token = check_input_ids[-1] == tokenizer.eos_token_id
    if not add_eos_token and is_main_process(train_args.local_rank):
        logger.warning(
            "tokenizer에 add_eos_token이 False로 되어 있음. 전처리 시, eos토큰이 삽입되지 않을 가능성이 있음."
        )

    if train_args.chat_template:
        logger.info(
            f"기존 tokenizer의 chat_template이 {tokenizer.chat_template} 이었음. 이걸 {train_args.chat_template}로 바꿈,"
        )
        tokenizer.chat_template = train_args.chat_template

    config = AutoConfig.from_pretrained(
        train_args.model_name_or_path,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation=train_args.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(train_args.model_name_or_path, config=config)

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    # load datasets
    context = (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    )
    with context:
        # load datasets
        train_dataset, valid_dataset, test_dataset = processing_datasets()

    collator = DataPackingCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=train_args.response_template,
        instruction_template=train_args.instruction_template,
        args=train_args,
        dtype=model.dtype,
    )

    sample_check = collator.torch_call([[train_dataset[0]]] if train_args.do_packing else [train_dataset[0]])
    if is_main_process(train_args.local_rank):
        sample_check["labels"] = sample_check["labels"][sample_check["labels"] != -100].tolist()
        check_labels = [tokenizer.convert_ids_to_tokens(token) for token in sample_check["labels"]]
        check_labels = ", ".join(check_labels)
        logger.info(f"collator_label: [-100,  ..., -100, {check_labels}]")

    if tokenizer.bos_token_ids not in sample_check["input_ids"].tolist()[0]:
        raise ValueError("BOS token이 없다. 이거 다시 전처리 해라.")

    if tokenizer.eos_token_ids not in sample_check["input_ids"].tolist()[0]:
        raise ValueError("EOS token이 없다. 이거 다시 전처리 해라.")

    if train_args.lr_scheduler_type == "better_cosine":
        from optimizer import get_better_cosine_schedule_with_warmup

        raise NotImplementedError("아직 구현 안함.")

    callbacks = None
    if train_args.do_logic_kor_at_save:
        from callbacks import LogicKorCallback

        callbacks = [LogicKorCallback()]

        logger.info("아직 구현중인 기능\n" * 10)

    trainer = PackingTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        args=train_args,
        callbacks=callbacks,
    )
    del model

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: Trainer, args: SFTTrainingArguments) -> None:
    profile_kwargs = ProfileKwargs(activities=["cpu", "cuda"], profile_memory=True, with_flops=True)
    context = trainer.accelerator.profile(profile_kwargs) if args.profiling else nullcontext()

    with context as prof:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    save_path = Path(args.output_dir)
    if prof:
        prof.export_memory_timeline(save_path.with_suffix(".memory_trace.json").as_posix())
        prof.export_chrome_trace(save_path.with_suffix(".chrome_trace.json").as_posix())
        print(prof.key_averages().table(sort_by="flops", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if "__main__" in __name__:
    parser = HfArgumentParser([SFTTrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
