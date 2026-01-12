import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import optimization
import torch
import torch.nn as nn
from datasets import Dataset
from preprocessor import PROCESSOR_REGISTRY, processing_datasets
from setproctitle import setproctitle
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import transformers
from transformers import AutoConfig, AutoProcessor, set_seed
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_pt_utils import get_model_param_count


# TrainingArguments, ModelArguments, DataArguments 이런식으로 나누면, args관리하기도 어렵고, wandb에는 TrainingArguments만 기록되는 문제가 있기 때문에 이런식으로 상속받아서 하나의 train_args에서 모든 것을 처리할 수 있게 만들었음.
@dataclass
class SFTScriptArguments(SFTConfig, ModelConfig):
    _VALID_DICT_FIELDS = SFTConfig._VALID_DICT_FIELDS + [
        "data_truncate_map",
        "data_name_map",
        "config_kwargs",
        "tokenizer_kwargs",
        "dataset_prefix",
    ]
    # -------------------------- Datasets Args ------------------------- #
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={
            "help": "학습에 사용할 데이터셋의 hub_repo 이름을 리스트로 입력한다. 예: ['hf_repo1', 'hf_repo2']. datasets 4.0.0 이라면 remote_code는 동작하지 않기 때문에 이 점을 알고 사용해야 한다."
        },
    )
    data_preprocessor_type: str = field(
        default_factory=lambda: PROCESSOR_REGISTRY.keys(),
        metadata={
            "help": f"학습할 데이터의 전처리를 어떻게 할지 결정하는 값, {', '.join(PROCESSOR_REGISTRY.keys())} 를 지원한다."
        },
    )
    preprocessing_num_workers: int = field(
        default=5,
        metadata={"help": "데이터 전처리에 활용할 프로세스의 수, 1 이면 싱글 프로세스로 동작한다."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "데이터 전처리 시 사용할 배치 크기를 설정하는 값."},
    )
    dataset_prefix: dict = field(
        default_factory=lambda: defaultdict(list),
        metadata={
            "help": "데이터셋 로드 시 학습, 검증, 테스트 데이터를 구분하기 위해 사용되는 접두어 딕셔너리. 각 키는 'train', 'valid', 'test'로 구성된다."
        },
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": "데이터의 샘플 개수를 조절하기 위한 맵. 예: {'repo_name': {'train': 3000, 'validation': 1500}}. 데이터셋 처리 시 활용된다."
        },
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": "데이터셋의 구성 이름을 매핑하기 위한 맵. 예: {'repo_name': 'data_config_name'}. 데이터셋 로드 시 사용된다."
        },
    )

    # -------------------------- Training Args ------------------------- #

    lr_scheduler_type: Union[optimization.NewSchedulerType, str] = field(default="linear")

    chat_template: str = field(
        default=None,
        metadata={"help": "The template for chat interactions."},
    )

    config_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={"help": ""},
    )

    tokenizer_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={"help": ""},
    )

    def __post_init__(self) -> None:
        # train_args 조정 및 필요한 값들 설정
        self.config_kwargs = {
            **self.config_kwargs,
            "attn_implementation": self.attn_implementation,
            "use_cache": not self.gradient_checkpointing,
        }

        quantization_config = get_quantization_config(self)
        self.model_init_kwargs = {
            **(self.model_init_kwargs or {}),
            "revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": quantization_config,
            "device_map": get_kbit_device_map() if quantization_config is not None else None,
            "torch_dtype": self.dtype if self.dtype in ["auto", None] else getattr(torch, self.dtype),
        }
        self.tokenizer_kwargs = {
            **(self.tokenizer_kwargs or {}),
            "revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.chat_template:
            self.tokenizer_kwargs["chat_template"] = self.chat_template

        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

        if self.group_by_length:
            logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")


class PackingCollatorForLLM(DataCollatorMixin):
    def __init__(
        self,
        args: SFTScriptArguments,
        model: nn.Module,
        tokenizer: Union[AutoProcessor, transformers.PreTrainedTokenizer],
        return_tensors: Optional[str] = "pt",
        sample_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Args:
            args (`TrainingArguments`):
                현재 학습 상태 및 설정을 확인하기 위한 값
            model (`nn.Module`):
                현재 학습 중인지 아닌지 확인하기 위한 값
            processor (`ProcessorMixin` or `PreTrainedTokenizer`):
                입력받은 데이터를 pad처리 하거나, 추가적인 전처리를 진행하기 위해 사용하는 프로세서
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                입력받은 값은 pt, tf, np 중 하나로 변환. 기본값은 "pt"입니다.
            sample_dataset (`Dataset`, *optional*):
                전처리가 끝난 샘플 데이터셋. 이 데이터셋을 통해 입력값이 올바르게 처리되는지 확인.
                만약 제공되지 않는다면, 입력값이 올바르게 처리되는지 확인하는 과정은 생략.
                주호 학습 전에 Collator가 정상 동작 하는지와, BOS, EOS 토큰이 올바르게 학습 데이터에 포함되어 있는지 확인하는 용도로 사용된다.
        """
        self.args = args
        self.model = model
        self.return_tensors = return_tensors
        self.tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

        self.process_type = args.data_preprocessor_type

        if sample_dataset is not None and self.args.distributed_state.is_local_main_process:
            sample = sample_dataset[0]
            sample_check = self([sample])

            input_ids, labels = sample_check["input_ids"].tolist()[0], sample_check["labels"]
            labels = labels[labels != -100].tolist()

            str_labels = [self.tokenizer.convert_ids_to_tokens(token) for token in labels]
            str_input_ids = [self.tokenizer.convert_ids_to_tokens(token) for token in input_ids]

            logger.info(f"\nlabel-values: [{', '.join(str_labels)}]\ninput-values: [{', '.join(str_input_ids)}]\n")

            if self.tokenizer.bos_token_id and self.tokenizer.bos_token_id not in input_ids:
                raise ValueError("BOS 토큰이 데이터에서 검출되지 않는다. 전처리가 다시 필요하다.")
            if self.tokenizer.eos_token_id not in input_ids:
                raise ValueError("EOS 토큰이 데이터에서 검출되지 않는다. 전처리가 다시 필요하다.")

            if self.model.config._attn_implementation == "eager" and self.args.spfhp_packing:
                msg = "attention implementation이 eager인데, packing을 사용하고 있다. flash attention으로 변경해라."
                raise ValueError(msg)

    def _pack_collate(self, features_ls: List[List[dict]]) -> dict:
        if features_ls and isinstance(features_ls[0], dict):
            features_ls = [features_ls]

        input_ids_ls, labels_ls, position_ids_ls, input_length_ls = [], [], [], []
        for features in features_ls:
            for feature in features:
                length = len(feature["input_ids"])
                input_ids_ls.append(feature["input_ids"])
                labels_ls.append(feature["labels"] if self.process_type != "pretrain" else feature["input_ids"])
                position_ids_ls.append(torch.arange(length))
                input_length_ls.append(length)

        batch = {
            "input_ids": torch.cat(input_ids_ls)[None],
            "labels": torch.cat(labels_ls)[None],
            "position_ids": torch.cat(position_ids_ls)[None],
        }

        return batch

    def _pad_collate(self, features_ls: Union[List[dict], List[List[Dict]]]) -> dict:
        def flatten(features_ls):
            return [
                feature
                for features in features_ls
                for feature in (features if isinstance(features, list) else [features])
            ]

        feature_ls = flatten(features_ls)
        input_ids_features = [{"input_ids": feature["input_ids"]} for feature in feature_ls]
        labels_features = [
            {"input_ids": feature["labels"] if self.process_type != "pretrain" else feature["input_ids"]}
            for feature in feature_ls
        ]

        input_output = self.tokenizer.pad(input_ids_features, padding_side="left", return_tensors="pt")
        labels_output = self.tokenizer.pad(labels_features, padding_side="left", return_tensors="pt")

        batch = {
            "input_ids": input_output.input_ids,
            "labels": labels_output.input_ids,
            "attention_mask": input_output.attention_mask,
        }
        return batch

    def torch_call(self, features_ls: Union[List[dict], List[List[dict]]]) -> dict:
        use_packing = getattr(self.args, "spfhp_packing", False)
        if use_packing and self.model.training:
            return self._pack_collate(features_ls)
        else:
            return self._pad_collate(features_ls)


logger = transformers.utils.logging.get_logger("transformers")


def main(train_args: SFTScriptArguments) -> None:
    processor = AutoProcessor.from_pretrained(train_args.model_name_or_path, **train_args.tokenizer_kwargs)
    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **train_args.config_kwargs)

    train_dataset, valid_dataset, test_dataset = processing_datasets(train_args, processor)

    # 학습에 활용할 값들을 로딩
    model_kwargs = {"config": config, **train_args.model_init_kwargs}
    architecture = getattr(transformers, config.architectures[0].replace("FSDP", ""))
    model = architecture.from_pretrained(train_args.model_name_or_path, **model_kwargs).train()
    model.use_cache = False if train_args.gradient_checkpointing else True
    logger.info(f"Model parameter count: {get_model_param_count(model)}")

    # _no_split_modules에 model에 존재하지 않는 모듈이 있는 경우 FSDP에서 애러가 발생하기 때문에, 필터링이 필요로 함.
    exist_module = {module.__class__.__name__ for module in model.modules()}
    model._no_split_modules = list(set(model._no_split_modules).intersection(exist_module))

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    collator = PackingCollatorForLLM(
        args=train_args,
        model=model,
        tokenizer=processor,
        sample_dataset=train_dataset or valid_dataset or test_dataset,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if train_args.eval_strategy != "no" else None,
        processing_class=processor,
        data_collator=collator,
        args=train_args,
        peft_config=get_peft_config(train_args),
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: SFTTrainer, args: SFTScriptArguments) -> None:
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.log_metrics("train", outputs.metrics)
    trainer.save_metrics("train", outputs.metrics)


@torch.no_grad()
def valid(trainer: SFTTrainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    metrics = trainer.evaluate(valid_datasets)
    metrics = {key: obj for key, obj in metrics.items() if type(obj).__module__ == "builtins"}

    trainer.log_metrics("valid", metrics)
    trainer.save_metrics("valid", metrics)


if "__main__" in __name__:
    parser = TrlParser([SFTScriptArguments])
    train_args, remain_args = parser.parse_args_and_config(return_remaining_strings=True)

    if remain_args and train_args.distributed_state.is_local_main_process:
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    main(train_args)
