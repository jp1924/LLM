import logging
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import datasets
import torch
from datasets import Dataset
from preprocessor import processing_datasets
from rewards import get_reward_func_ls
from setproctitle import setproctitle

# from trainer import CustomGRPOTrainer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import transformers
from transformers import AutoConfig, AutoProcessor, set_seed
from transformers.trainer_pt_utils import get_model_param_count


@dataclass
class DataScriptArguments:
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={
            "help": "학습에 사용할 데이터셋의 hub_repo 이름을 리스트로 입력한다. 예: ['hf_repo1', 'hf_repo2']. datasets 4.0.0 이라면 remote_code는 동작하지 않기 때문에 이 점을 알고 사용해야 한다."
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
    do_data_main_process_first: bool = field(
        default=False,
        metadata={
            "help": "데이터 전처리를 메인 프로세스에서 먼저 실행할지 여부를 결정하는 값. main_process_first 컨텍스트에서 사용된다."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "캐시 파일을 저장할 디렉토리를 설정하는 값. 데이터셋 로드 및 모델 저장 시 사용된다."},
    )


# NOTE: TrainingArguments, ModelArguments, DataArguments 이런식으로 나누면,
#       args관리하기도 어렵고, wandb에는 TrainingArguments만 기록되는 문제가 있기 때문에 이런식으로 상속받아서 하나의 train_args에서 모든 것을 처리할 수 있게 만들었음.
@dataclass
class GRPOScriptArguments(GRPOConfig, DataScriptArguments, ModelConfig):
    _VALID_DICT_FIELDS = GRPOConfig._VALID_DICT_FIELDS + [
        "data_truncate_map",
        "data_name_map",
        "config_kwargs",
        "tokenizer_kwargs",
        "dataset_prefix",
    ]

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
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"help": "System prompt for the model."},
    )


logger = transformers.utils.logging.get_logger("transformers")


def main(train_args: GRPOScriptArguments) -> None:
    # NOTE: dataset 전처리를 위한 값들을 우선 로딩
    processor = AutoProcessor.from_pretrained(train_args.model_name_or_path, **train_args.tokenizer_kwargs)
    config_kwargs = {
        **train_args.config_kwargs,
        "bos_token_id": processor.bos_token_id,
        "eos_token_id": processor.eos_token_id,
        "pad_token_id": processor.pad_token_id,
    }
    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **config_kwargs)

    if not processor.chat_template:
        raise ValueError("chat template을 설정하지 않으면 코드가 정상적으로 동작하지 않는다.")

    with (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    ):
        train_dataset, valid_dataset, test_dataset = processing_datasets(train_args, processor)

    # NOTE: 학습에 활용할 값들을 로딩
    model_kwargs = {"config": config, **train_args.model_init_kwargs}
    architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(train_args.model_name_or_path, **model_kwargs).train()
    model.use_cache = False if train_args.gradient_checkpointing else True
    logger.info(f"Model parameter count: {get_model_param_count(model)}")
    train_args.model_init_kwargs = None

    # NOTE: _no_split_modules에 model에 존재하지 않는 모듈이 있는 경우 FSDP에서 애러가 발생하기 때문에, 필터링이 필요로 함.
    exist_module = {module.__class__.__name__ for module in model.modules()}
    model._no_split_modules = list(set(model._no_split_modules).intersection(exist_module))

    trainer = GRPOTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if train_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(train_args),
        reward_funcs=get_reward_func_ls(train_args),
        processing_class=processor,
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: GRPOTrainer, args: GRPOScriptArguments) -> None:
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.log_metrics("train", outputs.metrics)
    trainer.save_metrics("train", outputs.metrics)


@torch.no_grad()
def valid(trainer: GRPOTrainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    metrics = trainer.evaluate(valid_datasets)
    metrics = {key: obj for key, obj in metrics.items() if type(obj).__module__ == "builtins"}

    trainer.log_metrics("valid", metrics)
    trainer.save_metrics("valid", metrics)


if "__main__" in __name__:
    parser = TrlParser([GRPOScriptArguments])
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

    # NOTE: train_args 조정 및 필요한 값들 설정
    train_args.config_kwargs = {
        **train_args.config_kwargs,
        "attn_implementation": train_args.attn_implementation,
        "use_cache": not train_args.gradient_checkpointing,
    }

    quantization_config = get_quantization_config(train_args)
    train_args.model_init_kwargs = {
        **(train_args.model_init_kwargs or {}),
        "revision": train_args.model_revision,
        "trust_remote_code": train_args.trust_remote_code,
        "quantization_config": quantization_config,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "torch_dtype": train_args.torch_dtype
        if train_args.torch_dtype in ["auto", None]
        else getattr(torch, train_args.torch_dtype),
    }
    train_args.tokenizer_kwargs = {
        **(train_args.tokenizer_kwargs or {}),
        "revision": train_args.model_revision,
        "trust_remote_code": train_args.trust_remote_code,
    }
    if train_args.chat_template:
        train_args.tokenizer_kwargs["chat_template"] = train_args.chat_template

    train_args.cache_dir = Path(train_args.cache_dir) if train_args.cache_dir else None
    train_args.model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path

    if train_args.group_by_length:
        logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")

    main(train_args)
