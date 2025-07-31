import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import datasets
import optimization
import torch
from datasets import Dataset
from metrics import METRICS_REGISTRY
from preprocessor import PROCESSOR_REGISTRY, processing_datasets
from setproctitle import setproctitle
from trainer import PackingCollatorForLLM, PackingTrainer
from trl import ModelConfig, SFTConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)
from transformers.trainer_pt_utils import get_model_param_count


@dataclass
class DataScriptArguments:
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of dataset repository names to use (via the datasets library)."},
    )
    data_preprocessor_type: str = field(
        default_factory=lambda: PROCESSOR_REGISTRY.keys(),
        metadata={"help": "preprocessor type"},
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
        default_factory=list,
        metadata={"help": "A prefix required to distinguish training splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish validation splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish test splits in the data loaded by load_dataset."},
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={
            "help": "A map to truncate part of the data. Example: {'repo_name': {'train': 3000, 'validation': 1500}}."
        },
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "A map to config_name of the data. Example: {'repo_name': 'data_config_name'}."},
    )
    do_data_main_process_first: bool = field(
        default=False,
        metadata={"help": "Whether to run data preprocessing on the main process first."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to store the cache files."},
    )


@dataclass
class SFTScriptArguments(SFTConfig, DataScriptArguments, ModelConfig):
    lr_scheduler_type: Union[optimization.NewSchedulerType, str] = field(default="linear")

    chat_template: str = field(
        default=None,
        metadata={"help": "The template for chat interactions."},
    )
    packing_max_elem: int = field(
        default=10,
        metadata={"help": "The maximum number of elements to pack together."},
    )

    config_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )

    tokenizer_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

    def __post_init__(self):
        if self.output_dir is None:
            raise ValueError("output_dir은 무조건 설정되어 있어야 한다.")

        super().__post_init__()

        def _convert_str_dict(passed_value: dict):
            "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
            for key, value in passed_value.items():
                if isinstance(value, dict):
                    passed_value[key] = _convert_str_dict(value)
                elif isinstance(value, str):
                    # First check for bool and convert
                    if value.lower() in ("true", "false"):
                        passed_value[key] = value.lower() == "true"
                    # Check for digit
                    elif value.isdigit():
                        passed_value[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        passed_value[key] = float(value)

            return passed_value

        _ADDITIONAL_VALID_DICT_FILEDS = [
            "data_truncate_map",
            "data_name_map",
            "config_kwargs",
            "model_init_kwargs",
            "tokenizer_kwargs",
        ]
        _VALID_LIST_FIELDS = [
            "train_dataset_prefix",
            "valid_dataset_prefix",
            "test_dataset_prefix",
        ]

        # copied from: transformers/training_args.py/__post_init__()
        for field in _ADDITIONAL_VALID_DICT_FILEDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indiciate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)
            elif isinstance(passed_value, dict) or passed_value is None:
                pass
            else:
                raise ValueError(f"{field}은 dict로 설정해야 함. {passed_value}")

        for field in _VALID_LIST_FIELDS:
            passed_value = getattr(self, field)
            if isinstance(passed_value, str) and passed_value.startswith("["):
                loaded_list = json.loads(passed_value)
                setattr(self, field, loaded_list)
            elif passed_value is None or isinstance(passed_value, list):
                pass
            else:
                raise ValueError(f"{field}은 list로 설정해야 함. {passed_value}")

        quantization_config = get_quantization_config(self)
        self.config_kwargs = {
            **self.config_kwargs,
            "attn_implementation": self.attn_implementation,
            "use_cache": False if self.gradient_checkpointing else True,
        }

        self.model_init_kwargs = {
            **(self.model_init_kwargs if self.model_init_kwargs else {}),
            "revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": quantization_config,
            "device_map": get_kbit_device_map() if quantization_config is not None else None,
            "torch_dtype": (
                self.torch_dtype if self.torch_dtype in ["auto", None] else getattr(torch, self.torch_dtype)
            ),
        }

        self.tokenizer_kwargs = {
            **(self.tokenizer_kwargs if self.tokenizer_kwargs else {}),
            "revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.chat_template:
            self.tokenizer_kwargs["chat_template"] = self.chat_template

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

        if self.group_by_length:
            logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")

        if self.dataset_kwargs:
            logger.warning(
                "skip_prepare_dataset이 True임! 이 코드엔 데이터셋을 준비하는 코드가 있기 때문에 자동으로 False로 바꿈."
            )
            self.dataset_kwargs["skip_prepare_dataset"] = False

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d


logger = transformers.utils.logging.get_logger("transformers")


def main(train_args: SFTScriptArguments) -> None:
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path, **train_args.tokenizer_kwargs)
    config_kwargs = {
        **train_args.config_kwargs,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **config_kwargs)

    with (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    ):
        train_dataset, valid_dataset, test_dataset = processing_datasets(
            train_args,
            tokenizer,
            PROCESSOR_REGISTRY[train_args.data_preprocessor_type],
        )

    model_kwargs = {"config": config, **train_args.model_init_kwargs}
    model = AutoModelForCausalLM.from_pretrained(train_args.model_name_or_path, **model_kwargs)
    model.train()

    logger.info(f"Model parameter count: {get_model_param_count(model)}")

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    compute_metrics, callbacks = None, None
    if train_args.data_preprocessor_type in METRICS_REGISTRY:
        from callbacks import OnlyPicklingCallback

        compute_metrics = partial(
            METRICS_REGISTRY[train_args.data_preprocessor_type],
            tokenizer=tokenizer,
            args=train_args,
        )

        callbacks = [OnlyPicklingCallback()]

    collator = PackingCollatorForLLM(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        sample_dataset=train_dataset or valid_dataset or test_dataset,
    )

    trainer = PackingTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if train_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=collator,
        args=train_args,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        peft_config=get_peft_config(train_args),
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: PackingTrainer, args: SFTScriptArguments) -> None:
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.log_metrics("train", outputs.metrics)
    trainer.save_metrics("train", outputs.metrics)


@torch.no_grad()
def valid(trainer: PackingTrainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    metrics = trainer.evaluate(valid_datasets)
    metrics = {key: obj for key, obj in metrics.items() if type(obj).__module__ == "builtins"}

    trainer.log_metrics("valid", metrics)
    trainer.save_metrics("valid", metrics)


if "__main__" in __name__:
    parser = TrlParser([SFTScriptArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

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
