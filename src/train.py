import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import ProfileKwargs
from datasets import Dataset
from datasets import logging as ds_logging
from setproctitle import setproctitle
from trl import SFTConfig

import optimization
from metrics import METRICS_REGISTRY
from preprocessor import PROCESSOR_REGISTRY, processing_datasets
from trainer import PackingCollatorForLLM, PackingTrainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import get_model_param_count
from transformers.utils import is_sagemaker_mp_enabled


@dataclass
class DataPipelineArguments:
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
class TrainPipelineArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "The name or path of the pre-trained model."},
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
    profiling_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "profiling_kwargs"},
    )
    config_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    model_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    tokenizer_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    freeze_named_param: List[str] = field(
        default=None,
        metadata={"help": "freeze_named_param"},
    )

    profiling: bool = field(
        default=False,
        metadata={"help": "Whether to enable profiling during training."},
    )
    save_finally: bool = field(
        default=False,
        metadata={"help": "Whether to save the model finally."},
    )


@dataclass
class EvalPipelineArguments:
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
    tasks: str = field(
        default="haerae",
        metadata={"help": "The tasks to evaluate on."},
    )


@dataclass
class SFTTrainingArguments(
    DataPipelineArguments, TrainPipelineArguments, EvalPipelineArguments, Seq2SeqTrainingArguments, SFTConfig
):
    lr_scheduler_type: Union[optimization.NewSchedulerType, str] = field(default="linear")

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
            "model_kwargs",
            "tokenizer_kwargs",
            "profiling_kwargs",
        ]
        _VALID_LIST_FIELDS = [
            "train_dataset_prefix",
            "valid_dataset_prefix",
            "test_dataset_prefix",
            "freeze_named_param",
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

        self.config_kwargs = {
            **self.config_kwargs,
            "attn_implementation": self.attn_implementation,
        }

        self.tokenizer_kwargs = {
            **self.tokenizer_kwargs,
            "padding_side": self.padding_side,
        }

        if self.chat_template:
            self.tokenizer_kwargs["chat_template"] = self.chat_template

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

        if self.group_by_length:
            logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")

    @property
    def is_local_process_zero(self) -> bool:
        return self.local_process_index == 0

    @property
    def is_world_process_zero(self) -> bool:
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp  # type: ignore

            return smp.rank() == 0
        else:
            return self.process_index == 0


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: SFTTrainingArguments) -> None:
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

    model_kwargs = {"config": config, **train_args.model_kwargs}
    model = AutoModelForCausalLM.from_pretrained(train_args.model_name_or_path, **model_kwargs)
    model.train()

    if train_args.freeze_named_param:
        freeze_param_ls = [param for name, param in model.named_parameters() if name in train_args.freeze_named_param]
        if not freeze_param_ls:
            raise ValueError("freeze_named_param에 해당하는 모듈이 없음.")

        for param in freeze_param_ls:
            param.requires_grad = False

        if train_args.is_world_process_zero:
            full_param_num = get_model_param_count(model, trainable_only=False)
            alive_param_num = get_model_param_count(model, trainable_only=True)
            dead_param_num = full_param_num - alive_param_num

            logger.info(
                f"얼린 파라미터 수: {dead_param_num}, 활성화된 파라미터 수: {alive_param_num}, 전체 파라미터 수: {full_param_num}"
            )

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    compute_metrics = None
    if train_args.data_preprocessor_type in METRICS_REGISTRY:
        compute_metrics = partial(METRICS_REGISTRY[train_args.data_preprocessor_type], tokenizer)

    collator = PackingCollatorForLLM(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        sample_dataset=train_dataset or valid_dataset or test_dataset,
    )

    trainer = PackingTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        args=train_args,
        compute_metrics=compute_metrics,
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: PackingTrainer, args: SFTTrainingArguments) -> None:
    from transformers.trainer_utils import TrainOutput

    context = trainer.accelerator.profile(ProfileKwargs(**args.profiling_kwargs)) if args.profiling else nullcontext()

    with context as prof:
        try:
            outputs: TrainOutput = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        finally:
            if args.save_finally:
                try:
                    trainer._save_checkpoint(trainer.model, None)
                except BaseException:
                    pass

    save_path = Path(args.output_dir)
    if prof:
        prof.export_memory_timeline(save_path.with_suffix(".memory_trace.json").as_posix())
        prof.export_chrome_trace(save_path.with_suffix(".chrome_trace.json").as_posix())
        print(prof.key_averages().table(sort_by="flops", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    trainer.log_metrics("train", outputs.metrics)
    trainer.save_metrics("train", outputs.metrics)


@torch.no_grad()
def valid(trainer: PackingTrainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


@torch.no_grad()
def test(trainer: PackingTrainer, test_datasets: Dataset) -> None:
    from transformers import GenerationConfig, TextGenerationPipeline
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    pipeline = TextGenerationPipeline(
        model=trainer.model.eval(),
        tokenizer=trainer.processing_class,
        device=trainer.model.device,
        batch_size=trainer.args.per_device_eval_batch_size,
    )
    generate_kwargs = {
        "generation_config": GenerationConfig(
            max_new_tokens=1024,
            use_cache=True,
            do_sample=True,
            repetition_penalty=1.2,
            cache_implementation="hybrid",
        ),
        "synced_gpus": is_deepspeed_zero3_enabled(),
    }
    for data in test_datasets:
        outputs_1 = pipeline(
            trainer.processing_class.decode(data["input_ids"]),
            return_full_text=True,
            **generate_kwargs,
        )


if "__main__" in __name__:
    parser = HfArgumentParser([SFTTrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and train_args.is_world_process_zero:
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

    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    torch.cuda.set_per_process_memory_fraction(0.3, train_args.local_process_index)

    main(train_args)
