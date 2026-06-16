import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import datasets
import torch
from adepters import get_peft_config
from preprocessor import PROCESSOR_REGISTRY, processing_datasets
from setproctitle import setproctitle
from trainer import PackingDPOTrainer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import get_model_param_count


logger = hf_logging.get_logger("transformers")


# SFT 와 동일하게 DPOConfig + ModelConfig 를 다중 상속해 하나의 train_args 로 모든 설정을 관리한다.
@dataclass
class DPOScriptArguments(DPOConfig, ModelConfig):
    _VALID_DICT_FIELDS = DPOConfig._VALID_DICT_FIELDS + [
        "dataset_truncate_map",
        "dataset_name_map",
        "dataset_prefix",
        "dataset_files_map",
        "config_kwargs",
        "processor_kwargs",
        "lora_kwargs",
    ]
    # -------------------------- Datasets Args ------------------------- #
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={"help": "학습에 사용할 데이터셋의 hub_repo 이름을 리스트로 입력한다. 예: ['hf_repo1', 'hf_repo2']."},
    )
    dataset_type: str = field(
        default="dpo",
        metadata={
            "help": f"데이터 전처리 방식. {', '.join(PROCESSOR_REGISTRY.keys())} 중 하나여야 한다.",
            "choices": list(PROCESSOR_REGISTRY.keys()),
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "데이터 전처리 시 사용할 배치 크기를 설정하는 값."},
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": "repo별 split→역할 매핑. {repo_or_'default': {'train': [split...], 'valid': [...], 'test': [...]}} 형태. "
            "특정 repo만 일부 역할에 기여시키려면 해당 repo 키에 train을 빼면 된다(기존 data_split_map 통합)."
        },
    )
    dataset_prefix: dict = field(
        default_factory=dict,
        metadata={
            "help": "데이터의 샘플 개수를 조절하기 위한 맵. 예: {'repo_name': {'train': 3000, 'validation': 1500}}. 데이터셋 처리 시 활용된다. subset 단위로 자르고 싶을 때는 {'repo_name-subset_name': 5000} 형태로 입력하면 된다."
        },
    )
    dataset_truncate_map: Union[dict, str] | None = field(
        default_factory=dict,
        metadata={"help": "데이터 샘플 개수 조절 맵. 예: {'repo_name': {'train': 3000, 'validation': 1500}}."},
    )
    dataset_name_map: Union[dict, str] | None = field(
        default_factory=dict,
        metadata={
            "help": "데이터셋의 구성 이름을 매핑하기 위한 맵. 예: {'repo_name': 'data_config_name'}. 데이터셋 로드 시 사용된다."
        },
    )
    dataset_files_map: Union[dict, str] | None = field(
        default_factory=dict,
        metadata={
            "help": "local에서 데이터를 불러올 때 {'train': 'train_file_path', 'validation': 'validation_file_path', 'test': 'test_file_path'} 형태로 데이터 파일 경로를 매핑하기 위한 맵. 데이터셋 로드 시 사용된다."
        },
    )

    config_kwargs: Union[dict, str] | None = field(default_factory=dict, metadata={"help": ""})
    processor_kwargs: Union[dict, str] | None = field(default_factory=dict, metadata={"help": ""})
    lora_kwargs: Union[dict, str] | None = field(
        default_factory=dict,
        metadata={
            "help": "PEFT의 LoraConfig에 추가적으로 전달할 인자들을 딕셔너리 형태로 입력한다. 예: {'lora_kwargs': {'initializer_range': 0.02}}. get_peft_config 함수에서 유효한 인자인지 확인한다."
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
        },
    )
    # -------------------------- Evaluate Args ------------------------- #
    eval_harness_tasks: List[str] = field(
        default=None,
        metadata={"help": "평가에 활용할 lm-eval-harness의 태스크 리스트."},
    )

    gpu_mem_check: bool = field(
        default=False,
        metadata={"help": "forward/backward/optimizer step 구간별 GPU peak memory 를 logging 한다."},
    )

    # -------------------------- Packing Args -------------------------- #
    # SFTConfig 에는 있으나 DPOConfig 에는 없는 packing 관련 필드. 공용 preprocessor
    # (_merge_datasets 등)가 train_args.packing 을 참조하므로 SFT 와 동일 기본값으로 추가한다.
    # DPO 에서 packing 은 데이터 처리 단계의 포맷 유지(list 포맷, set_format 생략)와
    # collator 단의 prefix-sharing packing(attention.py) 을 함께 제어한다.
    # flash-attn 2 미지원 모델에서도 SDPA fallback 으로 packing 가능.
    packing: bool = field(
        default=False,
        metadata={"help": "True 면 데이터를 list 포맷으로 유지하고 collator 에서 prefix-sharing packing 을 수행한다."},
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={"help": "SFT 호환용 필드(DPO 미사용). best-fit-decreasing('bfd') / 'wrapped'."},
    )
    eval_packing: bool | None = field(
        default=None,
        metadata={"help": "평가 데이터 packing 여부(SFT 호환용 필드)."},
    )

    def __post_init__(self) -> None:
        DPOConfig.__post_init__(self)
        ModelConfig.__post_init__(self)

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
        self.processor_kwargs = {
            **(self.processor_kwargs or {}),
            "revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.chat_template_path:
            self.processor_kwargs["chat_template"] = Path(self.chat_template_path).read_text(encoding="utf-8")

        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path


def main(train_args: DPOScriptArguments) -> None:
    try:
        processor = AutoProcessor.from_pretrained(train_args.model_name_or_path, **train_args.processor_kwargs)
    except OSError:
        processor = AutoTokenizer.from_pretrained(train_args.model_name_or_path, **train_args.processor_kwargs)

    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **train_args.config_kwargs)

    train_dataset, valid_dataset, test_dataset = processing_datasets(train_args, processor)

    model_kwargs = {"config": config, **(train_args.model_init_kwargs or {})}
    architecture = getattr(transformers, config.architectures[0].replace("FSDP", ""))
    model = architecture.from_pretrained(train_args.model_name_or_path, **model_kwargs).train()
    model.use_cache = False if train_args.gradient_checkpointing else True

    # FSDP 에서 존재하지 않는 모듈이 _no_split_modules 에 있으면 에러가 나므로 필터링한다.
    exist_module = {module.__class__.__name__ for module in model.modules()}
    model._no_split_modules = list(set(model._no_split_modules).intersection(exist_module))
    model.train()

    if train_args.use_peft:
        # reference model: full FT(use_peft=false) 면 동결된 동일 아키텍처를 한 번 더 로드,
        # LoRA(use_peft=true) 면 None(DPOTrainer 가 어댑터 off 로 reference logprob 계산).
        ref_model = None
    else:
        ref_model = architecture.from_pretrained(train_args.model_name_or_path, **model_kwargs).eval()
        ref_model.use_cache = False if train_args.gradient_checkpointing else True
        ref_model._no_split_modules = list(set(ref_model._no_split_modules).intersection(exist_module))
        ref_model.eval()

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    logger.info(f"Model parameter count: {get_model_param_count(model)}")

    train_args.eval_packing = False  # 이미 초기화 했기 때문에 SFTTrainer에서 모델 초기화 방식을 사용하지 않도록 설정
    train_args.model_init_kwargs = None

    trainer_cls = PackingDPOTrainer if train_args.packing else DPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if train_args.eval_strategy != "no" else None,
        processing_class=processor,
        peft_config=get_peft_config(train_args),
    )

    if train_args.eval_harness_tasks is not None:
        from callbacks import EvalHarnessCallBack

        trainer.add_callback(
            EvalHarnessCallBack(
                trainer,
                processor,
                train_args.eval_harness_tasks,
                eval_steps=train_args.eval_steps,
                do_init_eval=train_args.eval_on_start,
                eval_batch_size=train_args.eval_batch_size,
            )
        )

    if train_args.gpu_mem_check:
        from callbacks import GpuMemoryCallback

        trainer.add_callback(GpuMemoryCallback())

    if "wandb" in (train_args.report_to or []):
        from callbacks import WandbCodeArtifactCallback

        repo_root = Path(__file__).resolve().parents[2]
        trainer.add_callback(
            WandbCodeArtifactCallback(
                root=repo_root.as_posix(),
                include_dirs=("src/dpo", "scripts", "config"),
            )
        )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, None)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: DPOTrainer, args: DPOScriptArguments) -> None:
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.log_metrics("train", outputs.metrics)
    trainer.save_metrics("train", outputs.metrics)


@torch.no_grad()
def valid(trainer: DPOTrainer, valid_datasets) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    metrics = trainer.evaluate(valid_datasets)
    metrics = {key: obj for key, obj in metrics.items() if type(obj).__module__ == "builtins"}

    trainer.log_metrics("valid", metrics)
    trainer.save_metrics("valid", metrics)


if "__main__" in __name__:
    parser = TrlParser([DPOScriptArguments])
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
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    main(train_args)
