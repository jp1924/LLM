import inspect
import json

from peft import PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.mapping import get_peft_config as _peft_get_config
from peft.utils import PeftType

from transformers import TrainingArguments, logging


logger = logging.get_logger("transformers")


def _collect_lora_defaults(train_args: TrainingArguments) -> dict:
    """기존 lora_* args를 LoraConfig 파라미터명으로 변환."""
    return {
        "task_type": getattr(train_args, "lora_task_type", None),
        "r": getattr(train_args, "lora_r", None),
        "target_modules": getattr(train_args, "lora_target_modules", None),
        "target_parameters": getattr(train_args, "lora_target_parameters", None),
        "lora_alpha": getattr(train_args, "lora_alpha", None),
        "lora_dropout": getattr(train_args, "lora_dropout", None),
        "bias": "none",
        "use_rslora": getattr(train_args, "use_rslora", None),
        "use_dora": getattr(train_args, "use_dora", None),
        "modules_to_save": getattr(train_args, "lora_modules_to_save", None),
    }


def get_peft_config(train_args: TrainingArguments) -> PeftConfig | None:
    is_main = train_args.distributed_state.is_local_main_process

    if not train_args.use_peft:
        return None

    peft_type = getattr(train_args, "peft_type", "LORA").upper()

    if peft_type not in PeftType.__members__:
        raise ValueError(f"지원하지 않는 peft_type: '{peft_type}'. 사용 가능: {[e.value for e in PeftType]}")

    # 명시적으로 넘긴 peft_kwargs 파싱
    peft_kwargs = getattr(train_args, "peft_kwargs", {})
    if isinstance(peft_kwargs, str):
        peft_kwargs = json.loads(peft_kwargs)

    # LoRA일 때만 기존 lora_* args를 기본값으로 사용
    # peft_kwargs에 같은 키가 있으면 peft_kwargs가 우선
    if peft_type == PeftType.LORA.value:
        lora_defaults = _collect_lora_defaults(train_args)
        # None인 항목은 병합에서 제외 (Config 클래스 기본값 유지)
        lora_defaults = {k: v for k, v in lora_defaults.items() if v is not None}
        merged_kwargs = {**lora_defaults, **peft_kwargs}  # peft_kwargs가 우선
    else:
        merged_kwargs = peft_kwargs

    # Config 파라미터 유효성 검증
    config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[PeftType[peft_type]]
    valid_params = set(inspect.signature(config_cls.__init__).parameters) - {"self"}
    invalid = set(merged_kwargs) - valid_params
    if invalid:
        raise ValueError(f"{config_cls.__name__}에 유효하지 않은 인자: {invalid}\n유효한 인자: {valid_params}")

    peft_config = _peft_get_config({"peft_type": peft_type, **merged_kwargs})

    if is_main:
        logger.info(f"PEFT type: {peft_type}, config: {peft_config}")

    return peft_config
