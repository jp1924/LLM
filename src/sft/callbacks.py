import transformers
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


logger = transformers.utils.logging.get_logger("transformers")


class OnlyPicklingCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        processing_class: PreTrainedTokenizer,
        **kwargs,
    ):
        for idx, log in enumerate(state.log_history):
            state.log_history[idx] = {key: obj for key, obj in log.items() if type(obj).__module__ == "builtins"}
