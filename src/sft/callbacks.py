from typing import List, Literal, Optional, Union

import lm_eval
import torch
from accelerate import Accelerator
from lm_eval.api.model import TemplateLM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import configure_pad_token, get_dtype
from lm_eval.tasks import TaskManager, get_task_dict

import transformers
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


logger = transformers.utils.logging.get_logger("transformers")


class CustomHFLM(HFLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: str | None = "main",
        subfolder: str = "",
        tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None = None,
        truncation: bool | None = False,
        logits_cache: bool = True,
        max_length: int | None = None,
        softmax_dtype: str | torch.dtype | None = None,
        mixed_precision_dtype: str | torch.dtype | None = None,
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = None,
        prefix_token_id: int | None = None,
        **kwargs,
    ) -> None:
        TemplateLM.__init__(self)

        accelerator = Accelerator()
        self.accelerator = accelerator

        self._device = self.accelerator.device
        self._model = pretrained
        self._config = pretrained.config
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(config=self.config, backend=backend, trust_remote_code=trust_remote_code)

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=None,
            add_bos_token=add_bos_token,
        )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token

        self._max_length = max_length
        self.pretrained = pretrained

        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        self.softmax_dtype = get_dtype(softmax_dtype) if softmax_dtype is not None else None
        self.mixed_precision_dtype = get_dtype(mixed_precision_dtype) if mixed_precision_dtype is not None else None

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            logger.info(f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}")

        #
        self.delta = None
        self.peft = None
        self.softmax_dtype = None
        self.mixed_precision_dtype = None

    def _model_generate(self, *args, **kwargs):
        raise NotImplementedError("_model_generate is not implemented for CustomHFLM")


class EvalHarnessCallBack(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        tokenizer,
        tasks: List[str],
        eval_steps=None,
        eval_start=0,
        do_init_eval=False,
        eval_batch_size=32,
    ) -> None:
        self.eval_batch_size = eval_batch_size
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.eval_start = eval_start if eval_start is not None else 0
        self.do_init_eval = do_init_eval

        self.task_dict = get_task_dict(tasks, TaskManager())

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 0 and self.do_init_eval:
            self.log_likelihood_evaluate(args, state, control, **kwargs)
        if (
            state.global_step % self.eval_steps == 0
            and state.global_step != 0
            and state.global_step >= self.eval_start
        ):
            self.log_likelihood_evaluate(args, state, control, **kwargs)

    def log_likelihood_evaluate(
        self, limit: Optional[Union[int, float]] = None, samples: Optional[dict] = None, **kwargs
    ) -> dict:
        if limit is not None and samples is not None:
            raise ValueError("Either 'limit' or 'samples' must be None, but both are not None.")

        lm = CustomHFLM(
            pretrained=self.trainer.model,
            tokenizer=self.tokenizer,
            batch_size=self.eval_batch_size,
            max_batch_size=128,
        )

        outputs = lm_eval.evaluate(lm=lm, task_dict=self.task_dict, limit=limit, samples=samples)
        self.trainer.model.train()
        # huggingface trainer의 log를 불러와 logging을 진행

        final_map = {}
        for k, v in outputs["results"].items():
            if "kmmlu" == k:
                name = "kmmlu/total"
            else:
                name = k.replace("kmmlu_", "kmmlu/")

            final_map[name] = v["acc,none"]
        self.trainer.log(final_map)
