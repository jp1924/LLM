from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union

import lm_eval
import torch
import torch.distributed as dist
from accelerate import Accelerator
from lm_eval.api.model import TemplateLM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import configure_pad_token, get_dtype
from lm_eval.tasks import TaskManager, get_task_dict
from torch.distributed.fsdp import FullyShardedDataParallel

import transformers
from transformers import Trainer, TrainerCallback


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

        self.model.eval()
        if not isinstance(self.model, FullyShardedDataParallel):
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

        self.delta = None
        self.peft = None
        self.softmax_dtype = None
        self.mixed_precision_dtype = None

    def _model_generate(self, *args, **kwargs):
        raise NotImplementedError("_model_generate is not implemented for CustomHFLM")

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator") and False:
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Multi-GPU 동기화를 위한 패딩 추가/제거 로직이 포함된 _loglikelihood_tokens
        """
        padding_indices = []  # 패딩 인덱스 추적

        # Multi-GPU 환경이고 context caching이 활성화된 경우에만 패딩 적용
        if self._world_size > 1 and self.backend == "causal" and self.logits_cache:
            # 패딩 추가 (패딩 인덱스도 함께 반환)
            requests, num_padding_added, padding_indices = self._add_sync_padding(requests)
        else:
            num_padding_added = 0

        results = super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)

        # 패딩 인덱스에 해당하는 결과만 제거
        if num_padding_added > 0:
            # 패딩이 아닌 결과만 선택
            padding_set = set(padding_indices)
            results = [res for idx, res in enumerate(results) if idx not in padding_set]

        return results

    def _add_sync_padding(
        self, requests: List[Tuple[Tuple[str, str], List[int], List[int]]]
    ) -> Tuple[List[Tuple[Tuple[str, str], List[int], List[int]]], int, List[int]]:
        """
        Multi-GPU 동기화를 위한 패딩 추가

        Returns:
            (padded_requests, num_padding_added, padding_indices)
        """
        local_num_groups = self._count_unique_context_groups(requests)
        global_max_groups = self._sync_max_groups(local_num_groups)
        num_padding_needed = global_max_groups - local_num_groups

        padding_indices = []  # 패딩이 추가된 위치 추적

        if num_padding_needed > 0:
            # 패딩 시작 인덱스
            start_idx = len(requests)
            padded_requests = self._append_dummy_requests(requests, num_padding_needed)
            # 패딩 인덱스 기록
            padding_indices = list(range(start_idx, start_idx + num_padding_needed))
        else:
            padded_requests = requests

        if dist.is_initialized():
            dist.barrier()

        return padded_requests, num_padding_needed, padding_indices

    def _count_unique_context_groups(self, requests: List[Tuple[Tuple[str, str], List[int], List[int]]]) -> int:
        """
        Collator의 context grouping 로직을 시뮬레이션하여 unique 그룹 수 계산

        이 메서드는 Collator가 내부적으로 수행하는 그룹핑을 미리 계산합니다:
        - group_fn = _lookup_one_token_cont
        - key = context_enc + continuation_enc[:-1]
        """
        context_groups: Dict[tuple, list] = defaultdict(list)

        for req in requests:
            # req 구조: ((str, str), context_enc, continuation_enc)
            _, context_enc, continuation_enc = req

            # _lookup_one_token_cont와 동일한 키 생성
            # context + continuation[:-1]을 키로 사용
            grouping_key = tuple(context_enc + continuation_enc[:-1])

            context_groups[grouping_key].append(req)

        # Collator. get_batched()는 각 그룹에서 1개씩만 선택
        # 따라서 그룹 수 = 실제 처리될 요청 수
        num_unique_groups = len(context_groups)

        return num_unique_groups

    def _sync_max_groups(self, local_value: int) -> int:
        """
        모든 GPU에서 최대 그룹 수를 찾아 동기화

        Args:
            local_value: 현재 GPU의 unique 그룹 수

        Returns:
            모든 GPU 중 최대 그룹 수
        """
        if not dist.is_initialized() or self._world_size == 1:
            return local_value

        # Tensor로 변환 (GPU 메모리에 올림)
        local_tensor = torch.tensor([local_value], dtype=torch.long, device=self._device)

        # All-Reduce with MAX operation
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)

        max_groups = local_tensor.item()

        if self._rank == 0:
            logger.debug(f"[Rank {self._rank}] Local groups: {local_value}, Global max groups: {max_groups}")

        return max_groups

    def _append_dummy_requests(
        self, requests: List[Tuple[Tuple[str, str], List[int], List[int]]], num_padding: int
    ) -> List[Tuple[Tuple[str, str], List[int], List[int]]]:
        """
        Dummy 패딩 요청 추가

        전략:
        1. 가장 짧은 요청을 기반으로 패딩 생성 (계산 비용 최소화)
        2. 각 패딩에 고유한 context를 부여 (중복 그룹핑 방지)
        3. 식별 가능한 마커 추가 (디버깅 용이)

        Args:
            requests: 원본 요청 리스트
            num_padding: 추가할 패딩 개수

        Returns:
            패딩이 추가된 요청 리스트
        """
        if num_padding == 0:
            return requests

        # 패딩용 토큰 ID 가져오기
        pad_token_id = self._get_safe_padding_token_id()

        # 패딩 요청 생성
        padded_requests = requests.copy()

        for i in range(num_padding):
            # 고유한 context 생성 (각 rank와 index에 따라 다름)
            # 이렇게 하면 서로 다른 그룹으로 인식됨
            unique_context = [
                pad_token_id,  # 기본 패딩 토큰
                (self._rank * 10000 + i) % self.vocab_size,  # rank별 고유 ID
            ]

            # 단일 토큰 continuation (최소 비용)
            unique_continuation = [pad_token_id]

            # 식별 가능한 마커 (디버깅용)
            dummy_request = (
                (f"__PADDING_RANK{self._rank}_IDX{i}__", ""),  # str tuple
                unique_context,  # context_enc
                unique_continuation,  # continuation_enc
            )

            padded_requests.append(dummy_request)

        return padded_requests

    def _get_safe_padding_token_id(self) -> int:
        """
        안전한 패딩 토큰 ID 반환

        우선순위:
        1. tokenizer. pad_token_id
        2. tokenizer.eos_token_id
        3. tokenizer. unk_token_id
        4. 0 (fallback)
        """
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        elif self.tokenizer.unk_token_id is not None:
            return self.tokenizer.unk_token_id
        else:
            logger.warning(f"[Rank {self._rank}] No valid padding token found. Using 0 as fallback.")
            return 0


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
        args,
        state,
        control,
        model,
        processing_class,
        **kwargs,
    ):
        if state.global_step == 0 and self.do_init_eval:
            self.log_likelihood_evaluate(model=model, processing_class=processing_class, **kwargs)
        if (
            state.global_step % self.eval_steps == 0
            and state.global_step != 0
            and state.global_step >= self.eval_start
        ):
            self.log_likelihood_evaluate(model=model, processing_class=processing_class, **kwargs)

    @torch.no_grad()
    def log_likelihood_evaluate(
        self,
        model,
        processing_class,
        limit: Optional[Union[int, float]] = None,
        samples: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if limit is not None and samples is not None:
            raise ValueError("Either 'limit' or 'samples' must be None, but both are not None.")

        lm = CustomHFLM(
            pretrained=model,
            tokenizer=getattr(processing_class, "tokenizer", processing_class),
            batch_size=self.eval_batch_size,
        )
        outputs = lm_eval.evaluate(lm=lm, task_dict=self.task_dict, limit=limit, samples=samples)
        model.train()

        if outputs:
            final_map = {}
            for k, v in outputs["results"].items():
                final_map[f"lm_eval/{k}"] = v["acc,none"]
            self.trainer.log(final_map)
