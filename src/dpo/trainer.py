from typing import List, Union

import torch
import torch.nn.functional as F
from attention import (
    as_mask_mapping,
    build_additive_mask,
    build_block_mask,
    derive_gathers,
    derive_seg_group,
    enable_prefix_packing,
)
from trl import DPOTrainer

from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature


def _ids(example: dict, *names):
    for n in names:
        v = example.get(n)
        if v is not None:
            return list(v)
    return None


def _pad_stack(seqs: List[List[int]], pad_value: int, side: str = "right") -> torch.Tensor:
    """가변 길이 1D int 리스트들을 [N, Lmax] 로 패딩."""
    L = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        p = L - len(s)
        pad = [pad_value] * p
        out.append(torch.tensor(s + pad if side == "right" else pad + s, dtype=torch.long))
    return torch.stack(out)


class DataCollatorForPreference(DataCollatorMixin):
    """preference 데이터 collator. packing/non-packing/vision 분기.

    Args:
        args: DPOScriptArguments (use_prefix_packing, max_length 등).
        processing_class: AutoTokenizer 또는 AutoProcessor(vision).
        return_tensors: 기본 "pt".
    """

    def __init__(self, args, processing_class, return_tensors: str = "pt") -> None:
        self.args = args
        self.processor = processing_class
        self.tokenizer = getattr(processing_class, "tokenizer", processing_class)
        self.use_packing = bool(getattr(args, "use_prefix_packing", False))
        self.max_length = getattr(args, "max_length", None)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.return_tensors = return_tensors

    # ----------------------------- 텍스트: packing ----------------------------- #
    def _pack_collate(self, examples: List[dict]) -> dict:
        """[P1, C1_tail, R1_tail, P2, ...] 단일 packed 행 + position_ids 만 생성.

        chosen 을 먼저, reject 를 나중에 쌓는다(순서 규약). 이 순서가 position_ids 의 run 구조로
        인코딩되어 attention.derive_seg_group/derive_gathers 가 side(0=chosen,1=reject)를 복원한다.
        """
        input_ids: List[int] = []
        pos_ids: List[int] = []
        for ex in examples:
            prompt = _ids(ex, "prompt_ids", "prompt_input_ids", "input_ids")
            chosen = _ids(ex, "chosen_ids", "chosen_input_ids", "chosen")
            rejected = _ids(ex, "rejected_ids", "rejected_input_ids", "rejected")
            P = len(prompt)
            input_ids += prompt
            pos_ids += list(range(P))
            for tail in (chosen, rejected):  # 순서 고정: chosen → reject
                input_ids += tail
                pos_ids += list(range(P, P + len(tail)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long)[None],
            "position_ids": torch.tensor(pos_ids, dtype=torch.long)[None],
        }

    # --------------------------- 텍스트: non-packing --------------------------- #
    def _pad_collate(self, examples: List[dict]) -> dict:
        """TRL DataCollatorForPreference 와 동일 포맷: [chosen 행들; rejected 행들]."""
        prompt_chosen, prompt_rejected, chosen_m, rejected_m = [], [], [], []
        for ex in examples:
            p = _ids(ex, "prompt_ids", "prompt_input_ids", "input_ids")
            c = _ids(ex, "chosen_ids", "chosen_input_ids", "chosen")
            r = _ids(ex, "rejected_ids", "rejected_input_ids", "rejected")
            prompt_chosen.append(p + c)
            prompt_rejected.append(p + r)
            chosen_m.append([0] * len(p) + [1] * len(c))
            rejected_m.append([0] * len(p) + [1] * len(r))

        if self.max_length is not None:
            prompt_chosen = [x[: self.max_length] for x in prompt_chosen]
            prompt_rejected = [x[: self.max_length] for x in prompt_rejected]
            chosen_m = [x[: self.max_length] for x in chosen_m]
            rejected_m = [x[: self.max_length] for x in rejected_m]

        input_ids = prompt_chosen + prompt_rejected
        completion_mask = chosen_m + rejected_m
        attention_mask = [[1] * len(x) for x in input_ids]
        return {
            "input_ids": _pad_stack(input_ids, self.pad_token_id),
            "attention_mask": _pad_stack(attention_mask, 0),
            "completion_mask": _pad_stack(completion_mask, 0),
        }

    # ------------------------------- vision ----------------------------------- #
    def _vision_collate(self, examples: List[dict]) -> dict:
        """이미지 포함 preference. TRL DataCollatorForVisionPreference 방식(on-the-fly).

        prompt/chosen/rejected 가 messages(conversational) 라고 가정하고 processor 로 처리.
        packing 은 image 토큰 구조상 비지원 → non-packing(표준) 출력 + pixel_values.
        """
        if "image" in examples[0]:
            for ex in examples:
                ex["images"] = [ex.pop("image")]
        images = [ex["images"] for ex in examples] * 2  # chosen/rejected 복제
        if all(im == [] for im in images):
            images = None

        prompts = [ex["prompt"] for ex in examples] * 2
        chosens = [ex["chosen"] for ex in examples]
        rejecteds = [ex["rejected"] for ex in examples]

        proc_p = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        proc_c = self.processor(
            text=chosens,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        proc_r = self.processor(
            text=rejecteds,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def _pad_two(a, b, val):
            L = max(a.shape[1], b.shape[1])
            a = torch.nn.functional.pad(a, (0, L - a.shape[1]), value=val)
            b = torch.nn.functional.pad(b, (0, L - b.shape[1]), value=val)
            return torch.cat([a, b], dim=0)

        comp_ids = _pad_two(proc_c["input_ids"], proc_r["input_ids"], pad_id)
        comp_mask = _pad_two(proc_c["attention_mask"], proc_r["attention_mask"], 0)
        input_ids = torch.cat((proc_p["input_ids"], comp_ids), dim=1)
        attention_mask = torch.cat((proc_p["attention_mask"], comp_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(proc_p["attention_mask"]), comp_mask), dim=1)

        out = dict(proc_p)  # pixel_values / image_grid_thw 등 포함
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["completion_mask"] = completion_mask
        return out

    # ------------------------------- dispatch --------------------------------- #
    def torch_call(self, examples: Union[List[dict], List[List[dict]]]) -> BatchFeature:
        if examples and isinstance(examples[0], list):  # nested(packing dataset) 평탄화
            examples = [e for sub in examples for e in sub]

        is_vision = ("images" in examples[0]) or ("image" in examples[0])
        if is_vision:
            if self.use_packing:
                raise NotImplementedError("vision preference 는 아직 packing 미지원이다.")
            batch = self._vision_collate(examples)
        elif self.use_packing:
            batch = self._pack_collate(examples)
        else:
            batch = self._pad_collate(examples)
        return BatchFeature(batch, self.return_tensors)


class PackingDPOTrainer(DPOTrainer):
    """prefix-packing 으로 DPO logp 를 계산하는 트레이너.

    지원:
      - 표준 경로: TRL 의 모든 loss_type(sigmoid/hinge/ipo/exo_pair/nca_pair/robust/bco_pair/
        sppo_hard/aot/aot_unpaired/apo_zero/apo_down/discopop/sft/sigmoid_norm) + 다중 loss 조합,
        f_divergence_type(reverse_kl/forward_kl/js_divergence/alpha_divergence), use_weighting(WPO).
        packed forward 의 '정확한' logits(softcapping 포함)로 logp 를 계산한다.
      - liger 경로(use_liger_kernel=True): liger 가 지원하는 5종
        (sigmoid/apo_zero/apo_down/sppo_hard/nca_pair)만. packed decoder hidden 을 gather 해
        [chosen_rows; rejected_rows] 로 재조립한 뒤 LigerFusedLinearDPOLoss(lm_head+loss fused)에 전달.
        주의: liger 는 hidden@W 만 계산하므로 final_logit_softcapping(gemma 등)은 미적용 → 해당
        모델에서는 표준 경로와 미세한 수치 차이가 있다(TRL liger 경로와 동일한 한계).

    미지원: ld_alpha(LD-DPO), precompute_ref_log_probs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # packing/non-packing/vision 분기하는 통합 collator.
        self.data_collator = DataCollatorForPreference(self.args, self.processing_class)
        self._assert_supported()
        # config 의 base attn_implementation → packing 변종으로 영구 전환(monkey-patch, context manager X).
        # 영구 설정이라 backward recompute 도 packing 으로 실행되어 checkpoint-safe.
        self._packing_impl = enable_prefix_packing(self.accelerator.unwrap_model(self.model))
        if self.ref_model is not None:
            enable_prefix_packing(self.accelerator.unwrap_model(self.ref_model))

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """커스텀 preprocessor(dpo_processor)가 이미 prompt/chosen/rejected 를 토큰화해 두었으므로
        TRL 의 표준 토큰화(_prepare_dataset)는 건너뛴다(no-op). collator 가 raw id 를 직접 소비한다."""
        return dataset

    # liger 가 hidden@W 만으로 계산 가능한 loss 집합.
    _LIGER_LOSS_TYPES = {"sigmoid", "apo_zero", "apo_down", "sppo_hard", "nca_pair"}

    def _assert_supported(self) -> None:
        if getattr(self, "ld_alpha", None) is not None:
            raise NotImplementedError("PackingDPOTrainer 는 ld_alpha(LD-DPO) 를 지원하지 않는다.")
        if getattr(self, "precompute_ref_logps", False):
            raise NotImplementedError("PackingDPOTrainer 는 precompute_ref_log_probs=True 를 지원하지 않는다.")
        if getattr(self, "use_liger_kernel", False):
            # liger 는 reference hidden 이 필요하므로 ref_model 이 있어야 한다(use_peft 불가).
            if self.ref_model is None:
                raise NotImplementedError("liger packing 은 reference model 이 필요하다(use_peft=True 불가).")
            bad = [lt for lt in self.loss_types if lt not in self._LIGER_LOSS_TYPES]
            if bad:
                raise NotImplementedError(
                    f"liger packing 은 {sorted(self._LIGER_LOSS_TYPES)} 만 지원한다 (받음: {self.loss_types})."
                )

    # ------------------------------------------------------------------ #
    # f-divergence score 변환 (TRL dpo_trainer 와 동일)
    # ------------------------------------------------------------------ #
    def _f_scores(self, chosen_logratios, rejected_logratios):
        fd = getattr(self, "f_divergence_type", "reverse_kl")
        if fd == "reverse_kl":
            return chosen_logratios, rejected_logratios
        if fd == "forward_kl":
            return -torch.exp(-chosen_logratios), -torch.exp(-rejected_logratios)
        if fd == "js_divergence":
            return F.logsigmoid(chosen_logratios), F.logsigmoid(rejected_logratios)
        if fd == "alpha_divergence":
            alpha = getattr(self, "f_alpha_divergence_coef", 0.5)
            if abs(alpha - 1.0) < 1e-6:
                return chosen_logratios, rejected_logratios
            coef = 1.0 / (alpha - 1.0)
            dtype = chosen_logratios.dtype
            clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}.get(dtype, 80.0)
            tc = torch.clamp(((alpha - 1.0) * chosen_logratios).float(), max=clamp_max)
            tr = torch.clamp(((alpha - 1.0) * rejected_logratios).float(), max=clamp_max)
            return torch.exp(tc).to(dtype) * coef, torch.exp(tr).to(dtype) * coef
        raise ValueError(f"Unknown f_divergence_type: {fd}")

    def _loss_weights(self):
        w = getattr(self, "loss_weights", None)
        return w if w else [1.0] * len(self.loss_types)

    def _dpo_loss(self, pq, ref_chosen_logps, ref_rejected_logps):
        """packing 으로 구한 per-sequence logp(pq) 로 loss_type 별 loss 합산 (TRL 수식 포팅)."""
        beta = self.beta
        ls = getattr(self, "label_smoothing", 0.0)
        cl, rl = pq["chosen_logps"], pq["rejected_logps"]
        device = cl.device
        chosen_logratios = cl - ref_chosen_logps
        rejected_logratios = rl - ref_rejected_logps
        chosen_scores, rejected_scores = self._f_scores(chosen_logratios, rejected_logratios)
        delta = chosen_scores - rejected_scores

        loss = 0.0
        for loss_type, lw in zip(self.loss_types, self._loss_weights()):
            if loss_type == "sigmoid":
                ps = -F.logsigmoid(beta * delta)
            elif loss_type == "hinge":
                ps = torch.relu(1 - beta * delta)
            elif loss_type == "ipo":
                ca = chosen_scores / pq["chosen_len"].clamp(min=1.0)
                ra = rejected_scores / pq["rejected_len"].clamp(min=1.0)
                ps = (ca - ra - 1 / (2 * beta)) ** 2
            elif loss_type == "exo_pair":
                eps = torch.tensor(ls, device=device)
                qw, log_qw = torch.sigmoid(beta * delta), F.logsigmoid(beta * delta)
                ql, log_ql = torch.sigmoid(-beta * delta), F.logsigmoid(-beta * delta)
                ps = qw * (log_qw - torch.log1p(-eps)) + ql * (log_ql - torch.log(eps))
            elif loss_type == "nca_pair":
                cr, rr = beta * chosen_scores, beta * rejected_scores
                ps = -F.logsigmoid(cr) - 0.5 * F.logsigmoid(-cr) - 0.5 * F.logsigmoid(-rr)
            elif loss_type == "robust":
                ps = (-(1 - ls) * F.logsigmoid(beta * delta) - ls * F.logsigmoid(-beta * delta)) / (1 - 2 * ls)
            elif loss_type == "bco_pair":
                ps = -F.logsigmoid(beta * chosen_scores) - F.logsigmoid(-beta * rejected_scores)
            elif loss_type == "sppo_hard":
                ps = (chosen_scores - 0.5 / beta) ** 2 + (rejected_scores + 0.5 / beta) ** 2
            elif loss_type == "aot":
                lr = torch.sort(cl - rl, dim=0)[0]
                rlr = torch.sort(ref_chosen_logps - ref_rejected_logps, dim=0)[0]
                d = lr - rlr
                ps = -F.logsigmoid(beta * d) * (1 - ls) - F.logsigmoid(-beta * d) * ls
            elif loss_type == "aot_unpaired":
                d = torch.sort(chosen_logratios, dim=0)[0] - torch.sort(rejected_logratios, dim=0)[0]
                ps = -F.logsigmoid(beta * d) * (1 - ls) - F.logsigmoid(-beta * d) * ls
            elif loss_type == "apo_zero":
                ps = (1 - torch.sigmoid(beta * chosen_logratios)) + torch.sigmoid(beta * rejected_logratios)
            elif loss_type == "apo_down":
                ps = torch.sigmoid(beta * chosen_logratios) + (1 - torch.sigmoid(beta * delta))
            elif loss_type == "discopop":
                logits = delta * beta
                mod = torch.sigmoid(logits / self.args.discopop_tau)
                ps = -F.logsigmoid(logits) * (1 - mod) + torch.exp(-logits) * mod
            elif loss_type == "sft":
                sft = -(pq["sft_sum"] / max(int(pq["sft_cnt"]), 1))
                ps = sft.expand(cl.shape[0])
            elif loss_type == "sigmoid_norm":
                ca = chosen_scores / pq["chosen_len"].clamp(min=1.0)
                ra = rejected_scores / pq["rejected_len"].clamp(min=1.0)
                ps = -F.logsigmoid(beta * (ca - ra))
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            if getattr(self, "use_weighting", False) and pq.get("chosen_w") is not None:
                ps = ps * (pq["chosen_w"] * pq["rejected_w"])
            loss = loss + ps.mean() * lw

        chosen_rewards = (beta * chosen_logratios).detach()
        rejected_rewards = (beta * rejected_logratios).detach()
        return loss, chosen_rewards, rejected_rewards

    def _build_mask(self, model, seg_ids, group_ids, position_ids, device):
        """forward 바깥에서 마스크 1회 생성(checkpoint-safe) 후 layer_type 별 dict 로 감싸 반환.

        packing 변종이 flex 면 BlockMask, 그 외(sdpa/flash)면 additive. attention_mask 로 그대로 넘긴다.
        """
        if self._packing_impl == "prefix_packing_flex_attention" and device.type == "cuda":
            mask = build_block_mask(seg_ids, group_ids, position_ids)
        else:
            dtype = next(self.model.parameters()).dtype
            mask = build_additive_mask(seg_ids, group_ids, position_ids, dtype)
        return as_mask_mapping(self.accelerator.unwrap_model(model), mask)

    def _packed_logps(self, model, input_ids, position_ids, mask_mapping, gathers, device, want_weight=False):
        """packed forward 1 회 → 그룹·side 별 completion logp 합 및 부수 정보를 dict 로 반환.

        반환 dict:
          chosen_logps[K], rejected_logps[K] : completion logp 합(그룹 순)
          chosen_len[K], rejected_len[K]     : completion 토큰 수
          sft_sum(scalar), sft_cnt(int)      : chosen completion per-token logp 합/개수(sft 용)
          chosen_w[K]/rejected_w[K] or None  : WPO 가중치(use_weighting 시)
          logits[L,V]                        : packed logits(return_outputs/metrics 용)
        attn_implementation 은 __init__ 에서 packing 변종으로 영구 설정됨(context manager 불필요).
        """
        out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask_mapping, use_cache=False)
        logits = out.logits[0].float()  # [L, V]

        chosen, rejected, clen, rlen = [], [], [], []
        chosen_w, rejected_w = [], []
        sft_sum = torch.zeros((), device=device)
        sft_cnt = 0
        for g in gathers:
            pred = logits[g["pred_idx"].to(device)]  # [T, V]
            tgt = input_ids[0, g["tgt_idx"].to(device)]  # [T]
            tok_logp = torch.gather(F.log_softmax(pred, dim=-1), 1, tgt[:, None]).squeeze(1)  # [T]
            seq_logp = tok_logp.sum()
            T = tok_logp.shape[0]
            w = None
            if want_weight:
                with torch.no_grad():
                    log_denom = torch.logsumexp(2.0 * pred, dim=-1) - 2.0 * torch.logsumexp(pred, dim=-1)
                    w = torch.exp((tok_logp - log_denom).mean())
            if g["side"] == 0:
                chosen.append(seq_logp)
                clen.append(T)
                sft_sum = sft_sum + seq_logp
                sft_cnt += T
                if want_weight:
                    chosen_w.append(w)
            else:
                rejected.append(seq_logp)
                rlen.append(T)
                if want_weight:
                    rejected_w.append(w)
        return {
            "chosen_logps": torch.stack(chosen),
            "rejected_logps": torch.stack(rejected),
            "chosen_len": torch.tensor(clen, dtype=torch.float, device=device),
            "rejected_len": torch.tensor(rlen, dtype=torch.float, device=device),
            "sft_sum": sft_sum,
            "sft_cnt": sft_cnt,
            "chosen_w": torch.stack(chosen_w) if want_weight else None,
            "rejected_w": torch.stack(rejected_w) if want_weight else None,
            "logits": logits,
        }

    def _log_dpo_metrics(self, mode, chosen_logps, rejected_logps, chosen_rewards, rejected_rewards):
        m = self._metrics[mode]
        g = self.accelerator.gather_for_metrics
        m["rewards/chosen"].append(g(chosen_rewards).mean().item())
        m["rewards/rejected"].append(g(rejected_rewards).mean().item())
        m["rewards/accuracies"].append(g((chosen_rewards > rejected_rewards).float()).mean().item())
        m["rewards/margins"].append(g(chosen_rewards - rejected_rewards).mean().item())
        m["logps/chosen"].append(g(chosen_logps.detach()).mean().item())
        m["logps/rejected"].append(g(rejected_logps.detach()).mean().item())

    def _compute_loss(self, model, inputs, return_outputs=False):
        # non-packing(표준) 배치면 TRL 기본 경로로 위임.
        if "position_ids" not in inputs:
            return super()._compute_loss(model, inputs, return_outputs)

        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        want_weight = bool(getattr(self, "use_weighting", False))

        input_ids = inputs["input_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)

        # seg/group/gathers 는 collator 가 아니라 여기서 position_ids 로부터 파생(attention.derive_*).
        seg_ids, group_ids = derive_seg_group(position_ids)
        gathers = derive_gathers(position_ids)

        # 마스크는 forward 바깥에서 1회 생성 → policy/ref 공용
        mask_mapping = self._build_mask(model, seg_ids, group_ids, position_ids, device)

        # policy forward (grad)
        pq = self._packed_logps(model, input_ids, position_ids, mask_mapping, gathers, device, want_weight)

        # reference forward (no grad)
        with torch.no_grad():
            if self.ref_model is not None:
                rq = self._packed_logps(self.ref_model, input_ids, position_ids, mask_mapping, gathers, device)
            else:
                base = self.accelerator.unwrap_model(model)
                with base.disable_adapter():
                    rq = self._packed_logps(model, input_ids, position_ids, mask_mapping, gathers, device)

        loss, chosen_rewards, rejected_rewards = self._dpo_loss(pq, rq["chosen_logps"], rq["rejected_logps"])
        self._log_dpo_metrics(mode, pq["chosen_logps"], pq["rejected_logps"], chosen_rewards, rejected_rewards)

        if return_outputs:
            return loss, {"logits": pq["logits"][None]}
        return loss

    # ------------------------------------------------------------------ #
    # liger 경로 (use_liger_kernel=True)
    # ------------------------------------------------------------------ #
    def _decoder_hidden(self, model, input_ids, position_ids, mask_mapping):
        """packed 시퀀스를 decoder 만 forward(lm_head 미적용) → last_hidden_state [L, H]."""
        out = model.get_decoder()(
            input_ids=input_ids, position_ids=position_ids, attention_mask=mask_mapping, use_cache=False
        )
        return out.last_hidden_state[0]  # [L, H]

    def _liger_rows(self, hidden, input_ids, gathers, device):
        """packed hidden 에서 (predictor hidden, target) 를 gather 해 [chosen_rows; rejected_rows] 구성.

        liger 는 row 별 (target != -100) 토큰의 logp 합으로 sequence logp 를 구하므로,
        completion 토큰의 예측자(predictor) hidden 과 target 토큰만 모아 행으로 쌓으면 된다.
        반환: _input [2K, Tmax, H], target [2K, Tmax] (부족분 hidden=0 / target=-100 패딩).
        """
        chosen_h, chosen_t, rej_h, rej_t = [], [], [], []
        for g in gathers:
            h = hidden[g["pred_idx"].to(device)]  # [T, H]
            t = input_ids[0, g["tgt_idx"].to(device)]  # [T]
            if g["side"] == 0:
                chosen_h.append(h)
                chosen_t.append(t)
            else:
                rej_h.append(h)
                rej_t.append(t)
        rows_h = chosen_h + rej_h
        rows_t = chosen_t + rej_t
        tmax = max(h.shape[0] for h in rows_h)
        _input = torch.stack([F.pad(h, (0, 0, 0, tmax - h.shape[0])) for h in rows_h])  # [2K, Tmax, H]
        target = torch.stack([F.pad(t, (0, tmax - t.shape[0]), value=-100) for t in rows_t])  # [2K, Tmax]
        return _input, target

    def _compute_loss_liger(self, model, inputs, return_outputs=False):
        # non-packing(표준) 배치면 TRL liger 경로로 위임.
        if "position_ids" not in inputs:
            return super()._compute_loss_liger(model, inputs, return_outputs)
        if return_outputs:
            raise RuntimeError("liger packing 은 logits 를 materialize 하지 않아 return_outputs=True 를 지원하지 않는다.")

        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        input_ids = inputs["input_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)

        seg_ids, group_ids = derive_seg_group(position_ids)
        gathers = derive_gathers(position_ids)
        mask_mapping = self._build_mask(model, seg_ids, group_ids, position_ids, device)

        # policy / reference 를 decoder 만 forward(packing 마스크 적용) → hidden gather → liger fused loss.
        hidden = self._decoder_hidden(model, input_ids, position_ids, mask_mapping)
        _input, target = self._liger_rows(hidden, input_ids, gathers, device)
        with torch.no_grad():
            ref_hidden = self._decoder_hidden(self.ref_model, input_ids, position_ids, mask_mapping)
            ref_input, _ = self._liger_rows(ref_hidden, input_ids, gathers, device)

        lm_head = model.get_output_embeddings()
        ref_lm_head = self.ref_model.get_output_embeddings()
        loss, metrics = self.liger_loss_fn(
            lm_head.weight, _input, target, lm_head.bias, ref_input, ref_lm_head.weight, ref_lm_head.bias
        )
        chosen_logps, rejected_logps, _clm, _rlm, _nll, chosen_rewards, rejected_rewards = metrics
        self._log_dpo_metrics(mode, chosen_logps, rejected_logps, chosen_rewards, rejected_rewards)
        return loss
