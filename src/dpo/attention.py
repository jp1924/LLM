from typing import List, Sequence

import torch
import torch.nn.functional as F

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv


def longest_common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """두 토큰 시퀀스의 공통 prefix 길이 (= 공유 prompt 길이)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_prefix_packed_mask(prefix_len: int, tail_lens: Sequence[int], dtype, device) -> torch.Tensor:
    """``[prefix, tail_0, tail_1, ...]`` packed 시퀀스용 4D additive mask (단일 그룹).

    허용=0, 금지=-inf. prefix 는 내부 causal, tail_i 는 prefix 전체 + 자기 tail 내부 causal.
    반환 shape: ``[1, 1, L, L]``.
    """
    total = prefix_len + sum(tail_lens)
    neg = torch.finfo(dtype).min
    mask = torch.full((total, total), neg, dtype=dtype, device=device)

    starts = [prefix_len]
    for t in tail_lens[:-1]:
        starts.append(starts[-1] + t)

    pf = torch.arange(prefix_len, device=device)
    mask[:prefix_len, :prefix_len].masked_fill_(pf[:, None] >= pf[None, :], 0.0)

    for start, t in zip(starts, tail_lens):
        end = start + t
        mask[start:end, :prefix_len] = 0.0
        idx = torch.arange(t, device=device)
        mask[start:end, start:end].masked_fill_(idx[:, None] >= idx[None, :], 0.0)

    return mask[None, None]


def pack_groups(groups: List[tuple]):
    """여러 (공유 prompt, full 시퀀스들) 그룹을 하나의 block-diagonal packed 표현으로 변환.

    레이아웃: ``[P1, tail1_0, tail1_1, ..., P2, tail2_0, ...]`` (max_length 까지 greedy pack 호출측 책임)

    Args:
        groups: ``[(prompt_ids, [full_seq_0, full_seq_1, ...]), ...]``
                full_seq 는 ``prompt + completion`` 전체 토큰. side 0=chosen, 1=reject ... 순.
    Returns:
        input_ids [1, L], position_ids [1, L], seg_ids [1, L], group_ids [1, L],
        gathers(list[dict]): 그룹·side별 {group, side, pred_idx, tgt_idx} (completion logp 수집용)

    seg_ids: 0=prefix, 1=side0(chosen), 2=side1(reject), ... / group_ids: 그룹 번호.
    """
    input_ids: List[int] = []
    pos_id: List[int] = []
    group_id: List[int] = []
    seg_id: List[int] = []
    gathers = []

    for gi, (prompt_ids, full_seqs) in enumerate(groups):
        P = len(prompt_ids)
        offset = len(input_ids)

        input_ids.extend(prompt_ids)
        pos_id.extend(range(P))
        group_id.extend([gi] * P)
        seg_id.extend([0] * P)
        prefix_last = offset + P - 1

        for side, seq in enumerate(full_seqs):
            tail = seq[P:]
            T = len(tail)
            start = len(input_ids)
            input_ids.extend(tail)
            pos_id.extend(range(P, P + T))
            group_id.extend([gi] * T)
            seg_id.extend([side + 1] * T)

            tgt_idx = list(range(start, start + T))
            pred_idx = [prefix_last] + tgt_idx[:-1]
            gathers.append(
                {
                    "group": gi,
                    "side": side,
                    "pred_idx": torch.tensor(pred_idx, dtype=torch.long),
                    "tgt_idx": torch.tensor(tgt_idx, dtype=torch.long),
                }
            )

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)[None]
    position_ids_t = torch.tensor(pos_id, dtype=torch.long)[None]
    seg_ids_t = torch.tensor(seg_id, dtype=torch.long)[None]
    group_ids_t = torch.tensor(group_id, dtype=torch.long)[None]
    return input_ids_t, position_ids_t, seg_ids_t, group_ids_t, gathers


def pack_group(prompt_ids: List[int], full_seqs: List[List[int]]):
    """공유 prompt + 여러 full 시퀀스를 packed 표현으로 변환 (단일 그룹, 레거시).

    Returns: input_ids [1,L], position_ids [1,L], meta(prefix_len/tail_lens), gathers.
    """
    P = len(prompt_ids)
    tails = [seq[P:] for seq in full_seqs]

    packed = list(prompt_ids)
    positions = list(range(P))
    starts = []
    for tail in tails:
        starts.append(len(packed))
        packed.extend(tail)
        positions.extend(range(P, P + len(tail)))

    input_ids = torch.tensor(packed, dtype=torch.long)[None]
    position_ids = torch.tensor(positions, dtype=torch.long)[None]
    meta = {"prefix_len": P, "tail_lens": [len(t) for t in tails]}

    gathers = []
    for start, tail in zip(starts, tails):
        T = len(tail)
        tgt_idx = list(range(start, start + T))
        pred_idx = [P - 1] + tgt_idx[:-1]
        gathers.append(
            {
                "pred_idx": torch.tensor(pred_idx, dtype=torch.long),
                "tgt_idx": torch.tensor(tgt_idx, dtype=torch.long),
            }
        )
    return input_ids, position_ids, meta, gathers


# --------------------------------------------------------------------------- #
# position_ids → seg/group/gathers 파생 (collator 는 position_ids 만 준다)
# --------------------------------------------------------------------------- #
def derive_seg_group(position_ids):
    """packed ``position_ids`` 만으로 seg_ids/group_ids 를 파생한다.

    캐너니컬 레이아웃 ``[P(0..P-1), tail0(P..), tail1(P..), ...]`` 가정:
      - group 시작 = position_id==0
      - 그룹 내 첫 run = prefix+chosen(연속), 이후 run 각각 = tail(P부터 재시작)
      - reject(첫 tail run)의 재시작 값 = P(prefix 길이) → prefix/chosen 경계 복원
    seg_ids: 0=prefix, 1=chosen(side0), 2=reject(side1), ... / group_ids: 그룹 번호. 각각 [1, L].
    """
    pos = position_ids.flatten()
    dev = pos.device
    group_id = (pos == 0).to(torch.long).cumsum(0) - 1
    seg_id = torch.zeros_like(pos)
    for g in range(int(group_id.max().item()) + 1):
        gidx = (group_id == g).nonzero(as_tuple=True)[0]
        gp = pos[gidx]
        rel_drop = torch.zeros(len(gidx), dtype=torch.bool, device=dev)
        rel_drop[1:] = gp[1:] <= gp[:-1]  # run 경계(드롭/동일)
        drops = rel_drop.nonzero(as_tuple=True)[0]
        if len(drops) == 0:
            continue  # reject 없음 → 단일 causal run, 전부 prefix(0) 으로 두어도 마스크 정상
        P = int(gp[drops[0]].item())  # reject 재시작값 = prefix 길이
        run0 = gidx[: drops[0]]
        seg_id[run0] = torch.where(pos[run0] < P, 0, 1)  # prefix / chosen
        run_starts = drops.tolist() + [len(gidx)]
        for side, (s, e) in enumerate(zip(run_starts[:-1], run_starts[1:]), start=1):
            seg_id[gidx[s:e]] = side + 1  # reject=2, ...
    return seg_id[None], group_id[None]


def derive_gathers(position_ids):
    """packed ``position_ids`` 로부터 그룹·side별 gather 인덱스(pred_idx/tgt_idx) 파생."""
    pos = position_ids.flatten()
    seg_ids, group_ids = derive_seg_group(position_ids)
    seg, grp = seg_ids.flatten(), group_ids.flatten()
    gathers = []
    for g in range(int(grp.max().item()) + 1):
        gmask = grp == g
        prefix_idx = (gmask & (seg == 0)).nonzero(as_tuple=True)[0]
        prefix_last = prefix_idx[-1]
        sides = sorted(set(seg[gmask & (seg >= 1)].tolist()))
        for s in sides:
            tgt = (gmask & (seg == s)).nonzero(as_tuple=True)[0]
            pred = torch.cat([prefix_last.view(1), tgt[:-1]])
            gathers.append({"group": g, "side": s - 1, "pred_idx": pred, "tgt_idx": tgt})
    return gathers


def _make_mask_mod(group_id, seg_id, pos_id):
    """위치별 라벨 텐서로 block-diagonal prefix-shared causal mask_mod 생성.

    규칙: 같은 그룹 && (key가 prefix || 같은 side) && pos causal.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        same_group = group_id[q_idx] == group_id[kv_idx]
        key_is_prefix = seg_id[kv_idx] == 0
        same_seg = seg_id[q_idx] == seg_id[kv_idx]
        causal = pos_id[q_idx] >= pos_id[kv_idx]
        return same_group & (key_is_prefix | same_seg) & causal

    return mask_mod


# --------------------------------------------------------------------------- #
# 마스크 생성 (forward 바깥에서 1회 → checkpoint-safe). _prefix_packed 마커 부착.
# --------------------------------------------------------------------------- #
def build_block_mask(seg_ids, group_ids, position_ids):
    """FlexAttention BlockMask 를 forward 바깥에서 1회 생성(checkpoint-safe).

    레이어 forward 내부에서 생성하면 recompute 와 저장 텐서 수가 어긋나 CheckpointError 가 난다.
    seg_ids/group_ids/position_ids: [1,L] 또는 [L] long tensor(동일 device).
    """
    from torch.nn.attention.flex_attention import create_block_mask

    sid, gid, pid = seg_ids.flatten(), group_ids.flatten(), position_ids.flatten()
    L = sid.shape[0]
    bm = create_block_mask(_make_mask_mod(gid, sid, pid), B=None, H=None, Q_LEN=L, KV_LEN=L, device=sid.device)
    bm._prefix_packed = True  # packing 마커(비-packing generate 호출과 구분 → delegate 판별)
    return bm


def build_additive_mask(seg_ids, group_ids, position_ids, dtype):
    """SDPA 용 [1,1,L,L] additive mask (flex 와 동일 규칙). forward 바깥에서 1회 생성."""
    sid, gid, pid = seg_ids.flatten(), group_ids.flatten(), position_ids.flatten()
    same_group = gid[:, None] == gid[None, :]
    key_is_prefix = (sid == 0)[None, :]
    same_seg = sid[:, None] == sid[None, :]
    causal = pid[:, None] >= pid[None, :]
    allow = same_group & (key_is_prefix | same_seg) & causal
    neg = torch.finfo(dtype).min
    mask = torch.where(
        allow,
        torch.zeros((), dtype=dtype, device=sid.device),
        torch.full((), neg, dtype=dtype, device=sid.device),
    )
    mask = mask[None, None]
    mask._prefix_packed = True  # packing 마커
    return mask


# --------------------------------------------------------------------------- #
# FlexAttention 컴파일 래퍼
# --------------------------------------------------------------------------- #
_FLEX_FN = None
_FLEX_COMPILE = True  # 운영시 True 권장(torch.compile). 정확도 디버깅시 False 로 eager.


def _get_flex_fn():
    global _FLEX_FN
    if _FLEX_FN is not None:
        return _FLEX_FN
    from torch.nn.attention.flex_attention import flex_attention

    _FLEX_FN = torch.compile(flex_attention) if _FLEX_COMPILE else flex_attention
    return _FLEX_FN


# --------------------------------------------------------------------------- #
# packed 마스크 소비 (사전 생성된 마스크만 받는다 → 내부 생성 X → checkpoint 안전)
# --------------------------------------------------------------------------- #
def _flex_with_block_mask(module, query, key, value, scaling, block_mask):
    flex = _get_flex_fn()
    out = flex(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=hasattr(module, "num_key_value_groups"),
    )
    return out.transpose(1, 2).contiguous(), None


def _sdpa_with_mask(module, query, key, value, scaling, dropout, attn_mask):
    if attn_mask.dtype != query.dtype:
        attn_mask = attn_mask.to(query.dtype)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout, scale=scaling, is_causal=False
    )
    return out.transpose(1, 2).contiguous(), None


def _sdpa_prefix_packed(meta, module, query, key, value, scaling, dropout):
    """레거시 단일그룹 SDPA (meta: prefix_len/tail_lens)."""
    cache = meta.setdefault("_mask_cache", {})
    L = query.shape[2]
    ckey = (L, query.dtype, str(query.device))
    if ckey not in cache:
        cache[ckey] = build_prefix_packed_mask(meta["prefix_len"], meta["tail_lens"], query.dtype, query.device)
    mask = cache[ckey]
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling, is_causal=False
    )
    return out.transpose(1, 2).contiguous(), None


# --------------------------------------------------------------------------- #
# flash-attn varlen (레거시 단일그룹)
# --------------------------------------------------------------------------- #
_FLASH_VARLEN = None
_FLASH_TRIED = False


def get_flash_varlen_func():
    """`flash_attn_varlen_func` 확보: (1) flash_attn 패키지, (2) kernels-community/flash-attn2. 없으면 None."""
    global _FLASH_VARLEN, _FLASH_TRIED
    if _FLASH_TRIED:
        return _FLASH_VARLEN
    _FLASH_TRIED = True
    try:
        from flash_attn import flash_attn_varlen_func

        _FLASH_VARLEN = flash_attn_varlen_func
        return _FLASH_VARLEN
    except Exception:
        pass
    try:
        from kernels import get_kernel

        kernel = get_kernel("kernels-community/flash-attn2", revision="main")
        _FLASH_VARLEN = kernel.flash_attn_varlen_func
    except Exception:
        _FLASH_VARLEN = None
    return _FLASH_VARLEN


def _flash_prefix_packed(meta, flash_attn_varlen_func, module, query, key, value, scaling, dropout):
    """레거시 단일그룹 flash varlen: K/V prefix 복제 + 비대칭 cu_seqlens."""
    P = meta["prefix_len"]
    tail_lens = meta["tail_lens"]

    q = query[0].transpose(0, 1)
    k = key[0].transpose(0, 1)
    v = value[0].transpose(0, 1)

    prefix_k = slice(0, P)
    starts, s = [], P
    for t in tail_lens:
        starts.append(s)
        s += t

    k_parts, v_parts, cu_q, cu_k = [], [], [0], [0]
    for start, t in zip(starts, tail_lens):
        k_parts += [k[prefix_k], k[start : start + t]]
        v_parts += [v[prefix_k], v[start : start + t]]
        cu_q.append(cu_q[-1] + (P + t if start == starts[0] else t))
        cu_k.append(cu_k[-1] + P + t)
    q_parts = [q[: P + tail_lens[0]]] + [q[start : start + t] for start, t in zip(starts[1:], tail_lens[1:])]

    q_cat = torch.cat(q_parts, dim=0)
    k_cat = torch.cat(k_parts, dim=0)
    v_cat = torch.cat(v_parts, dim=0)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=q.device)

    out = flash_attn_varlen_func(
        q_cat,
        k_cat,
        v_cat,
        cu_seqlens_q=cu_q_t,
        cu_seqlens_k=cu_k_t,
        max_seqlen_q=int((cu_q_t[1:] - cu_q_t[:-1]).max()),
        max_seqlen_k=int((cu_k_t[1:] - cu_k_t[:-1]).max()),
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out[None], None


# --------------------------------------------------------------------------- #
# 3 백엔드별 packing forward (sdpa / flash_attention_2 / flex_attention)
# packed 마스크면 packing, 아니면 원본 백엔드로 위임(delegate).
# → attn_implementation 을 packing 변종으로 영구 설정해도 generate 등 비-packing 호출은 안전.
# --------------------------------------------------------------------------- #
PACKING_IMPL = {
    "sdpa": "prefix_packing_sdpa",
    "flash_attention_2": "prefix_packing_flash_attention_2",
    "flex_attention": "prefix_packing_flex_attention",
}


def _is_packed(attention_mask) -> bool:
    """build_block_mask/build_additive_mask 가 붙인 packing 마커 여부."""
    return bool(getattr(attention_mask, "_prefix_packed", False))


def prefix_packing_flex_attention_forward(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    if _is_packed(attention_mask):
        return _flex_with_block_mask(module, query, key, value, scaling, attention_mask)
    from transformers.integrations.flex_attention import flex_attention_forward

    return flex_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)


def prefix_packing_sdpa_forward(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    if _is_packed(attention_mask):
        return _sdpa_with_mask(module, query, key, value, scaling, dropout, attention_mask)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )


def prefix_packing_flash_attention_forward(
    module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs
):
    if _is_packed(attention_mask):
        # flash-attn varlen packing(cu_seqlens + prefix 복제)은 마스크가 아닌 meta 가 필요하다.
        # 현재는 동일 규칙의 SDPA packing 으로 처리(정확도 동일, FA2 커널 이득은 미적용).
        return _sdpa_with_mask(module, query, key, value, scaling, dropout, attention_mask)
    from transformers.integrations.flash_attention import flash_attention_forward

    return flash_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )


for _name, _fn in {
    "prefix_packing_sdpa": prefix_packing_sdpa_forward,
    "prefix_packing_flash_attention_2": prefix_packing_flash_attention_forward,
    "prefix_packing_flex_attention": prefix_packing_flex_attention_forward,
}.items():
    try:
        AttentionInterface.register(_name, _fn)
    except ValueError:
        pass


def enable_prefix_packing(model) -> str:
    """config 의 base attn_implementation(sdpa|flash_attention_2|flex_attention)을 보고
    대응 packing 변종으로 **영구 전환**한다(context manager 아님; monkey-patch).

    영구 설정이므로 backward 의 gradient checkpointing recompute 도 packing 변종으로 실행되어
    저장 텐서 수가 일치한다(CheckpointError 회피). 비-packing 호출(generate 등)은 마스크가
    packed 가 아니므로 자동으로 원본 백엔드로 위임된다.
    """
    base = model.config._attn_implementation
    if base in PACKING_IMPL.values():  # 이미 packing 변종이면 그대로
        return base
    target = PACKING_IMPL.get(base)
    if target is None:
        raise ValueError(f"packing 은 attn_implementation ∈ {list(PACKING_IMPL)} 에서만 지원한다 (받음: {base}).")
    model.set_attn_implementation(target)
    return target


def as_mask_mapping(model, mask):
    """마스크를 모델의 layer_type 별 dict 로 감싼다.

    transformers 모델은 forward 의 ``attention_mask`` 가 dict 면 내부 마스크 생성을 건너뛰고
    ``mask[layer_type]`` 를 각 어텐션 레이어에 그대로 전달한다(gemma 등). 즉 우리가 만든
    BlockMask/additive 마스크를 그대로 attention 백엔드까지 흘려보낼 수 있다.
    """
    layer_types = getattr(model.config, "layer_types", None)
    keys = set(layer_types) if layer_types else {"full_attention", "sliding_attention"}
    return {k: mask for k in keys}
