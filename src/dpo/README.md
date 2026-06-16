# DPO

SFT와 같은 철학(단일 train_args, main.py 구조, 데이터 맵, 캐시)을 그대로 잇는다. 공통 규칙은 최상위 README, 라벨링은 SFT README를 본다. 여기선 DPO 고유한 부분만 적었음.

## 입력 포맷 흡수

`preprocessor.py`가 다양한 preference 포맷을 chosen/reject 두 conversation으로 정규화한다.

- chosen/reject가 통째로 conversation인 경우
- prompt가 conversation(이전 턴 포함) + chosen/reject가 문자열
- history + 단일 prompt 문자열 + chosen/reject 문자열
- 단일 prompt 문자열 + chosen/reject 문자열 (싱글턴)

chosen/reject는 마지막 답변만 다르고 prompt 구간은 같아야 한다. 어긋나거나 빈 답변은 건너뛴다.

map 출력은 packing과 무관한 conversational 형태로 둔다. 토큰화는 packing 분기되는 `_pack` 단계에서만 한다. 덕분에 packing on/off를 바꿔도 같은 map 캐시를 재사용한다.


## prefix-sharing packing

![prefix_sharing](https://arxiv.org/html/2410.20305v2/extracted/5966621/figures/prefix_sharing.png)

DPO는 (prompt + chosen), (prompt + rejected) 두 시퀀스를 forward해야 하는데 prompt가 중복으로 계산된다. 그래서 prompt를 한 번만 두고 `[P, chosen_tail, reject_tail]`로 이어붙인 뒤, 커스텀 어텐션 마스크로 chosen / reject가 공유 prefix만 보게 한다. prompt 연산을 한 번으로 줄여 GPU 처리량을 향상시킴. (SFT의 packing과 같은 동기다)

어텐션 구현은 https://github.com/li-plus/flash-preference.git 를 참고했다.

- collator는 `position_ids`만 넘기고, seg/group/gather는 거기서 복원한다 (`derive_seg_group`, `derive_gathers`).
- 백엔드 3종 지원: sdpa는 additive 마스크, flex는 BlockMask. flash_attention_2는 현재 동일 규칙의 sdpa로 처리한다.
- 마스크는 forward 바깥에서 1회 생성한다. gradient checkpointing recompute 때 저장 텐서 수가 어긋나 CheckpointError가 나는 걸 막기 위함이다.


## PackingDPOTrainer

packed forward로 logp를 직접 계산해 DPO loss를 구한다.

- 표준 경로: TRL의 모든 loss_type(sigmoid/hinge/ipo/... ) + 다중 loss 조합,
  f-divergence, WPO를 packed logits로 그대로 재현한다.
- liger 경로(`use_liger_kernel=True`): sigmoid/apo_zero/apo_down/sppo_hard/nca_pair만. ref_model이 필요하고(use_peft 불가), softcapping(gemma 등)은 미적용이라 표준 경로와 미세한 수치 차이가 있다.
- 미지원: ld_alpha, precompute_ref_log_probs.
- non-packing 배치면 TRL 기본 경로로 위임하는 하이브리드다.

reference model은 full FT면 동일 아키텍처를 동결해 한 번 더 로드하고, LoRA면 None으로 두고 어댑터를 꺼서 reference logp를 계산한다.


