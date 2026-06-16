
# SupervisedFineTuning/Pretrain

`main.py`에 학습과 관련된 대부분의 코드를 구현하고, 데이터 전처리는 `preprocessor.py`로 분할.

## 학습 방식 선택

`data_preprocessor_type`으로 sft / pretrain을 선택

- sft: 권장 컬럼 `conversations`, `prompt`, `answer`
- pretrain: 권장 컬럼 `corpus`, `sentence_ls`
  여러 형태의 데이터를(conversations / messages / prompt + answer / QA pair)을 자동으로 conversation 형태로 정규화

## assistant-only 라벨링

SFT의 경우 user/system를 제외한 assistant 구간만 loss 계산에 포함
모델 라인업마다 chat-template이 달라 user/assistant 구간 분리가 까다롭고, 어떤 tokenizer는 같은 단어도 앞뒤에 붙는 토큰에 따라 다르게 인코딩 되는 이슈가 있어서 번거롭지만 offset 기반 + decode 반복 방식으로 대부분의 tokenizer에서 assistant 구간을 잡는다.

- fast: `return_offsets_mapping=True`로 토큰별 char offset을 받고 `char_to_token`으로 content의 char 위치를 토큰 인덱스로 바꿔 라벨링한다(`_labels_via_offsets`).
- slow(sentencepiece): offset을 안 주므로 `get_sentencepiece_offset`으로 토큰을 하나씩 decode해 char offset을 직접 쌓고, content도 encode→decode로 맞춰 찾는다.

세부 처리:
- 양쪽 다 원본이 아니라 encode→decode 디코딩 공간에서 찾는다. zero-width, 특수문자, 공백 처리로 원본과 디코딩 결과가 달라지기 때문이다.
- 턴을 순서대로 훑으며 cursor를 전진시켜, assistant가 prompt 문구를 인용해도 prompt 구간에 잘못 매칭되지 않게 한다.
- chat template이 턴 앞뒤 공백을 strip하므로 미리 strip한다. 안 그러면 offset 매칭이 깨진다.
- 마지막 종료 토큰(EOS 등)까지 라벨에 포함한다. 멈추는 법을 배워야 하기 때문이다.
- 빈 assistant나 구간을 못 찾는 degenerate 샘플은 건너뛴다.

> assistant가 D, C 같은 단답(선택지)이고 그 값이 user에도 있으면 offset 시작 위치가 어긋나 인코딩이 실패한다.
> 단답 샘플은 애초에 필터링하는 게 맞다 싶어 따로 처리하지 않고 둔다(건너뛰기 대상).
> 오동작 여지는 있지만 대부분의 데이터에선 정상 동작해 이대로 쓴다.

## packing + no-pad collator

`packing=True`면 여러 시퀀스를 pad 없이 하나로 이어붙이고, `position_ids`로 경계를 표현한다. 이때 batch_size가 아니라 max_length를 키워서 GPU 유휴 자원을 거의 없애고 throughput을 끌어올림.
전처리 단계에서 이미 packing을 끝내므로 SFTTrainer 쪽 packing은 끈다 (`skip_prepare_dataset=True`).
학습 시작 전 collator가 샘플 2개를 직접 돌려서 BOS/EOS가 데이터에 실제로 들어갔는지, eager attention인데 packing을 쓰고 있진 않은지 검증한다. 잘못된 학습을 미리 막는다.

## SP (Sequence Parallelism)

`sp_enabled`면 max_length를 world_size 배수로 맞추고, `max_length *= batch_size`,
`batch_size=1`로 바꿔 시퀀스를 GPU에 쪼개 올린다.

## 설계 노트

### packing: SPFHP sampler → trl.pack_dataset

예전엔 PackedBERT의 SPFHP sampler를 적용해 별도 `trainer.py`까지 만들었다. 지금은 trl `pack_dataset`과 효율 차이가 거의 없어서 그걸 쓰고 현재 구조로 단순화했다.

### builder script와 고정 컬럼

1회용으로 버려지던 전처리를 builder script로 올려 고정 컬럼을 갖게 만들었다. 여러 데이터를 써도 컬럼이 같으니 전처리가 단순해진다. SFT 권장 컬럼은 그 컨벤션이다. (datasets 4.0.0에서 builder script가 폐지돼 대안은 재설계 중. 최상위 README 참고)
