# LLM

LLM/LVM을 SFT, DPO, Pretrain으로 학습하고 전처리하는 코드를 정리한 repo.
자주 사용하는 학습 방식을 코드로 굳혀둔 것이고, 매번 만들던 1회용 코드를 최대한 줄이고 GPU를 낭비 없이 효율적으로 사용하는데 초점이 맞춰져 있다.

## 개발 이유

- LLM 학습은 너무 비효율적이다. GPU 자원을 최대로 쓰면서 처리량을 올리는 방법들을 구현.
- 내가 자주 하는 학습 스타일을 코드로 정리해 굳혀둔다.
- 1회용 작업을 최대한 줄인다.

## 디렉터리 규칙

`{method}/` 아래는 항상 이렇게 둔다.

- `main.py` (필수): args + `def main` + `if "__main__" in __name__`
- `preprocessor.py`: 데이터 전처리
- 그 외(callback 등)는 `callbacks.py` 같은 파일로만 분리

> 예전엔 `args.py`, `collator.py`로 쪼갰더니 파일과 VSCode 창만 늘고 한눈에 안 들어왔다.
> Python 모듈 철학과는 어긋나지만 `main.py` 통합을 택했다. 파일 수를 최소화하는 게 목적이다.

## main.py 작성 규칙

위에서부터 `args` → `def main` → `def train / valid / predict` 순서로 둔다.
train/valid 로직을 `main` 안에 박지 않고 함수로 뺀 이유는, 박아두면 코드 분리가 안 돼서 한눈에 안 보이기 때문이다.

`if "__main__" in __name__:` 블록은 logging, setproctitle, set_seed, args 파싱처럼 한 줄로 끝나는 잡설정 전용 장소다.


## 단일 train_args

args를 model/data/train으로 나누지 않고 하나로 상속해 합쳤다.
> 나눠두면 wandb엔 train args만 기록돼서 model, dataset 설정이 재현성 추적에서 누락된다.
> 실험 관리를 wandb로 하다 보니 전부 한 곳에 기록되게 합쳤다.


## 데이터 전처리

학습마다 별도 전처리 코드를 짜는 대신, 자주 쓰는 기능을 args 맵으로 정규화했다.

- `dataset_name_map`: subset(name) 매핑
- `dataset_prefix`: split → 역할(train/valid/test) 매핑
- `dataset_truncate_map`: 샘플 수 조절 / 분할
- `dataset_files_map`: 로컬 파일 매핑

여러 dataset을 섞어 쓰면 split, 분할, 이상치 처리 같은 1회용 작업이 매번 따라붙는데, 이걸 맵으로 빼서 args만 바꿔 끝낼 수 있게 했다.

권장 입력 컬럼:
- SFT: `conversations`, `prompt`, `answer`, `images`
- Pretrain: `corpus`, `sentence_ls`


## Args

대부분의 args는 [HF TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)를 그대로 따른다.
SFT는 추가로 TRL SFTConfig, ModelConfig, DPO는 DPOConfig, ModelConfig를 상속. 내가 추가한 args 기능은 아래와 같다.

데이터
- `dataset_repo_ls`: 학습에 쓸 dataset repo 목록
- `dataset_name_map` / `dataset_prefix` / `dataset_truncate_map` / `dataset_files_map`: 위 데이터 전처리 참고
- `dataset_batch_size`: 전처리 배치 크기

모델/토크나이저
- `config_kwargs`: AutoConfig에 넘길 추가 인자
- `tokenizer_kwargs`(SFT) / `processor_kwargs`(DPO): 토크나이저, 프로세서 추가 인자
- `chat_template_path`(DPO): chat template 지정
- `lora_kwargs`: LoraConfig에 넘길 추가 인자 (`peft_kwargs`로 LoRA 외 PEFT 타입 확장도 가능)

학습/평가
- `dataset_type`(SFT): sft / pretrain / dpo 선택
- `packing` / `packing_strategy` / `eval_packing`: packing 제어
- `eval_harness_tasks`: 학습 중 lm-eval-harness로 평가할 태스크 목록
- `gpu_mem_check`: forward/backward/optimizer 구간별 GPU peak memory 로깅

참고로 wandb로 report하면 코드 스냅샷을 아티팩트로 같이 저장해 재현성을 남긴다.


## callbacks

콜백은 `src/callbacks.py`에 모여 있고, SFT, DPO가 공용한다. 해당 args가 켜질 때만 붙는다.

- `EvalHarnessCallBack` (`eval_harness_tasks`): 학습 중 eval 시점에 lm-eval-harness를 같이 돌린다. 예전엔 학습이 끝난 뒤 checkpoint를 하나씩 lm eval 돌렸는데, 너무 번거롭고 시간도 많이 먹었다. trainer에 eval/predict loop가 있는 이유가 자동화해 사람 공수를 줄이는 것이라서 lm eval도 그 안으로 포함시킴.
- `WandbCodeArtifactCallback` (`report_to`에 wandb): base code 대비 뭐가 바뀌었고 실험이 어떻게 되는지를 artifact로 남긴다. 재현성, 버전 관리를 wandb로 하기 때문이다.
- `GpuMemoryCallback` (`gpu_mem_check`): 단순 memory 디버깅용. 구간별 GPU peak memory를 찍는다.


## SFT와 Pretrain 혼용

SFT 스크립트 하나로 sft와 pretrain을 둘 다 돌린다. `dataset_type`만 sft / pretrain으로 바꾸면 된다.

전처리(assistant-only 라벨링 vs 통문장 토큰화)와 collator의 라벨 처리(`labels` vs `input_ids`)가 이 값에 따라 갈릴 뿐, packing, 캐시, 데이터 맵 같은 나머지 파이프라인은 그대로 공유한다. 그래서 같은 데이터 구성 위에서 pretrain 후 sft로 이어가는 식의 실험을 args만 바꿔서 할 수 있다.


## 캐시

전처리 캐시는 최초 1회 실행에서만 생성된다. 최초 1회는 `main_process_first` 컨텍스트로 main process에서만 만들고, 이후 실행은 datasets가 그 캐시를 그대로 로드한다.

캐시 위치는 원본 데이터와 분리하고, 상태(모델/repo/run/max_length)별로 나눠 저장한다.

> datasets는 arrow 캐시를 원본과 같은 폴더에 쌓는데, 캐시를 정리하다 원본까지 지우는 사고가 잦았다. 그래서 분리했다.


## running

```
git clone https://github.com/jp1924/LLM.git
docker compose up -d # 도커 환경 구성
docker compose stop  # 도커 환경 종료
```

Docker/Compose: 27.3.1-build ce12230, v2.29.7
패키지 버전이 빨리 바뀌어서 실행 안 되면 Dockerfile 수정이 필요할 수 있다.

학습 실행은 `scripts/` 아래 스크립트를 쓴다.

```
bash scripts/run_zero3_sft.sh
bash scripts/run_zero3_dpo.sh
```


## flash-attn 설치

pypi로 받으면 컴파일에 오래 걸리지만, flash-attn repo가 주는 whl은 컴파일이
끝나 있어 빠르다. 먼저 환경 버전을 확인한다.

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.compiled_with_cxx11_abi())"
```

이후 flash-attn repo에서 환경에 맞는 whl을 받아 설치하면 된다.


## 작업 환경, 워크플로

내가 주로 작업하는 방식. 그냥 참고용임.

- docker 환경 `/root/workspace` 안에서 작업한다. 초기 파이썬 세팅은 `uv sync`로 맞춘다.
- 주로 VSCode + tmux로 작업. `.vscode`에 자주 쓰는 vscode extensions과 debug에 필요한 설정은 `launch.json`에 위치.
- tmux, bashrc, vi 같은 개발환경 구성 파일은 `env` 폴더에
- 학습 로그는 `tee`로 `logs/{method}.log`에 남긴다.

실험 관리는 wandb로 한다. 그래서 `config/{method}.yaml`을 실험마다 복사하거나 이름을 바꿔 차이를 설명하지 않고, `WANDB_NOTES` env로 실험별 특징을 적는 걸 선호한다.
로그도 `{method}.log` 하나로 둔다. 복사본을 만들면 나중에 diff, 버전 관리가 어려워지기 때문이다.
