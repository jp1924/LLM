# SFT

LLM/LMM의 SFT와 Pretrain 학습을 위한 코드 모음.<br/>
Transformers, TRL, DeepSpeed, FSDP 등 오픈소스 라이브러리를 직접 다루며 디버깅하고, PR을 날리기 위한 기준 코드로도 활용한다.

**설계 목표:**

- `main.py` 코드를 건드리지 않고 yaml 파일만 바꿔서 학습을 돌릴 수 있도록 한다.
- SFT와 Pretrain을 하나의 `main.py`로 처리한다 (loss 계산 방식만 다름).
- 학습 중 lm-eval-harness 평가를 callback으로 수행해 별도 평가 스크립트 없이도 성능을 추적한다.
- HuggingFace Hub에 올라간 데이터셋을 기준으로 작동한다.

---

## 파일 설명

```
sft
 ┣ callbacks.py    # lm-eval-harness를 학습 중 실행하는 EvalHarnessCallBack 정의
 ┣ main.py         # SFTScriptArguments, PackingCollatorForLLM, 학습 진입점
 ┣ optimization.py # better_cosine 등 커스텀 LR 스케줄러 정의
 ┣ preprocessor.py # sft_processor, pretrain_processor 및 processing_datasets 정의
 └ README.md
```

---

## 워크플로우

```
HuggingFace Hub에 데이터 업로드
        ↓
yaml 파일로 args 설정 (dataset_repo_ls, data_preprocessor_type 등)
        ↓
torchrun / accelerate launch main.py --config config.yaml
        ↓
(선택) 학습 중 lm-eval-harness 자동 평가 (eval_harness_tasks 설정 시)
```

---

## 데이터 전처리

`data_preprocessor_type`으로 전처리 방식을 선택한다.

### SFT (`data_preprocessor_type: sft`)

아래 우선순위로 대화 형식을 자동 감지해서 처리한다.

| 우선순위 | 감지 조건                                                                                                              | 비고      |
| -------- | :--------------------------------------------------------------------------------------------------------------------- | --------- |
| 1        | `conversations` / `conversation` / `messages` 컬럼 (role+content 형식)                                                 | 권장 형식 |
| 2        | TRL conversational 형식 (`messages`, `prompt`+`completion` 등)                                                         | TRL 호환  |
| 3        | 단순 QA 쌍 (`prompt`/`question`/`input`/`instruction` + `completion`/`answer`/`label`/`output`/`response`/`assistant`) | 자동 변환 |

- `images` 컬럼이 있으면 멀티모달(LMM) 학습으로 자동 처리된다.
- assistant 턴만 loss를 계산하도록 레이블을 생성한다 (나머지는 `-100`).

### Pretrain (`data_preprocessor_type: pretrain`)

| 컬럼          | 설명                          |
| ------------- | :---------------------------- |
| `sentence`    | 단일 문자열                   |
| `sentence_ls` | 문자열 리스트 (배치 tokenize) |

- BOS/EOS 토큰을 자동으로 추가한다.
- `input_ids` 전체를 label로 사용한다 (CLM).

### 전처리 캐시

전처리 결과는 `$HF_DATASETS_CACHE/preprocess_cache/{model_name}/{run_name}/{repo_name}/` 아래에 Arrow 파일로 캐시된다.<br/>
재실행 시 캐시가 있으면 전처리를 건너뛴다.

---

## Arguments

`SFTScriptArguments`는 TRL의 `SFTConfig`와 `ModelConfig`를 다중 상속해서 하나의 args에서 모든 설정을 관리한다.<br/>
(wandb에도 TrainingArguments 하나만 기록되기 때문에 별도 dataclass로 분리하지 않음)

### 데이터 관련 Args

| 인자                        | 타입        | 기본값 | 설명                                                                                                          |
| :-------------------------- | ----------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| `dataset_repo_ls`           | `List[str]` | `[]`   | 학습에 사용할 HuggingFace Hub repo 이름 목록. 예: `["org/repo1", "org/repo2"]`                                |
| `data_preprocessor_type`    | `str`       | —      | 전처리 방식 선택. `sft` 또는 `pretrain`                                                                       |
| `preprocessing_num_workers` | `int`       | `5`    | 전처리에 사용할 프로세스 수. `1`이면 싱글 프로세스로 동작                                                     |
| `preprocessing_batch_size`  | `int`       | `1000` | `datasets.map` 호출 시 배치 크기                                                                              |
| `dataset_prefix`            | `dict`      | `{}`   | split 이름을 train/valid/test 로 분류하기 위한 prefix 맵. 예: `{"train": ["train"], "valid": ["validation"]}` |
| `data_truncate_map`         | `dict`      | `{}`   | repo별 샘플 수 제한. 예: `{"org/repo1": {"train": 10000, "validation": 1000}}`                                |
| `data_name_map`             | `dict`      | `{}`   | repo별 datasets config name 지정. 예: `{"org/repo1": "ko"}`                                                   |

### 모델 / 토크나이저 관련 Args

| 인자                  | 타입   | 기본값 | 설명                                                                                                                         |
| --------------------- | :----- | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `model_name_or_path`  | `str`  | —      | 학습할 모델의 Hub repo 또는 로컬 경로 (ModelConfig에서 상속)                                                                 |
| `chat_template`       | `str`  | `None` | 토크나이저에 적용할 chat template. 설정 시 `tokenizer_kwargs`에 자동 포함됨                                                  |
| `config_kwargs`       | `dict` | `{}`   | `AutoConfig.from_pretrained`에 전달할 추가 kwargs. `attn_implementation`, `use_cache`는 자동 설정됨                          |
| `tokenizer_kwargs`    | `dict` | `{}`   | `AutoTokenizer` / `AutoProcessor` 로딩 시 전달할 추가 kwargs. `trust_remote_code`, `revision`은 자동 설정됨                  |
| `attn_implementation` | `str`  | —      | attention 구현체 선택. `flash_attention_2`, `sdpa`, `eager` (ModelConfig에서 상속). packing 사용 시 `flash_attention_2` 필요 |
| `torch_dtype`         | `str`  | —      | 모델 dtype. `bfloat16`, `float16`, `float32`, `auto` (ModelConfig에서 상속)                                                  |

### 평가 관련 Args

| 인자                 | 타입        | 기본값  | 설명                                                                                                                       |
| -------------------- | :---------- | ------- | -------------------------------------------------------------------------------------------------------------------------- |
| `eval_harness_tasks` | `List[str]` | `None`  | 학습 중 실행할 lm-eval-harness 태스크 리스트. 설정 시 `EvalHarnessCallBack`이 자동 등록됨. 예: `["hellaswag", "arc_easy"]` |
| `eval_on_start`      | `bool`      | `False` | 학습 시작 전(step 0) lm-eval-harness 평가 수행 여부                                                                        |

### Sequence Parallelism (SP)

`accelerate`의 `ParallelismConfig`를 통해 SP가 활성화되면 아래 동작이 자동으로 적용된다.

- `max_length`가 `world_size`의 배수인지 검증
- `max_length *= per_device_train_batch_size`, `per_device_train_batch_size = 1`로 자동 조정
- Collator에서도 `world_size` 배수가 되도록 자동 패딩

---

## lm-eval-harness Callback

`eval_harness_tasks`를 설정하면 `EvalHarnessCallBack`이 자동으로 등록된다.<br/>
`eval_steps` 주기마다 현재 학습 중인 모델을 그대로 사용해 lm-eval-harness를 실행하고 wandb에 기록한다.<br/>
별도 체크포인트 저장 후 평가하는 과정 없이 학습 중 실시간 성능 추적이 가능하다.

- Multi-GPU 환경에서 GPU 간 동기화 문제(요청 수 불일치)를 패딩으로 해결한 `CustomHFLM`을 내부적으로 사용한다.
- 평가 후 모델은 자동으로 `train()` 모드로 복귀한다.
- 평가 결과는 `test_{task_name}` 키로 wandb에 로깅된다.

---

## 사용 예시

### yaml 설정 파일 예시 (SFT)

```yaml
# modeling
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
attn_implementation: flash_attention_2
torch_dtype: bfloat16

# data preprocessing
dataset_repo_ls:
  - org/my-sft-dataset
data_preprocessor_type: sft
dataset_prefix:
  train: [train]
  valid: [validation]
data_truncate_map:
  org/my-sft-dataset:
    train: 50000
max_length: 4096

# training
output_dir: ./outputs/qwen2.5-7b-sft
run_name: qwen2.5-7b-sft-v1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
num_train_epochs: 3
packing: true
gradient_checkpointing: true

# lm eval harness와 trainer eval loop
eval_strategy: steps
eval_steps: 500
eval_harness_tasks:
  - hellaswag
  - arc_easy

# logging
report_to: wandb
seed: 42
```

### yaml 설정 파일 예시 (Pretrain)

```yaml
model_name_or_path: meta-llama/Llama-3.1-8B
attn_implementation: flash_attention_2
torch_dtype: bfloat16

dataset_repo_ls:
  - org/my-pretrain-corpus
data_preprocessor_type: pretrain
dataset_prefix:
  train: [train]
max_length: 8192

output_dir: ./outputs/llama3.1-8b-pretrain
run_name: llama3.1-8b-pretrain-v1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 3.0e-4
lr_scheduler_type: better_cosine
warmup_ratio: 0.01
max_steps: 100000
packing: true
gradient_checkpointing: true

report_to: wandb
seed: 42
```

### 실행

```bash
# 단일 GPU
python main.py config.yaml

# 멀티 GPU (torchrun)
torchrun --nproc_per_node=8 main.py config.yaml

# accelerate
accelerate launch --config_file accelerate_config.yaml main.py config.yaml
```
