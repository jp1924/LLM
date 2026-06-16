export LLM_CACHE_ROOT="${LLM_CACHE_ROOT:-/root/.cache/runtime}"
export TMPDIR="${TMPDIR:-$LLM_CACHE_ROOT/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LLM_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LLM_CACHE_ROOT/triton}"

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# src/ 를 import 경로에 추가 (processing_utils 등 공통 모듈 해석용)
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

accelerate launch \
    --config-file ./config/accelerator/zero3.yaml \
    ./src/dpo/main.py \
    --config ./config/dpo.yaml \
    "$@" 2>&1 | tee ./logs/dpo.log
