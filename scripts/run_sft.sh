export LLM_CACHE_ROOT="${LLM_CACHE_ROOT:-/root/.cache/runtime}"
export TMPDIR="${TMPDIR:-$LLM_CACHE_ROOT/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LLM_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LLM_CACHE_ROOT/triton}"

mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

accelerate launch \
    --config-file ./config/accelerator/zero3.yaml \
    ./src/sft/main.py \
    --config ./config/sft.yaml \
    "$@" 2>&1 | tee ./logs/sft.log
