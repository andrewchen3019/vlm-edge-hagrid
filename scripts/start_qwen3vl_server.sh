#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/vlm-edge-project}"
SERVER="${SERVER:-$PROJECT_ROOT/llama.cpp/build/bin/llama-server}"

MODEL="${1:-$PROJECT_ROOT/models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-Q4_K_M-self.gguf}"
ALIAS="${2:-qwen3vl4b-q4}"
MMPROJ="${MMPROJ:-$PROJECT_ROOT/models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# Final benchmark configuration.
CTX="${CTX:-512}"
NGL="${NGL:-all}"
FIT="${FIT:-off}"
FIT_TARGET="${FIT_TARGET:-768}"

PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-64}"
MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-128}"

# Keep these identical across quantization runs for a fair comparison.
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# Supported values: none, mmap, mlock, dio.
LOAD_MODE="${LOAD_MODE:-mmap}"
VERBOSITY="${VERBOSITY:-3}"

case "$FIT" in
    on|off)
        ;;
    *)
        echo "ERROR: FIT must be 'on' or 'off'; got: $FIT" >&2
        exit 2
        ;;
esac

case "$LOAD_MODE" in
    none|mmap|mlock|dio)
        ;;
    *)
        echo \
            "ERROR: LOAD_MODE must be one of: none, mmap, mlock, dio; got: $LOAD_MODE" \
            >&2
        exit 2
        ;;
esac

# Validate numeric settings.
for value_name in \
    PORT \
    CTX \
    PARALLEL \
    BATCH_SIZE \
    UBATCH_SIZE \
    MTMD_BATCH_TOKENS \
    VERBOSITY
do
    value="${!value_name}"

    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo \
            "ERROR: $value_name must be a non-negative integer; got: $value" \
            >&2
        exit 2
    fi
done

# Validate required files.
for file in "$SERVER" "$MODEL" "$MMPROJ"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: File not found: $file" >&2
        exit 1
    fi
done

if [[ ! -x "$SERVER" ]]; then
    echo "ERROR: Server is not executable: $SERVER" >&2
    exit 1
fi

ARGS=(
    --model "$MODEL"
    --mmproj "$MMPROJ"
    --alias "$ALIAS"

    --host "$HOST"
    --port "$PORT"

    --jinja

    --ctx-size "$CTX"
    --parallel "$PARALLEL"
    --no-cont-batching
    --no-kv-unified

    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --mtmd-batch-max-tokens "$MTMD_BATCH_TOKENS"

    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --no-cache-prompt
    --cache-ram 0
    --ctx-checkpoints 0
    --no-cache-idle-slots

    --fit "$FIT"
    --n-gpu-layers "$NGL"
    --load-mode "$LOAD_MODE"

    --warmup
    --metrics

    --mmproj-offload
    --op-offload
    --kv-offload
    --flash-attn on

    --verbosity "$VERBOSITY"
)

# --fit-target only applies when automatic fitting is enabled.
if [[ "$FIT" == "on" ]]; then
    if [[ ! "$FIT_TARGET" =~ ^[0-9]+$ ]]; then
        echo \
            "ERROR: FIT_TARGET must be a non-negative integer; got: $FIT_TARGET" \
            >&2
        exit 2
    fi

    ARGS+=(--fit-target "$FIT_TARGET")
fi

echo "Starting Qwen3-VL llama-server"

printf '%-20s %s\n' \
    "Server:" "$SERVER" \
    "Model:" "$MODEL" \
    "mmproj:" "$MMPROJ" \
    "Alias:" "$ALIAS" \
    "Address:" "http://$HOST:$PORT" \
    "Context:" "$CTX" \
    "GPU layers:" "$NGL" \
    "Fit:" "$FIT" \
    "Parallel slots:" "$PARALLEL" \
    "Batch size:" "$BATCH_SIZE" \
    "Microbatch size:" "$UBATCH_SIZE" \
    "MTMD batch tokens:" "$MTMD_BATCH_TOKENS" \
    "KV cache:" "K=$CACHE_TYPE_K V=$CACHE_TYPE_V" \
    "Load mode:" "$LOAD_MODE" \
    "Verbosity:" "$VERBOSITY"

if [[ "$FIT" == "on" ]]; then
    printf '%-20s %s\n' \
        "Fit target:" "${FIT_TARGET} MiB"
fi

echo
exec "$SERVER" "${ARGS[@]}"