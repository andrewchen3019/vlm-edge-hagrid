#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/vlm-edge-project}"
SERVER="${SERVER:-$PROJECT_ROOT/llama.cpp/build/bin/llama-server}"
MODEL="${1:-$PROJECT_ROOT/models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-Q4_K_M-self.gguf}"
ALIAS="${2:-qwen3vl4b-q4}"
MMPROJ="${MMPROJ:-$PROJECT_ROOT/models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-512}"
NGL="${NGL:-all}"
FIT="${FIT:-off}"
FIT_TARGET="${FIT_TARGET:-768}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-64}"
MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-128}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
LOAD_MODE="${LOAD_MODE:-mmap}"
VERBOSITY="${VERBOSITY:-3}"

case "$FIT" in on|off) ;; *) echo "ERROR: FIT must be on or off" >&2; exit 2 ;; esac
case "$LOAD_MODE" in none|mmap|mlock|dio) ;; *) echo "ERROR: invalid LOAD_MODE=$LOAD_MODE" >&2; exit 2 ;; esac

for name in PORT CTX PARALLEL BATCH_SIZE UBATCH_SIZE MTMD_BATCH_TOKENS VERBOSITY; do
    value="${!name}"
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "ERROR: $name must be an integer" >&2; exit 2; }
done

for file in "$SERVER" "$MODEL" "$MMPROJ"; do
    [[ -f "$file" ]] || { echo "ERROR: File not found: $file" >&2; exit 1; }
done
[[ -x "$SERVER" ]] || { echo "ERROR: Server is not executable: $SERVER" >&2; exit 1; }

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

if [[ "$FIT" == "on" ]]; then
    [[ "$FIT_TARGET" =~ ^[0-9]+$ ]] || { echo "ERROR: FIT_TARGET must be an integer" >&2; exit 2; }
    ARGS+=(--fit-target "$FIT_TARGET")
fi

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
    "Load mode:" "$LOAD_MODE"
[[ "$FIT" == "on" ]] && printf '%-20s %s\n' "Fit target:" "${FIT_TARGET} MiB"
echo
exec "$SERVER" "${ARGS[@]}"
