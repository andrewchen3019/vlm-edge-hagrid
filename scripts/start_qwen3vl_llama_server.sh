#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/vlm-edge-project}"

SERVER="${SERVER:-$PROJECT_ROOT/llama.cpp/build/bin/llama-server}"

MODEL="${1:-$PROJECT_ROOT/models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-Q4_K_M-self.gguf}"

ALIAS="${2:-qwen3vl4b-q4}"

MMPROJ="${MMPROJ:-$PROJECT_ROOT/models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# Keep 1024 initially. Your measured prompt uses about 375 tokens,
# so this provides sufficient headroom.
CTX="${CTX:-1024}"

# "all" requests full language-model layer offload.
NGL="${NGL:-all}"

# One slot gives clean serial latency measurements.
PARALLEL="${PARALLEL:-1}"

# Reduced batching lowers peak memory use on the Jetson.
BATCH_SIZE="${BATCH_SIZE:-256}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-256}"

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

echo "Starting Qwen3-VL llama-server"
echo "Server:             $SERVER"
echo "Model:              $MODEL"
echo "mmproj:             $MMPROJ"
echo "Alias:              $ALIAS"
echo "Address:            http://$HOST:$PORT"
echo "Context:            $CTX"
echo "GPU layers:         $NGL"
echo "Parallel slots:     $PARALLEL"
echo "Batch size:         $BATCH_SIZE"
echo "Microbatch size:    $UBATCH_SIZE"
echo "MTMD batch tokens:  $MTMD_BATCH_TOKENS"
echo

exec "$SERVER" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --mtmd-batch-max-tokens "$MTMD_BATCH_TOKENS" \
    --no-cache-prompt \
    --cache-ram 0 \
    --ctx-checkpoints 0 \
    --no-cache-idle-slots \
    --fit off \
    --warmup \
    --metrics \
    --n-gpu-layers "$NGL" \
    --mmproj-offload \
    --op-offload \
    --kv-offload \
    --flash-attn on