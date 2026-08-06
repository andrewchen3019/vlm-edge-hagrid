#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/vlm-edge-project}"
SERVER="${SERVER:-$PROJECT_ROOT/llama.cpp/build/bin/llama-server}"

MODEL="${1:-$PROJECT_ROOT/models/qwen3-vl-4b-custom-quants/Qwen3VL-4B-Instruct-Q4_K_M-self.gguf}"
ALIAS="${2:-qwen3vl4b-q4}"
MMPROJ="${MMPROJ:-$PROJECT_ROOT/models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# 375 prompt tokens + at most 6 output tokens fit within 512.
CTX="${CTX:-512}"

# Safe default for Q8 on an 8 GB Jetson. Use NGL=all only after
# freeing enough RAM and confirming that at least ~500-800 MB remains free.
NGL="${NGL:-all}"
FIT="${FIT:-off}"
FIT_TARGET="${FIT_TARGET:-768}"

# Smaller working buffers reduce transient unified-memory pressure.
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-64}"
MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-128}"

# Quantized KV cache saves additional RAM. Keep this identical across
# all quantization runs for a fair comparison.
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# mmap is fast, but under extreme memory pressure it can page out.
# Test LOAD_MODE=none only if the safe configuration still stalls.
LOAD_MODE="${LOAD_MODE:-mmap}"

VERBOSITY="${VERBOSITY:-3}"

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
echo "Fit:                $FIT"
echo "Fit target:         ${FIT_TARGET} MiB"
echo "Parallel slots:     1"
echo "Batch size:         $BATCH_SIZE"
echo "Microbatch size:    $UBATCH_SIZE"
echo "MTMD batch tokens:  $MTMD_BATCH_TOKENS"
echo "KV cache:           K=$CACHE_TYPE_K V=$CACHE_TYPE_V"
echo "Load mode:          $LOAD_MODE"
echo

exec "$SERVER" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --ctx-size "$CTX" \
    --parallel 1 \
    --no-cont-batching \
    --no-kv-unified \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --mtmd-batch-max-tokens "$MTMD_BATCH_TOKENS" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --no-cache-prompt \
    --cache-ram 0 \
    --ctx-checkpoints 0 \
    --no-cache-idle-slots \
    --fit "$FIT" \
    --fit-target "$FIT_TARGET" \
    --n-gpu-layers "$NGL" \
    --warmup \
    --metrics \
    --mmproj-offload \
    --op-offload \
    --kv-offload \
    --flash-attn on \
    --verbosity "$VERBOSITY"
