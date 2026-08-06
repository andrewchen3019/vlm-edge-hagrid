#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
PYTHON="${PYTHON:-python3}"
if [[ -f "$REPO_ROOT/configs/benchmark.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/configs/benchmark.env"
fi
QUANT="${1:-}"

if [[ -z "$QUANT" ]]; then
    echo "Usage: bash scripts/run_benchmark.sh {q8|q6|q5|q4|q3}" >&2
    exit 2
fi

case "$QUANT" in
    q8) MODEL_FILE="Qwen3VL-4B-Instruct-Q8_0-self.gguf";   ALIAS="qwen3vl4b-q8" ;;
    q6) MODEL_FILE="Qwen3VL-4B-Instruct-Q6_K-self.gguf";   ALIAS="qwen3vl4b-q6" ;;
    q5) MODEL_FILE="Qwen3VL-4B-Instruct-Q5_K_M-self.gguf"; ALIAS="qwen3vl4b-q5" ;;
    q4) MODEL_FILE="Qwen3VL-4B-Instruct-Q4_K_M-self.gguf"; ALIAS="qwen3vl4b-q4" ;;
    q3) MODEL_FILE="Qwen3VL-4B-Instruct-Q3_K_M-self.gguf"; ALIAS="qwen3vl4b-q3" ;;
    *) echo "ERROR: Unknown quantization: $QUANT" >&2; exit 2 ;;
esac

MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/qwen3-vl-4b-custom-quants}"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MMPROJ="${MMPROJ:-$PROJECT_ROOT/models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf}"
METADATA="${METADATA:-$PROJECT_ROOT/data/hagrid_380_resize336/metadata.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/qwen_hagrid_label.txt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE_URL="http://$HOST:$PORT"
OVERWRITE="${OVERWRITE:-0}"
USE_SUDO="${USE_SUDO:-1}"
TEGRASTATS_INTERVAL_MS="${TEGRASTATS_INTERVAL_MS:-1000}"
HASH_MODE="${HASH_MODE:-full}"

STEM="qwen3vl4b_${QUANT}_336px_380_fullgpu"
PREDICTION_CSV="$PROJECT_ROOT/results/image_results/${STEM}.csv"
TEGRA_LOG="$PROJECT_ROOT/results/tegrastat_logs/${STEM}_tegrastats.txt"
SERVER_LOG="$PROJECT_ROOT/results/server_logs/${STEM}_server.log"
MANIFEST="$PROJECT_ROOT/results/manifests/${STEM}_manifest.json"

for command in "$PYTHON" curl sha256sum; do
    command -v "$command" >/dev/null 2>&1 || { echo "ERROR: Missing command: $command" >&2; exit 1; }
done
command -v tegrastats >/dev/null 2>&1 || { echo "ERROR: tegrastats not found" >&2; exit 1; }

for file in "$MODEL_PATH" "$MMPROJ" "$METADATA" "$PROMPT_FILE"; do
    [[ -f "$file" ]] || { echo "ERROR: File not found: $file" >&2; exit 1; }
done

if curl --silent --fail --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
    echo "ERROR: A server is already responding at $BASE_URL" >&2
    exit 1
fi

if [[ "$OVERWRITE" != "1" ]]; then
    for file in "$PREDICTION_CSV" "$TEGRA_LOG" "$SERVER_LOG" "$MANIFEST"; do
        [[ ! -e "$file" ]] || { echo "ERROR: Output exists: $file (set OVERWRITE=1)" >&2; exit 1; }
    done
fi

mkdir -p \
    "$(dirname "$PREDICTION_CSV")" \
    "$(dirname "$TEGRA_LOG")" \
    "$(dirname "$SERVER_LOG")" \
    "$(dirname "$MANIFEST")"
rm -f "$PREDICTION_CSV" "$TEGRA_LOG" "$SERVER_LOG" "$MANIFEST"

SERVER_PID=""
TEGRA_PID=""
stop_processes() {
    set +e
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    if [[ -n "$TEGRA_PID" ]] && kill -0 "$TEGRA_PID" 2>/dev/null; then
        if [[ "$USE_SUDO" == "1" ]]; then
            sudo kill "$TEGRA_PID" 2>/dev/null
        else
            kill "$TEGRA_PID" 2>/dev/null
        fi
        wait "$TEGRA_PID" 2>/dev/null
    fi
    SERVER_PID=""
    TEGRA_PID=""
}
trap stop_processes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$USE_SUDO" == "1" ]]; then
    sudo -v
    sudo tegrastats --interval "$TEGRASTATS_INTERVAL_MS" --logfile "$TEGRA_LOG" >/dev/null 2>&1 &
else
    tegrastats --interval "$TEGRASTATS_INTERVAL_MS" --logfile "$TEGRA_LOG" >/dev/null 2>&1 &
fi
TEGRA_PID=$!

PROJECT_ROOT="$PROJECT_ROOT" \
HOST="$HOST" PORT="$PORT" MMPROJ="$MMPROJ" \
NGL="${NGL:-all}" FIT="${FIT:-off}" CTX="${CTX:-512}" \
BATCH_SIZE="${BATCH_SIZE:-128}" UBATCH_SIZE="${UBATCH_SIZE:-64}" \
MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-128}" \
bash "$PROJECT_ROOT/scripts/start_qwen3vl_server.sh" \
    "$MODEL_PATH" "$ALIAS" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 150); do
    if curl --silent --fail --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited during startup" >&2
        tail -100 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 2
done
[[ "$ready" == "1" ]] || { echo "ERROR: Server did not become ready" >&2; exit 1; }

"$PYTHON" "$PROJECT_ROOT/src/evaluation/eval_qwen3vl_server.py" \
    --metadata "$METADATA" \
    --prompt-file "$PROMPT_FILE" \
    --model "$ALIAS" \
    --base-url "$BASE_URL" \
    --per-class 20 \
    --max-tokens 6 \
    --warmup 1 \
    --seed 0 \
    --flush-every 10 \
    --output-path-root "$PROJECT_ROOT" \
    --overwrite \
    --out "$PREDICTION_CSV"

stop_processes
sleep 1

MODEL_SHA=""
MMPROJ_SHA=""
if [[ "$HASH_MODE" == "full" ]]; then
    MODEL_SHA="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
    MMPROJ_SHA="$(sha256sum "$MMPROJ" | awk '{print $1}')"
fi
PROMPT_SHA="$(sha256sum "$PROMPT_FILE" | awk '{print $1}')"
METADATA_SHA="$(sha256sum "$METADATA" | awk '{print $1}')"
REPO_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)"
LLAMA_COMMIT="$(git -C "$PROJECT_ROOT/llama.cpp" rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

export QUANT ALIAS MODEL_PATH MODEL_SHA MMPROJ MMPROJ_SHA PROMPT_FILE PROMPT_SHA
export METADATA METADATA_SHA PREDICTION_CSV TEGRA_LOG SERVER_LOG
export REPO_COMMIT LLAMA_COMMIT GIT_DIRTY HOST PORT
export CTX="${CTX:-512}" NGL="${NGL:-all}" FIT="${FIT:-off}"
export BATCH_SIZE="${BATCH_SIZE:-128}" UBATCH_SIZE="${UBATCH_SIZE:-64}"
export MTMD_BATCH_TOKENS="${MTMD_BATCH_TOKENS:-128}"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = Path(sys.argv[1])
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "quantization_key": os.environ["QUANT"],
    "model_alias": os.environ["ALIAS"],
    "model_path": os.environ["MODEL_PATH"],
    "model_sha256": os.environ["MODEL_SHA"] or None,
    "mmproj_path": os.environ["MMPROJ"],
    "mmproj_sha256": os.environ["MMPROJ_SHA"] or None,
    "prompt_file": os.environ["PROMPT_FILE"],
    "prompt_sha256": os.environ["PROMPT_SHA"],
    "metadata_file": os.environ["METADATA"],
    "metadata_sha256": os.environ["METADATA_SHA"],
    "prediction_csv": os.environ["PREDICTION_CSV"],
    "tegrastats_log": os.environ["TEGRA_LOG"],
    "server_log": os.environ["SERVER_LOG"],
    "repository_commit": os.environ["REPO_COMMIT"] or None,
    "llama_cpp_commit": os.environ["LLAMA_COMMIT"] or None,
    "repository_dirty_file_count": int(os.environ["GIT_DIRTY"] or 0),
    "server": {"host": os.environ["HOST"], "port": int(os.environ["PORT"])},
    "benchmark": {
        "resolution_px": 336,
        "classes": 19,
        "images_per_class": 20,
        "images_total": 380,
        "max_tokens": 6,
        "warmup_requests": 1,
        "seed": 0,
    },
    "llama_server": {
        "ctx": os.environ["CTX"],
        "n_gpu_layers": os.environ["NGL"],
        "fit": os.environ["FIT"],
        "batch_size": os.environ["BATCH_SIZE"],
        "ubatch_size": os.environ["UBATCH_SIZE"],
        "mtmd_batch_tokens": os.environ["MTMD_BATCH_TOKENS"],
    },
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "Benchmark complete"
echo "Predictions: $PREDICTION_CSV"
echo "tegrastats:  $TEGRA_LOG"
echo "Server log:  $SERVER_LOG"
echo "Manifest:    $MANIFEST"
