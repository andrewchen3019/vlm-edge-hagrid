#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for quant in q8 q6 q5 q4 q3; do
    echo
    echo "============================================================"
    echo "Running $quant"
    echo "============================================================"
    bash "$REPO_ROOT/scripts/run_benchmark.sh" "$quant"
done
python3 "$REPO_ROOT/src/analysis/compile_final_results.py"
