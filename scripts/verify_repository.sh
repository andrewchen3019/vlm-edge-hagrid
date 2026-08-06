#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m compileall -q src
for script in scripts/*.sh; do bash -n "$script"; done
python3 -m pytest -q
python3 src/tools/sanitize_result_paths.py --check results/image_results

TEMP_SUMMARY="$(mktemp --suffix=.csv)"
trap 'rm -f "$TEMP_SUMMARY"' EXIT
python3 src/analysis/compile_final_results.py --quiet --out "$TEMP_SUMMARY"
python3 - "$TEMP_SUMMARY" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
assert [row["model"] for row in rows] == ["Q8", "Q6", "Q5", "Q4", "Q3"]
assert all(int(row["images_evaluated"]) == 380 for row in rows)
print("Generated summary validated:", path)
PY

echo "Repository verification passed."
