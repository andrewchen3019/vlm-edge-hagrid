#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO_ROOT/docs/environment.generated.md}"
mkdir -p "$(dirname "$OUT")"

capture() {
    local title="$1"
    shift
    {
        echo "## $title"
        echo
        echo '```text'
        "$@" 2>&1 || true
        echo '```'
        echo
    } >> "$OUT"
}

cat > "$OUT" <<EOF
# Experiment Environment

Generated: $(date --iso-8601=seconds)

Repository commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
llama.cpp commit: $(git -C "$REPO_ROOT/llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)

EOF

capture "Kernel" uname -a
if [[ -f /etc/nv_tegra_release ]]; then
    capture "Jetson Linux" cat /etc/nv_tegra_release
fi
capture "JetPack packages" bash -lc "dpkg-query -W 'nvidia-jetpack*' 'nvidia-l4t-core' 2>/dev/null"
if command -v nvcc >/dev/null 2>&1; then
    capture "CUDA compiler" nvcc --version
fi
capture "Python" python3 --version
capture "Pip" python3 -m pip --version
capture "Installed Python packages" python3 -m pip freeze
if command -v nvpmodel >/dev/null 2>&1; then
    capture "Jetson power mode" sudo nvpmodel -q
fi
if [[ -x "$REPO_ROOT/llama.cpp/build/bin/llama-server" ]]; then
    capture "llama-server version" "$REPO_ROOT/llama.cpp/build/bin/llama-server" --version
fi
capture "Git submodule status" git -C "$REPO_ROOT" submodule status
capture "Git working tree" git -C "$REPO_ROOT" status --short

echo "Wrote $OUT"
