#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSEEK_REPO="deepseek-ai/DeepSeek-Prover-V2-7B"
DEEPSEEK_PROVER_DIR="${DEEPSEEK_PROVER_DIR:-$SCRIPT_DIR/models/${DEEPSEEK_REPO##*/}}"

# Check if uv is installed and on path
if ! command -v uv >/dev/null 2>&1; then
    if [[ -t 0 ]]; then
        read -r -p "==> uv is required and not found. Install it now? [y/N] " answer || answer="n"
    else
        answer="n"
    fi
    if [[ "$answer" =~ ^[Yy] ]]; then
        # trusting the official astral download
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        hash -r
        command -v uv >/dev/null || {
            echo "==> uv installed but not on PATH. Open a new shell and re-run" >&2
            exit 1
        }
    else
        echo "==> uv is required. Install from https://astral.sh/uv/install.sh and re-run this script" >&2
        exit 1
    fi
else
    echo "==> uv found at $(command -v uv)" >&2
fi

# Download DeepSeek-Prover-V2-7B
echo "==> downloading ${DEEPSEEK_REPO##*/} (~14 GB)..." >&2
HF_HUB_ENABLE_HF_TRANSFER=1 \
    uv run hf download "$DEEPSEEK_REPO" --local-dir "$DEEPSEEK_PROVER_DIR" --quiet >/dev/null
echo "==> ${DEEPSEEK_REPO##*/} stored at $DEEPSEEK_PROVER_DIR" >&2
