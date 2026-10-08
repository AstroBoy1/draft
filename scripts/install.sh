#!/usr/bin/env bash
# Create .llm_venv with uv. Run from the repository root.
# requirements.txt has the exact versions used for the paper.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Point these at a filesystem with space on a cluster.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export TMPDIR="${TMPDIR:-$HOME/.tmp}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"

uv venv .llm_venv --python 3.12 --seed
source .llm_venv/bin/activate
uv pip install vllm --torch-backend=cu129
uv pip install chess pandas numpy scipy sentence_transformers zstandard
uv pip install openai google-genai anthropic

# API keys: keys/gpt_key, keys/gemini_key, keys/claude_key
mkdir -p keys
echo "Done. Activate with: source .llm_venv/bin/activate"
