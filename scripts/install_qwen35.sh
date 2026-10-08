# One-time setup for a separate venv to run Qwen3.5 models.
#
# Qwen3.5's `qwen3_5` architecture isn't recognized by the vllm/transformers
# versions pinned in .llm_venv (see scripts/install.sh), and bumping those in place
# would change behavior for every other model run out of .llm_venv. This
# installs a newer vllm (which bundles a newer transformers) into its own venv
# instead.
#
# Run this from the repo root, with the main .llm_venv already set up
# (scripts/install.sh) so UV_CACHE_DIR / TMPDIR are exported.

uv venv .llm_venv_qwen35 --python 3.12 --seed
source .llm_venv_qwen35/bin/activate

uv pip install vllm --torch-backend=cu129
uv pip install chess

deactivate

echo "Done. Run Qwen3.5 with VENV[qwen35-9b] from scripts/local_model_configs.sh"
