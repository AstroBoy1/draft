#!/usr/bin/env bash
# Decoding parameters, budgets and batch sizes of the eight open-weight models.
#
# Source this file from the repository root, then use the associative arrays below:
#
#   source scripts/local_model_configs.sh
#   for k in "${MODEL_KEYS[@]}"; do
#     "${VENV[$k]}/bin/python" -m chesscheck.query.local \
#       --model "${REPO[$k]}" ${SAMPLING[$k]} ${BUDGET[$k]} \
#       --batch-size "${BATCH[$k]}" \
#       --prompt-file prompts/llm_prompt.txt --num-rows 3000 \
#       --output-dir "$OUTPUT_DIR"
#   done
#
# Sampling: vendor parameters wherever the vendor states them (generation_config.json
# or the model card); greedy only where it states nothing, which is Mistral-7B alone.
# Two deviations, both chosen on parse and truncation rates, never on scores:
#   Llama-3.1-8B  repetition_penalty 1.1 added (Meta states none). Vendor sampling
#                 alone left 41.0% of responses truncated and 41.5% unparseable;
#                 with the penalty 5.0% / 5.0%.
#   Qwen2.5-7B    repetition_penalty raised from the stated 1.05 to 1.1. 1.05 gave
#                 46.2% truncated / 53.6% parsed; 1.1 gives 25.4% / 72.9%.
# Greedy is not a neutral default: every vendor config sets do_sample, Qwen3's card
# warns against it, and it was measured to halve usable output for both Qwen3 models.
#
# Budget: large enough not to bind, not identical. Llama-3-8B's context is 8192.
# Set --max-tokens and --max-model-len together, with the latter above prompt
# (~550 tokens) + max_tokens; setting them equal silently caps the budget.
#
# The NOTE entries record the measurements behind each choice.
# ---------------------------------------------------------------------------

declare -A REPO SAMPLING BUDGET BATCH VENV NOTE

MODEL_KEYS=(qwen25-7b qwen25-14b llama3-8b llama31-8b qwen3-8b qwen35-9b mistral-7b ministral-14b)

# --- Qwen2.5 pair: generation_config.json (card silent) ----------------------
REPO[qwen25-7b]="Qwen/Qwen2.5-7B-Instruct"
# repetition_penalty raised from the vendor's 1.05 to 1.1; see NOTE.
SAMPLING[qwen25-7b]="--temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1"
BUDGET[qwen25-7b]="--max-tokens 16384 --max-model-len 17408"
BATCH[qwen25-7b]=32; VENV[qwen25-7b]=".llm_venv"
NOTE[qwen25-7b]="Heavy looper (68% at cap, 43% looping at t=0/2048). Vendor rp=1.05 measured INSUFFICIENT at the full budget: 46.2% at cap, 53.6% parse, 32.5% loops, mean 8352 tok (n=832). rp=1.1 over the full 2917: 25.4% at cap, 72.9% parse, 19.0% loops, mean 4982 tok. Both rows are in uniform16k_decoding_health_results.csv; cap-2 convention. The unequal n does NOT confound: the rp arm's first 832 rows give 25.8/73.3/19.2, within 0.4pp of its full run, so the vendor arm's prefix is representative. IMPROVED, NOT RESOLVED: contrast Llama-3.1-8B, where rp=1.1 reached 95% parse. Accepted because 72.9% is the band Mistral-7B (71.8%) already ships in. Does NOT transfer to qwen25-14b, whose loop rate RISES with budget -- still untested."

REPO[qwen25-14b]="Qwen/Qwen2.5-14B-Instruct"
SAMPLING[qwen25-14b]="--temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.05"
BUDGET[qwen25-14b]="--max-tokens 16384 --max-model-len 17408"
BATCH[qwen25-14b]=32; VENV[qwen25-14b]=".llm_venv"
NOTE[qwen25-14b]="loop rate RISES with budget (6.5/13.5/15.5% at 2048/4096/8192) -- watch it"

# --- Llama pair: generation_config.json (both cards gated 401) ---------------
REPO[llama3-8b]="meta-llama/Meta-Llama-3-8B-Instruct"
SAMPLING[llama3-8b]="--temperature 0.6 --top-p 0.9"
BUDGET[llama3-8b]="--max-tokens 7400 --max-model-len 8192"
BATCH[llama3-8b]=64; VENV[llama3-8b]=".llm_venv"
NOTE[llama3-8b]="ONLY budget exception: max_position_embeddings is 8192, 16384 impossible. Non-binding anyway (mean 972 tok, 12.3% at cap at 2048). Smaller mml => bigger batch fits."

REPO[llama31-8b]="meta-llama/Meta-Llama-3.1-8B-Instruct"
# repetition_penalty 1.1 added; Meta states none. See NOTE.
SAMPLING[llama31-8b]="--temperature 0.6 --top-p 0.9 --repetition-penalty 1.1"
BUDGET[llama31-8b]="--max-tokens 16384 --max-model-len 17408"
BATCH[llama31-8b]=32; VENV[llama31-8b]=".llm_venv"
NOTE[llama31-8b]="WORST looper. Vendor sampling ALONE was measured insufficient at the full budget: 41.0% at cap, 58.5% parse, 22.0% loops, mean 7531 tok. Adding rp=1.1: 5.0% at cap, 95.0% parse, 3.0% loops, mean 1527 tok, and 4.9x cheaper (~8 GPU-h vs ~38 for 2917). Cost is legal-move RECALL only (f1 0.0395->0.0203 matched n=110), on a metric already at the floor for this model in BOTH arms (closed models run 0.5-0.97). CRITICALLY: the earlier 'rp=1.1 destroys UCI notation' finding (0 of 1495 well-formed moves) does NOT replicate under sampling -- precision 0.0390 vs vendor 0.0408. That collapse was an interaction with GREEDY decoding, not a property of the penalty. Does NOT generalise to qwen25-*: their vendor config already sets rp=1.05, so raising it is a different and smaller kind of deviation. Measured for qwen25-7b over its full 2917-row run: 1.05 does NOT suffice, and 1.1 lifts it only to 72.9% parse -- nowhere near this model's 95%. See NOTE[qwen25-7b]. qwen25-14b remains untested."

# --- Qwen3-8B: config AND card agree ----------------------------------------
REPO[qwen3-8b]="Qwen/Qwen3-8B"
SAMPLING[qwen3-8b]="--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0.0"
BUDGET[qwen3-8b]="--max-tokens 16384 --max-model-len 17408"
BATCH[qwen3-8b]=16; VENV[qwen3-8b]=".llm_venv"
NOTE[qwen3-8b]="May not need re-running: shipped run already had ~15828 ceiling at 6.0% at cap, so 16384 is +3.5% on a non-binding budget. Only argument to re-run is provenance. Long responses (mean 9573) => smaller batch."

# --- Qwen3.5-9B: card ONLY, no generation_config.json exists upstream --------
REPO[qwen35-9b]="Qwen/Qwen3.5-9B"
SAMPLING[qwen35-9b]="--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repetition-penalty 1.0"
BUDGET[qwen35-9b]="--max-tokens 32768 --max-model-len 33792"
BATCH[qwen35-9b]=16; VENV[qwen35-9b]=".llm_venv_qwen35"
NOTE[qwen35-9b]="Budget exception #2: vendor card recommends 32768 output; 16384 would still bind (33% at cap at ~15.8k). SAMPLING SWITCHED 2026-08-30 from the card's thinking+precise-coding row (t=0.6, presence 0) to its thinking+GENERAL-tasks row (t=1.0, presence_penalty 1.5) by measurement: matched 200-puzzle pilot at the same 32k budget, at-cap 33.0%->1.5%, native parse 67.0%->99.0%, loops 9%->0%, mean tokens 15505->8820 (pilot_presence_qwen35_results.csv; pilot_presence_qwen35.sh). Both rows are vendor-stated; this follows the card's own note that presence_penalty reduces endless repetitions. A 1504-row partial run at the coding row is preserved in results/qwen35_9b_coding_row_arm/. NEEDS .llm_venv_qwen35 PLUS two env workarounds, see scripts/install_qwen35.sh. Batch 16 by KV-cache measurement (647,091-token cache = 19.15x concurrency at full 33,792; batch 8 measured 0.39 rows/min). mml verified NOT silently reduced: engine logged 'Using max model len 33792'."

# --- Mistral-7B: the only model with NO vendor opinion at all ----------------
REPO[mistral-7b]="mistralai/Mistral-7B-Instruct-v0.3"
SAMPLING[mistral-7b]="--temperature 0"
BUDGET[mistral-7b]="--max-tokens 16384 --max-model-len 17408"
BATCH[mistral-7b]=32; VENV[mistral-7b]=".llm_venv"
NOTE[mistral-7b]="generation_config.json is auto-generated (_from_model_config: true) and the card is silent, so greedy is the principled fallback -- NOT inherited from Ministral (different model class). Its real problem is formatting: 86.3% clean finish but only 71.8% parse."

# --- Ministral-3-14B: card only, and the card contradicts itself -------------
REPO[ministral-14b]="mistralai/Ministral-3-14B-Reasoning-2512"
SAMPLING[ministral-14b]="--temperature 1"
BUDGET[ministral-14b]="--max-tokens 16384 --max-model-len 17408"
BATCH[ministral-14b]=32; VENV[ministral-14b]=".llm_venv"
NOTE[ministral-14b]="Card prose says 'a temperature of 1'; its three code examples all set TEMP=0.7. Resolved by measurement 2026-08-24: t=1 gives 98.5% usable vs 83% greedy, 0% at cap, 3.8x fewer tokens. We take the prose temperature without the examples' top_p."

# --- Invariants for every model ---------------------------------------------
PROMPT_FILE="prompts/llm_prompt.txt"   # structured JSON prompt the eval pipeline scores
NUM_ROWS=3000                       # 2917 real rows; CSV is 2918 lines incl. header
OUTPUT_DIR="results/raw/local"
# The output file is opened in APPEND mode: point OUTPUT_DIR at an empty directory
# for a fresh run, or resume an interrupted one with --start-row.
#
# Call the venv interpreter by absolute path. Bare `python` on this cluster resolves
# to miniconda and has no vllm.
#
# Batch sizes above are raised from the historical 10: batch 32 measured 2.1x faster
# on a 14B at mml 17408 (A40), with KV cache usage still only ~47%. Size from vLLM's
# startup "GPU KV cache size: N tokens" line -- N / max_model_len is the worst-case
# concurrency and must be >= 1 or vLLM refuses to start.
#
# Resume after a kill: --start-row $(wc -l < out.jsonl) is EXACT for this dataset
# (2917 usable rows, 0 silently skipped, verified). Repair a torn final line first:
# if the file does not end in a newline, truncate back to the last one.
