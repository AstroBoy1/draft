# ChessCheck

Code and the two masters' puzzle annotations for ChessCheck, an evaluation of LLM chess reasoning on Lichess puzzles along five dimensions: factuality, logical consistency, positional judgement, explanation quality and solution correctness.

## Pipeline

Three stages: query a model on 2,917 Lichess puzzles, parse the JSON responses,
score every dimension against ground truth (python-chess, Stockfish 11, two chess
masters' explanations) and build the tables. Each step is a module of the
`chesscheck` package, run from the repository root as `python -m chesscheck.<stage>.<step>`.

## Layout

```
data/        puzzles.csv, puzzle_labels.csv (ground truth), the two masters' commentary
             (CC BY 4.0), its decomposition into claims, raw Stockfish terms
prompts/     the LLM prompt, its paraphrase, the enumeration prompt, the two
             concept-coverage prompts, and the JSON schemas for Claude
chesscheck/  query/ (api, local)  parse/ (responses)  score/ (one module per dimension,
             concept_coverage)  tables/ (bootstrap, dimension_tables, conditioned)
             labels/ (how puzzle_labels.csv was made; not needed)
             models.py lists the 14 models and their names; paths.py every location
scripts/     install.sh, install_qwen35.sh, local_model_configs.sh, run_paper_sweep.sh
results/     everything the pipeline writes (gitignored): raw/, parsed/, scores/, tables/
```

## Setup

Python 3.12. `bash scripts/install.sh` creates `.llm_venv` with the direct
dependencies; `requirements.txt` pins the exact versions used for the paper.
Qwen3.5-9B needs a newer vLLM, so `scripts/install_qwen35.sh` builds a second
venv for it. API keys go in plain-text files `keys/gpt_key`, `keys/gemini_key`,
`keys/claude_key`.

## Stage 1: query

```
python -m chesscheck.query.api --provider gpt    --model-id gpt-5.4           --end-index 2917
python -m chesscheck.query.api --provider gemini --model-id gemini-2.5-pro    --end-index 2917
python -m chesscheck.query.api --provider claude --model-id claude-opus-4-6   --end-index 2917
```

Also gpt-5-mini, gemini-2.5-flash, claude-sonnet-4-6. Resume with `--start-index`;
`--retry-errors` re-requests only failed rows. Open-weight models need a GPU; the
eight models' decoding parameters live in `scripts/local_model_configs.sh`:

```
source scripts/local_model_configs.sh
for k in "${MODEL_KEYS[@]}"; do
  "${VENV[$k]}/bin/python" -m chesscheck.query.local --model "${REPO[$k]}" ${SAMPLING[$k]} ${BUDGET[$k]} \
      --batch-size "${BATCH[$k]}" --prompt-file "$PROMPT_FILE" --num-rows 3000 --output-dir "$OUTPUT_DIR"
done
```

Raw responses land in `results/raw/{api,local}/outputs_<prompt>_<model>.jsonl`.

## Stage 2: parse

```
python -m chesscheck.parse.responses --input-dir results/raw/api
python -m chesscheck.parse.responses --input-dir results/raw/local
```

Writes `results/parsed/parsed_<raw name>`, one row per puzzle with an `error`
field for responses that did not parse. Parsing is native (the paper's
convention); `--repair` adds an LLM fallback. The scorers read every `.jsonl` in
`results/parsed/` (or `PARSED_DIR` / `--parsed-dir`), so keep exactly the 14 files
to report there. A model is recognised by its id anywhere in the file name.

## Stage 3: score and build tables

```
python -m chesscheck.score.factuality      # legal moves F1, material, PV legality
python -m chesscheck.score.consistency     # material, chosen legal, candidates legal, move eval
python -m chesscheck.score.strategic       # mobility, king safety, pawn structure, evaluation
python -m chesscheck.score.explanation     # explanation similarity to the two masters (slow: cross-encoder)
python -m chesscheck.score.correctness     # first-move accuracy, whole-PV correctness
python -m chesscheck.tables.bootstrap           # CIs over puzzles -> results/tables/bootstrap_*.csv
python -m chesscheck.tables.dimension_tables    # one table per dimension -> results/tables/dimension_tables/
```

`bash scripts/run_paper_sweep.sh` runs those seven in order. Each scorer writes
`results/scores/<dimension>_rows.csv` (model x puzzle) and a `_summary.csv`.

Concept coverage is an LLM-judge metric (OpenAI API, resumable, provenance in
sidecar JSONs). Its four subcommands default to the six open-weight models of the
re-run; the other eight were judged by hand with the same prompts, and their
files go in `results/concept_coverage/judged/` named by the token in `models.py`.

```
python -m chesscheck.score.concept_coverage inputs
python -m chesscheck.score.concept_coverage decompose
python -m chesscheck.score.concept_coverage judge
python -m chesscheck.score.concept_coverage summary   # -> results/scores/decomposition_recall_rows.csv
python -m chesscheck.tables.conditioned               # solved vs unsolved, puzzle ratings, LaTeX rows
```

**Other prompts.** Repeat stages 1 and 2 with `--prompt prompts/enumeration_prompt.txt`
(Claude also `--schema prompts/enumeration_prompt_schema.json`; open-weight
`--prompt-file`) or `prompts/llm_prompt_paraphrase.txt`, into a separate
directory, then score with `--parsed-dir` and a different `--output-rows`. The
robustness check uses the first 500 puzzles (`--end-index 500` / `--num-rows 500`).

## Conventions

- A response that failed to parse, or omits the field a metric needs, scores 0.
  Every mean is over all 2,917 puzzles.
- Moves are normalised to UCI before comparison; SAN and piece-prefixed UCI are
  accepted, and a token that resolves to no legal move is scored as illegal.
- Explanation similarity is the cross-encoder score against each master's
  explanation; the tables take the higher of the two.
- Puzzle ratings are the maximum-likelihood Elo rating given the solved set, with
  a bootstrap CI over puzzles.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The masters' annotations (`data/puzzle_commentary.csv`, `data/puzzle_commentary_fm.csv`)
are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
The puzzles come from the Lichess puzzle database, which is in the public domain (CC0).
