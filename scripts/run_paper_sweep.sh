#!/usr/bin/env bash
# Score the five dimensions, bootstrap the CIs and build the tables from the
# parsed files in results/parsed/ (or $PARSED_DIR). Takes hours; the explanation
# scorer runs a cross-encoder over every explanation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-$ROOT/.llm_venv/bin/python}"
[[ -x "$PY" ]] || PY=python
PARSED="${PARSED_DIR:-results/parsed}"
n=$(ls "$PARSED"/*.jsonl 2>/dev/null | wc -l)
echo "SWEEP START $(date '+%F %T')  $PARSED has $n files"
for step in factuality consistency strategic explanation correctness; do
    echo "STEP START $step $(date '+%T')"
    "$PY" -m chesscheck.score.$step > /dev/null
    echo "STEP DONE  $step $(date '+%T')"
done
echo "STEP START bootstrap $(date '+%T')"; "$PY" -m chesscheck.tables.bootstrap > /dev/null; echo "STEP DONE  bootstrap $(date '+%T')"
echo "STEP START dimension_tables $(date '+%T')"; "$PY" -m chesscheck.tables.dimension_tables; echo "STEP DONE  dimension_tables $(date '+%T')"
echo "SWEEP DONE $(date '+%F %T')"
