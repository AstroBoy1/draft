"""Positional judgement: the four Stockfish-derived labels, each a 3-way match.

    mobility_accuracy, king_safety_accuracy, pawn_structure_accuracy, initial_eval_accuracy

The columns are taken from the factuality scorer's rows and regrouped.
"""
import pandas as pd

from .common import cli, conditioned_summary, input_files, write_outputs
from .factuality import STRATEGIC_COLS, summarize as factuality_rows


def summarize(path):
    out = factuality_rows(path)
    out = out[["llm_file", "source", "row_index", "first_move_accuracy"] + STRATEGIC_COLS].copy()
    out["strategic_score"] = out[STRATEGIC_COLS].mean(axis=1, skipna=True)
    return out


def main():
    args = cli("strategic", __doc__).parse_args()
    rows = pd.concat([summarize(f) for f in input_files(args)], ignore_index=True)
    write_outputs(rows, conditioned_summary(rows, "strategic_score", STRATEGIC_COLS), args)


if __name__ == "__main__":
    main()
