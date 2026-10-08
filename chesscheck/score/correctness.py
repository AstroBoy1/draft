"""Solution correctness.

    first_move_accuracy    the chosen line opens with the solution move
    best_move_pv_correct   the whole line matches the solution from move 2, over the
                           shorter length

Both 1/0 per puzzle; a failed or line-less response scores 0.
"""
import pandas as pd

from ..scoring_rule import count_failures_as_wrong
from .common import (best_move_pv_matches_solution_rowwise, cli, first_move_accuracy_rowwise,
                     input_files, labels, load_predictions, meta_columns, write_outputs)

CORRECTNESS_COLS = ["first_move_accuracy", "best_move_pv_correct"]


def summarize(path):
    p = load_predictions(path)
    lab = labels()
    out = pd.DataFrame({
        **meta_columns(path),
        "first_move_accuracy": first_move_accuracy_rowwise(p.best_move_pv, lab["Moves"]),
        "best_move_pv_correct": best_move_pv_matches_solution_rowwise(p.best_move_pv, lab["Moves"], p.ok),
    })
    count_failures_as_wrong(out, CORRECTNESS_COLS)
    return out


def summary(rows_df):
    g = rows_df.groupby(["source", "llm_file"], dropna=False)
    out = g.agg(n_puzzles=("first_move_accuracy", "size"), n_solved=("first_move_accuracy", "sum"),
                first_move_accuracy=("first_move_accuracy", "mean"),
                best_move_pv_correct=("best_move_pv_correct", "mean")).reset_index()
    solved = rows_df[rows_df["first_move_accuracy"] == 1].groupby(["source", "llm_file"], dropna=False)
    out = out.merge(solved["best_move_pv_correct"].mean().reset_index(name="pv_correct_given_solved"),
                    on=["source", "llm_file"], how="left")
    out["n_solved"] = out["n_solved"].astype(int)
    return out


def main():
    args = cli("correctness", __doc__).parse_args()
    rows = pd.concat([summarize(f) for f in input_files(args)], ignore_index=True)
    write_outputs(rows, summary(rows), args)


if __name__ == "__main__":
    main()
