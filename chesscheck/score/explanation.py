"""Explanation quality: cross-encoder similarity of final_explanation_summary to
the national master's (`similarity_nm`, also `similarity`) and the FIDE
master's (`similarity_fm`) explanation. Concept coverage is the other half of
this dimension (concept_coverage.py). The slowest scorer.
"""
import numpy as np
import pandas as pd

from .. import references
from .common import cli, input_files, labels, load_predictions, meta_columns, write_outputs


def summarize(path):
    p = load_predictions(path)
    lab = labels()
    n = len(lab)
    sim = references.similarity_by_source(references.cross_encoder(), p.explanation,
                                          references.load_labeled_reference_lists(lab["PuzzleId"]))
    out = pd.DataFrame({
        **meta_columns(path),
        **{f"similarity_{src}": (v + [np.nan] * n)[:n] for src, v in sim.items()},
    })
    references.apply_failure_rule(out)
    return out[["llm_file", "source", "row_index", "similarity",
                *[f"similarity_{src}" for src in references.SOURCES]]]


def summary(rows_df):
    g = rows_df.groupby(["source", "llm_file"], dropna=False)
    return g.agg(**{f"avg_similarity_{src}": (f"similarity_{src}", "mean") for src in references.SOURCES},
                 **{f"n_{src}": (f"similarity_{src}", "count") for src in references.SOURCES}).reset_index()


def main():
    args = cli("explanation", __doc__).parse_args()
    files = input_files(args)
    for f in files:
        print(f"Processing {f.name}")
    rows = pd.concat([summarize(f) for f in files], ignore_index=True)
    write_outputs(rows, summary(rows), args)


if __name__ == "__main__":
    main()
