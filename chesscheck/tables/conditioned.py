"""Results on solved versus unsolved puzzles, puzzle ratings, and the LaTeX rows.

    python -m chesscheck.tables.conditioned [--write-tex paper/latex/acl_latex.tex]

Writes results/tables/conditioned_by_first_move.csv (per model: n_correct and
each sub-metric's mean over puzzles whose first move was correct, _C, and
wrong, _W; similarity_max is the per-puzzle max over the two masters) and
puzzle_ratings.csv (puzzle_rating: the maximum-likelihood Elo rating given the
solved set, with a bootstrap CI; puzzle_rating_sequential: one K=10 Elo update
per puzzle in file order, kept for reference).

Then prints the rows of four LaTeX tables: tab:conditioned_results and
tab:conditioned_positional (C and W columns, every model with at least one
solve) and tab:solved_factuality and tab:solved_positional (solved puzzles only,
models with at least MIN_SOLVED_BODY solves). Numbers are 0-100, one decimal
under 10 and integer above, except tab:solved_factuality which keeps one decimal.
--write-tex replaces the rows between the midrule and bottomrule of each float.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import models
from ..paths import PUZZLE_LABELS_CSV, SCORES_DIR, TABLES_DIR
from ..score.common import elo_rating

COND_CSV = TABLES_DIR / "conditioned_by_first_move.csv"
RATINGS_CSV = TABLES_DIR / "puzzle_ratings.csv"
BOOT = 2000
MIN_SOLVED = 1
MIN_SOLVED_BODY = 50
N_PUZZLES = 2917

FACT_CONS = ["rules", "material", "pv_legal", "material_cons", "chosen_legal", "cands_legal", "move_eval"]
POSITIONAL = ["mobility", "king_safety", "pawns", "eval", "concepts", "similarity_max"]
TABLES = {"tab:conditioned_results": FACT_CONS, "tab:conditioned_positional": POSITIONAL}
BODY = {"tab:solved_factuality": FACT_CONS, "tab:solved_positional": POSITIONAL}


def ml_rating(correct, ratings):
    """The R at which expected solves equal actual solves; NaN if nothing or
    everything was solved."""
    k = float(np.sum(correct))
    if k <= 0 or k >= len(correct):
        return np.nan
    return brentq(lambda r: np.sum(1.0 / (1.0 + 10.0 ** ((ratings - r) / 400.0))) - k, -5000.0, 8000.0)


def ml_rating_ci(correct, ratings, rng):
    idx = rng.integers(0, len(correct), size=(BOOT, len(correct)))
    boots = np.array([ml_rating(correct[i], ratings[i]) for i in idx])
    boots = np.where(np.isnan(boots), -np.inf, boots)   # a resample with no solve
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def _by_model(rows_csv):
    df = pd.read_csv(rows_csv)
    df = df[df["llm_file"].apply(models.is_baseline_file)]
    df["model"] = df["llm_file"].map(models.canonical_id)
    return df


def compute():
    labels = pd.read_csv(PUZZLE_LABELS_CSV)
    labels["row_index"] = range(len(labels))
    fact = _by_model(SCORES_DIR / "factuality_rows.csv")
    cons = _by_model(SCORES_DIR / "consistency_rows.csv")
    sim = _by_model(SCORES_DIR / "explanation_rows.csv")
    corr = _by_model(SCORES_DIR / "correctness_rows.csv")
    strat = _by_model(SCORES_DIR / "strategic_rows.csv")
    recall = pd.read_csv(SCORES_DIR / "decomposition_recall_rows.csv")
    recall = recall.merge(labels[["PuzzleId", "row_index"]], on="PuzzleId", how="left")

    for df in (fact, cons, sim, corr, strat):
        missing = set(models.MODEL_ORDER) - set(df["model"])
        assert not missing, f"models without scored rows: {missing}"

    cond_rows, rating_rows = [], []
    for mid in models.MODEL_ORDER:
        f = fact[fact.model == mid].set_index("row_index").sort_index()
        c = cons[cons.model == mid].set_index("row_index").sort_index()
        r = sim[sim.model == mid].set_index("row_index").sort_index()
        k = corr[corr.model == mid].set_index("row_index").sort_index()
        st = strat[strat.model == mid].set_index("row_index").sort_index()
        cov = recall[recall.model == mid].set_index("row_index")["recall"].reindex(f.index).fillna(0.0)
        assert len(f) == len(c) == len(r) == len(k) == len(st) == len(labels) == N_PUZZLES, (mid, len(f), len(c), len(r), len(k), len(st))

        correct = k["first_move_accuracy"].astype(int)
        assert set(correct.unique()) <= {0, 1}
        is_c = correct == 1
        sim_nm = r["similarity"].astype(float).fillna(0.0)
        # No fillna: NaN means the FM did not annotate the puzzle.
        sim_fm = r["similarity_fm"].astype(float)
        expl = (sim_nm + cov) / 2.0

        def col(series):
            return series.astype(float).fillna(0.0)

        submetrics = [
            ("rules", col(f["f1_score"])), ("material", col(f["material_accuracy"])),
            ("pv_legal", col(f["pv_legality_accuracy"])),
            ("material_cons", col(c["material_consistency"])),
            ("chosen_legal", col(c["best_move_in_pred_legal_percentage"])),
            ("cands_legal", col(c["candidate_in_pred_legal_percentage"])),
            ("move_eval", col(c["best_move_eval_label_matches_own_initial_eval"])),
            ("mobility", col(st["mobility_accuracy"])), ("king_safety", col(st["king_safety_accuracy"])),
            ("pawns", col(st["pawn_structure_accuracy"])), ("eval", col(st["initial_eval_accuracy"])),
            ("concepts", cov), ("similarity", sim_nm), ("similarity_fm", sim_fm),
            ("similarity_max", np.fmax(sim_nm, sim_fm)),
        ]
        cond_rows.append({
            "model": mid,
            "n_correct": int(is_c.sum()),
            "factuality_C": f.loc[is_c, "factuality_score"].mean(),
            "factuality_W": f.loc[~is_c, "factuality_score"].mean(),
            "consistency_C": c.loc[is_c, "consistency_score"].mean(),
            "consistency_W": c.loc[~is_c, "consistency_score"].mean(),
            "explanation_C": expl[is_c].mean(),
            "explanation_W": expl[~is_c].mean(),
            "pv_given_correct": k.loc[is_c, "best_move_pv_correct"].mean(),
            **{f"{name}_{tag}": series[mask].mean()
               for name, series in submetrics for tag, mask in (("C", is_c), ("W", ~is_c))},
        })

        c_arr, r_arr = correct.to_numpy(dtype=float), labels["Rating"].to_numpy(dtype=float)
        lo, hi = ml_rating_ci(c_arr, r_arr, np.random.default_rng(0))
        rating_rows.append({
            "model": mid,
            "first_move_accuracy": correct.mean(),
            "n_solved": int(c_arr.sum()),
            "puzzle_rating": ml_rating(c_arr, r_arr),
            "ci_lower": lo, "ci_upper": hi,
            "puzzle_rating_sequential": elo_rating(correct.tolist(), labels["Rating"].tolist()),
        })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cond = pd.DataFrame(cond_rows)
    cond.to_csv(COND_CSV, index=False, float_format="%.4f")
    ratings = pd.DataFrame(rating_rows)
    ratings.to_csv(RATINGS_CSV, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(cond.round(3).to_string(index=False))
    print()
    print(ratings.round(1).to_string(index=False))
    print(f"\nWrote {COND_CSV}\nWrote {RATINGS_CSV}")
    return cond


def fmt(v):
    """One decimal under 10, integer at or above."""
    d = Decimal(str(v)) * 100
    return str(d.quantize(Decimal("0.1") if d < 10 else Decimal("1"), rounding=ROUND_HALF_UP))


def fmt1(v):
    """One decimal."""
    return str((Decimal(str(v)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def latex_rows(cond):
    d = cond.set_index("model")
    tables = {}
    for label, metrics in TABLES.items():
        out = []
        for m in models.LATEX_ORDER:
            r = d.loc[m]
            if r["n_correct"] < MIN_SOLVED:
                continue
            cells = [str(int(r["n_correct"]))]
            for met in metrics:
                cells += [fmt(r[f"{met}_C"]), fmt(r[f"{met}_W"])]
            out.append(models.latex(m) + " & " + " & ".join(cells) + r" \\")
        tables[label] = out
    for label, metrics in BODY.items():
        out = []
        for m in models.LATEX_ORDER:
            r = d.loc[m]
            if r["n_correct"] < MIN_SOLVED_BODY:
                continue
            if label == "tab:solved_factuality":
                cells = [str(int(r["n_correct"]))] + [fmt1(r[f"{met}_C"]) for met in metrics]
            else:
                cells = ([fmt(r[f"{met}_C"]) for met in metrics]
                         + [fmt(r["pv_given_correct"]), fmt(r["n_correct"] / N_PUZZLES)])
            out.append(models.latex(m) + " & " + " & ".join(cells) + r" \\")
        tables[label] = out
    return tables


def write_tex(tex_path, tables):
    tex = Path(tex_path)
    s = tex.read_text(encoding="utf-8", newline="")
    nl = "\r\n" if "\r\n" in s else "\n"
    for label, lines in tables.items():
        lab = s.index(rf"\label{{{label}}}")
        start = s.rindex(r"\midrule", 0, lab) + len(r"\midrule") + len(nl)
        end = s.rindex(r"\bottomrule", 0, lab)
        s = s[:start] + nl.join(lines) + nl + s[end:]
    tex.write_text(s, encoding="utf-8", newline="")
    print(f"wrote {tex}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-tex", metavar="TEX", help="replace the rows of the four labelled floats in this file")
    args = ap.parse_args()
    compute()
    # Format from the written CSV so the rows match it exactly.
    tables = latex_rows(pd.read_csv(COND_CSV))
    print()
    for label, lines in tables.items():
        print(f"% {label}")
        print("\n".join(lines))
    if args.write_tex:
        write_tex(args.write_tex, tables)


if __name__ == "__main__":
    main()
