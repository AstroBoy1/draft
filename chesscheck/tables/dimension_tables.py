"""One table per dimension over all puzzles, from bootstrap_conditioned_metrics.csv:

    table1_factuality, table2_logical_consistency, table3_strategic_evaluation,
    table4_explanation_quality, table5_reasoning

Each also gets a _ci variant with "mean [lo, hi]" cells. Written to
results/tables/dimension_tables/.
"""
import argparse
from pathlib import Path

import pandas as pd

from .. import models
from ..paths import TABLES_DIR

# (file stem, title, sub-metrics in display order)
DIMENSIONS = [
    ("table1_factuality", "1. Factuality",
     ["material_accuracy", "pv_legality_accuracy", "f1_score"]),
    ("table2_logical_consistency", "2. Logical Consistency",
     ["material_consistency", "best_move_in_pred_legal_percentage",
      "candidate_in_pred_legal_percentage", "best_move_eval_label_matches_own_initial_eval"]),
    ("table3_strategic_evaluation", "3. Strategic Evaluation",
     ["mobility_accuracy", "king_safety_accuracy", "pawn_structure_accuracy", "initial_eval_accuracy"]),
    ("table4_explanation_quality", "4. Explanation Quality",
     ["similarity", "concept_coverage"]),
    ("table5_reasoning", "5. Reasoning",
     ["first_move_accuracy", "best_move_pv_correct"]),
]

SHORT = {
    "material_accuracy": "Material", "pv_legality_accuracy": "PV Legal", "f1_score": "Legal Moves F1",
    "material_consistency": "Material", "best_move_in_pred_legal_percentage": "Chosen Legal",
    "candidate_in_pred_legal_percentage": "Cands Legal",
    "best_move_eval_label_matches_own_initial_eval": "Move Eval",
    "best_move_eval_label_matches_initial_eval": "Eval Accuracy",
    "mobility_accuracy": "Mobility", "king_safety_accuracy": "King Safety",
    "pawn_structure_accuracy": "Pawns", "initial_eval_accuracy": "Eval",
    "similarity": "Similarity", "concept_coverage": "Concept Coverage",
    "first_move_accuracy": "First Move", "best_move_pv_correct": "PV Correct",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-csv", default=str(TABLES_DIR / "bootstrap_conditioned_metrics.csv"))
    ap.add_argument("--out-dir", default=str(TABLES_DIR / "dimension_tables"))
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    d = pd.read_csv(args.metrics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [m for m in models.MODEL_ORDER if m in set(d.model)]
    missing = [m for m in models.MODEL_ORDER if m not in set(d.model)]

    for stem, title, cols in DIMENSIONS:
        sub = d[d.metric.isin(cols)]
        if sub.empty:
            print(f"\n{title}: SKIP (no rows)")
            continue
        pivots = {v: sub.pivot_table(index="model", columns="metric", values=v, aggfunc="first")
                  for v in ("mean", "ci_lower_95", "ci_upper_95")}
        cols = [c for c in cols if c in pivots["mean"].columns]
        mean_t, lo_t, hi_t = (pivots[v].reindex(present)[cols] for v in ("mean", "ci_lower_95", "ci_upper_95"))

        pretty = mean_t.round(args.decimals).rename(columns=SHORT)
        pretty.index = [models.display(m) for m in pretty.index]
        pretty.index.name = "Model"
        pretty.to_csv(out_dir / f"{stem}.csv")

        ci = pd.DataFrame(index=pretty.index)
        for c in cols:
            ci[SHORT.get(c, c)] = [
                f"{m:.{args.decimals}f} [{l:.{args.decimals}f}, {h:.{args.decimals}f}]" if pd.notna(m) else ""
                for m, l, h in zip(mean_t[c], lo_t[c], hi_t[c])]
        ci.to_csv(out_dir / f"{stem}_ci.csv")

        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(pretty.to_string())

    print(f"\nWrote {len(DIMENSIONS)} tables (+ _ci variants) to {out_dir}")
    if missing:
        print(f"\nNOT YET PRESENT in {args.metrics_csv}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
