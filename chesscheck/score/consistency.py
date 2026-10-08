"""Logical consistency: is the response coherent with itself?

    material_consistency                           piece counts and side-ahead are right
                                                   together or wrong together
    best_move_in_pred_legal_percentage             the chosen move is in the model's own legal list
    candidate_in_pred_legal_percentage             fraction of candidates that are
    best_move_eval_label_matches_own_initial_eval  the chosen move's label equals the model's
                                                   own position evaluation

best_move_eval_label_matches_initial_eval (against ground truth) is emitted per
row but is not part of consistency_score.
"""
import ast

import numpy as np
import pandas as pd

from ..scoring_rule import count_failures_as_wrong
from .common import (best_move_pv_matches_solution_rowwise, cli, conditioned_summary,
                     first_move_accuracy_rowwise, input_files, labels, load_predictions,
                     meta_columns, write_outputs)

CONSISTENCY_COLS = [
    "material_consistency",
    "best_move_in_pred_legal_percentage",
    "candidate_in_pred_legal_percentage",
    "best_move_eval_label_matches_own_initial_eval",
]
SUMMARY_DIMS = CONSISTENCY_COLS[:3] + ["best_move_eval_label_matches_initial_eval",
                                       "best_move_eval_label_matches_own_initial_eval"]


def normalize_three_class_label(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "material_diff" in value:
            return normalize_three_class_label(value.get("material_diff"))
        if "side_ahead" in value:
            return normalize_three_class_label(value.get("side_ahead"))
        return None
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if not label:
        return None
    if "white" in label:
        return "white"
    if "black" in label:
        return "black"
    if label in {"equal", "even", "balanced", "draw", "neutral"}:
        return "equal"
    return None


def material_consistency_rowwise(material_pred, material_true, diff_pred, diff_true):
    scores = []
    for mp, mt, mdp, mdt in zip(material_pred, material_true, diff_pred, diff_true):
        if mp is None or mdp is None:
            scores.append(np.nan)
            continue
        try:
            mt_dict = ast.literal_eval(mt) if isinstance(mt, str) else mt
            mdt_dict = ast.literal_eval(mdt) if isinstance(mdt, str) else mdt
        except (ValueError, SyntaxError):
            scores.append(0.0)
            continue
        material_match = True
        for piece in ("P", "B", "R", "Q", "N"):
            for color in ("white", "black"):
                try:
                    if mp[color][piece] != mt_dict[color][piece]:
                        material_match = False
                        break
                except (TypeError, KeyError):
                    material_match = False
                    break
            if not material_match:
                break
        true_diff = normalize_three_class_label(mdt_dict)
        pred_diff = normalize_three_class_label(mdp)
        if true_diff is None or pred_diff is None:
            scores.append(0.0)
            continue
        scores.append(1.0 if material_match == (pred_diff == true_diff) else 0.0)
    return scores


def best_move_in_pred_legal_rowwise(best_moves, legal_lists):
    scores = []
    for best_move, legal in zip(best_moves, legal_lists):
        if best_move is None or legal is None:
            scores.append(np.nan)
        elif not isinstance(best_move, str):
            scores.append(0.0)
        else:
            scores.append(1.0 if best_move in set(legal) else 0.0)
    return scores


def candidate_in_pred_legal_rowwise(candidate_lists, legal_lists):
    scores = []
    for cands, legal in zip(candidate_lists, legal_lists):
        if cands is None or legal is None:
            scores.append(np.nan)
        elif len(cands) == 0:
            scores.append(0.0)
        else:
            legal_set = set(legal)
            scores.append(sum(1 for m in cands if m in legal_set) / len(cands))
    return scores


def chosen_eval_matches_rowwise(best_moves, eval_maps, reference_evals, require_reference=False):
    """The chosen move's eval_label against a reference label per row."""
    scores = []
    for best_move, eval_map, ref in zip(best_moves, eval_maps, reference_evals):
        if best_move is None or eval_map is None or (require_reference and ref is None):
            scores.append(np.nan)
            continue
        chosen = eval_map.get(best_move)
        if chosen is None:
            scores.append(0.0)
        else:
            scores.append(1.0 if isinstance(chosen, str) and isinstance(ref, str) and chosen == ref else 0.0)
    return scores


def summarize(path):
    p = load_predictions(path)
    lab = labels()
    out = pd.DataFrame({
        **meta_columns(path),
        "material_consistency": material_consistency_rowwise(
            p.material, lab["material_balance"], p.material_diff, lab["material_balance"]),
        "best_move_in_pred_legal_percentage": best_move_in_pred_legal_rowwise(p.best_move, p.legal_moves),
        "candidate_in_pred_legal_percentage": candidate_in_pred_legal_rowwise(p.candidate_moves, p.legal_moves),
        "best_move_eval_label_matches_initial_eval": chosen_eval_matches_rowwise(
            p.best_move, p.candidate_eval_map, lab["initial_eval"]),
        "best_move_eval_label_matches_own_initial_eval": chosen_eval_matches_rowwise(
            p.best_move, p.candidate_eval_map, p.initial_eval, require_reference=True),
        "best_move_pv_matches_solution": best_move_pv_matches_solution_rowwise(p.best_move_pv, lab["Moves"], p.ok),
        "first_move_accuracy": first_move_accuracy_rowwise(p.best_move_pv, lab["Moves"]),
    })
    count_failures_as_wrong(out, [c for c in out.columns if c not in ("llm_file", "source", "row_index")])
    out["consistency_score"] = out[CONSISTENCY_COLS].mean(axis=1, skipna=True)
    return out


def main():
    args = cli("consistency", __doc__).parse_args()
    rows = pd.concat([summarize(f) for f in input_files(args)], ignore_index=True)
    write_outputs(rows, conditioned_summary(rows, "consistency_score", SUMMARY_DIMS), args)


if __name__ == "__main__":
    main()
