"""Factuality: facts derivable from the FEN with python-chess.

    material_accuracy      piece counts match exactly
    pv_legality_accuracy   fraction of the model's lines that are legal throughout
    f1_score               legal-move F1 (macro precision and recall of all_legal_moves_uci)

The four positional labels are computed here as well (they come from the same
loaded response) but belong to the strategic dimension.
"""
import ast

import chess
import numpy as np
import pandas as pd

from ..scoring_rule import count_failures_as_wrong
from .common import (best_move_pv_matches_solution_rowwise, cli, conditioned_summary,
                     first_move_accuracy_rowwise, input_files, labels, load_predictions,
                     meta_columns, nanmean, normalize_move_sequence, write_outputs)

FACTUALITY_COLS = ["material_accuracy", "pv_legality_accuracy", "f1_score"]
STRATEGIC_COLS = ["initial_eval_accuracy", "mobility_accuracy", "king_safety_accuracy",
                  "pawn_structure_accuracy"]


def label_match_rowwise(pred, true):
    """Label agreement: either label contains the other."""
    scores = []
    for p, t in zip(pred, true):
        if p is None:
            scores.append(np.nan)
        elif not isinstance(p, str) or not isinstance(t, str):
            scores.append(0.0)
        else:
            scores.append(1.0 if (p in t or t in p) else 0.0)
    return scores


def material_accuracy_rowwise(pred, true):
    scores = []
    for p, t in zip(pred, true):
        if p is None:
            scores.append(np.nan)
            continue
        try:
            t_dict = ast.literal_eval(t) if isinstance(t, str) else t
        except (ValueError, SyntaxError):
            scores.append(0.0)
            continue
        match = True
        for piece in ("P", "B", "R", "Q", "N"):
            for color in ("white", "black"):
                try:
                    if p[color][piece] != t_dict[color][piece]:
                        match = False
                        break
                except (TypeError, KeyError):
                    match = False
                    break
            if not match:
                break
        scores.append(1.0 if match else 0.0)
    return scores


def macro_precision_rowwise(pred, true):
    scores = []
    for p, t in zip(pred, true):
        if p is None:
            scores.append(np.nan)
            continue
        p_set, t_set = set(p), set(t)
        tp, fp = len(p_set & t_set), len(p_set - t_set)
        scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return scores


def recall_rowwise(pred, true):
    scores = []
    for p, t in zip(pred, true):
        if p is None:
            scores.append(np.nan)
            continue
        p_set, t_set = set(p), set(t)
        tp, fn = len(p_set & t_set), len(t_set - p_set)
        scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return scores


def f1_rowwise(precision_scores, recall_scores):
    scores = []
    for p, r in zip(precision_scores, recall_scores):
        if pd.isna(p) or pd.isna(r):
            scores.append(np.nan)
            continue
        scores.append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
    return scores


def pv_legality_rowwise(pred_pvs, fen_list):
    scores = []
    for pvs, fen in zip(pred_pvs, fen_list):
        if pvs is None:
            scores.append(np.nan)
            continue
        if not isinstance(pvs, (list, tuple)):
            pvs = [pvs]
        if len(pvs) == 0:
            scores.append(0.0)
            continue
        legal = 0
        for pv in pvs:
            moves = normalize_move_sequence(pv)
            if not moves:
                continue
            board = chess.Board(fen)
            is_legal = 1
            for move in moves:
                try:
                    board.push_uci(move)
                except ValueError:
                    is_legal = 0
                    break
            legal += is_legal
        scores.append(legal / len(pvs))
    return scores


def summarize(path):
    p = load_predictions(path)
    lab = labels()
    precision = macro_precision_rowwise(p.legal_moves, lab["legal_moves"])
    recall = recall_rowwise(p.legal_moves, lab["legal_moves"])
    out = pd.DataFrame({
        **meta_columns(path),
        "initial_eval_accuracy": label_match_rowwise(p.initial_eval, lab["initial_eval"]),
        "mobility_accuracy": label_match_rowwise(p.mobility, lab["mobility"]),
        "king_safety_accuracy": label_match_rowwise(p.king_safety, lab["king_safety"]),
        "material_accuracy": material_accuracy_rowwise(p.material, lab["material_balance"]),
        "pawn_structure_accuracy": label_match_rowwise(p.pawn_structure, lab["pawn_structure"]),
        "pv_legality_accuracy": pv_legality_rowwise(p.pvs, lab["starting_position_fen"]),
        "macro_precision": precision,
        "recall": recall,
        "f1_score": f1_rowwise(precision, recall),
        "best_move_pv_matches_solution": best_move_pv_matches_solution_rowwise(p.best_move_pv, lab["Moves"], p.ok),
        "first_move_accuracy": first_move_accuracy_rowwise(p.best_move_pv, lab["Moves"]),
    })
    count_failures_as_wrong(out, [c for c in out.columns if c not in ("llm_file", "source", "row_index")])
    out["factuality_score"] = out[FACTUALITY_COLS].mean(axis=1, skipna=True)
    return out


def main():
    args = cli("factuality", __doc__).parse_args()
    rows = pd.concat([summarize(f) for f in input_files(args)], ignore_index=True)
    write_outputs(rows, conditioned_summary(rows, "factuality_score", FACTUALITY_COLS), args)


if __name__ == "__main__":
    main()
