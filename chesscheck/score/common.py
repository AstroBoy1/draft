"""Shared by the dimension scorers: ground truth, the parsed-response loader,
the two solution-correctness rules, and the conditioned summary.
"""
import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import models, move_notation
from ..parsed import parsed_files
from ..paths import PARSED_DIR, PUZZLE_LABELS_CSV, SCORES_DIR

META_COLUMNS = ("llm_file", "source", "row_index")


def _str_to_list(s):
    s = s.strip("[]")
    if not s:
        return []
    return [m.strip().strip("'").strip('"') for m in s.split(",")]


@lru_cache(maxsize=None)
def labels() -> pd.DataFrame:
    """puzzle_labels.csv with `legal_moves` parsed into a list."""
    df = pd.read_csv(PUZZLE_LABELS_CSV)
    df["legal_moves"] = df["legal_moves"].apply(_str_to_list)
    return df


def normalize_move_sequence(moves):
    if moves is None:
        return []
    if isinstance(moves, str):
        return [m for m in moves.strip().split() if m]
    if isinstance(moves, (list, tuple)):
        return [m for m in moves if isinstance(m, str) and m]
    return []


@dataclass
class Predictions:
    """One list per prompt field, aligned with `labels()`; None where the
    response failed (`ok[i]` False) or the field was absent."""
    best_move: list = field(default_factory=list)
    candidate_eval_map: list = field(default_factory=list)   # {candidate_move: eval_label}
    legal_moves: list = field(default_factory=list)
    initial_eval: list = field(default_factory=list)
    mobility: list = field(default_factory=list)
    king_safety: list = field(default_factory=list)
    material: list = field(default_factory=list)
    pawn_structure: list = field(default_factory=list)
    candidate_moves: list = field(default_factory=list)
    material_diff: list = field(default_factory=list)
    pvs: list = field(default_factory=list)                  # every candidate's pv_uci
    best_move_pv: list = field(default_factory=list)         # the chosen candidate's pv_uci
    explanation: list = field(default_factory=list)
    ok: list = field(default_factory=list)
    uci_native_rate: float = float("nan")

    FIELDS = ("best_move", "candidate_eval_map", "legal_moves", "initial_eval", "mobility",
              "king_safety", "material", "pawn_structure", "candidate_moves", "material_diff",
              "pvs", "best_move_pv", "explanation")

    def _append_failed(self):
        for f in self.FIELDS:
            getattr(self, f).append(None)
        self.ok.append(False)


def load_predictions(path) -> Predictions:
    """Read one parsed JSONL and normalise every move token to UCI (see
    move_notation)."""
    p = Predictions()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                p._append_failed()
                continue
            best_move = row.get("best_move")
            legal = row.get("all_legal_moves_uci")
            if legal is None:
                legal = row.get("ten_legal_moves_uci")
            pos = row.get("positional_evaluation_no_search") or row.get("positional_evaluation") or {}

            def label(name, fallback=None):
                v = pos.get(name)
                return row.get(fallback or name) if v is None else v

            pvs = row.get("principal_variation_analysis")
            # Tolerate a PV entry without pv_uci, or a non-dict entry.
            if not isinstance(pvs, list):
                pvs = [] if pvs is None else [pvs]
            pvs_list = [it.get("pv_uci") for it in pvs if isinstance(it, dict)] if pvs else None
            best_move_pv = None
            if pvs and isinstance(best_move, str):
                for it in pvs:
                    if isinstance(it, dict) and it.get("candidate_move") == best_move:
                        best_move_pv = it.get("pv_uci")
                        break
            eval_map = {it.get("candidate_move"): it.get("eval_label")
                        for it in pvs if isinstance(it, dict) and it.get("candidate_move") is not None} if pvs else {}
            try:
                material_diff = row.get("material_count").get("material_diff").get("side_ahead")
            except AttributeError:
                material_diff = None

            p.best_move.append(best_move)
            p.candidate_eval_map.append(eval_map)
            p.legal_moves.append(legal)
            p.initial_eval.append(label("initial_eval_label"))
            p.mobility.append(label("mobility"))
            p.king_safety.append(label("king_safety"))
            p.material.append(label("material_count"))
            p.pawn_structure.append(label("pawn_structure"))
            p.candidate_moves.append(row.get("candidate_moves_uci"))
            p.material_diff.append(material_diff)
            p.pvs.append(pvs_list)
            p.best_move_pv.append(best_move_pv)
            p.explanation.append(row.get("final_explanation_summary"))
            p.ok.append(True)

    fens = list(labels()["starting_position_fen"])
    native = []
    for i in range(min(len(fens), len(p.best_move))):
        fen = fens[i]
        if p.legal_moves[i] is not None:
            native.append(move_notation.uci_native_rate(p.legal_moves[i]))
            p.legal_moves[i] = move_notation.normalise_moves(p.legal_moves[i], fen)
        if p.candidate_moves[i] is not None:
            p.candidate_moves[i] = move_notation.normalise_moves(p.candidate_moves[i], fen)
        if isinstance(p.best_move[i], str):
            p.best_move[i] = move_notation.normalise_or_sentinel(p.best_move[i], fen)
        if isinstance(p.candidate_eval_map[i], dict):
            # Keys are looked up with the normalised best_move.
            p.candidate_eval_map[i] = {
                (move_notation.normalise_or_sentinel(k, fen) if isinstance(k, str) else k): v
                for k, v in p.candidate_eval_map[i].items()
            }
        if p.best_move_pv[i]:
            p.best_move_pv[i] = move_notation.normalise_sequence(p.best_move_pv[i], fen)
        pvs = p.pvs[i]
        if isinstance(pvs, list) and pvs and isinstance(pvs[0], list):
            p.pvs[i] = [move_notation.normalise_sequence(pv, fen) for pv in pvs]
    p.uci_native_rate = (sum(native) / len(native)) if native else float("nan")
    return p


def first_move_accuracy_rowwise(best_move_pv_list, puzzle_moves_list):
    """1 if the first move of best_move_pv is the solution move, 0 otherwise,
    NaN if there is no line."""
    scores = []
    for pv, solution in zip(best_move_pv_list, puzzle_moves_list):
        pv_moves = normalize_move_sequence(pv)
        solution_moves = normalize_move_sequence(solution)
        if not pv_moves or len(solution_moves) < 2:
            scores.append(np.nan)
            continue
        scores.append(1 if pv_moves[0].lower() == solution_moves[1].lower() else 0)
    return scores


def best_move_pv_matches_solution_rowwise(best_move_pv_list, puzzle_moves_list, ok):
    """1 if the PV equals the solution from move 2 over the shorter length;
    NaN for a failed response."""
    scores = []
    for i in range(len(puzzle_moves_list)):
        if i >= len(ok) or not ok[i]:
            scores.append(np.nan)
            continue
        pv = best_move_pv_list[i] if i < len(best_move_pv_list) else None
        pv_moves = normalize_move_sequence(pv)
        solution_moves = normalize_move_sequence(puzzle_moves_list[i])
        if not pv_moves:
            scores.append(0)
            continue
        solution_from_second = solution_moves[1:] if len(solution_moves) > 1 else []
        n = min(len(pv_moves), len(solution_from_second))
        scores.append(1 if pv_moves[:n] == solution_from_second[:n] else 0)
    return scores


def elo_rating(correct_list, puzzle_ratings, initial_rating=1500, k=10):
    """Sequential Elo rating, one update per puzzle."""
    rating = initial_rating
    for correct, puzzle_rating in zip(correct_list, puzzle_ratings):
        expected = 1 / (1 + 10 ** ((puzzle_rating - rating) / 400))
        rating += k * (correct - expected)
    return rating


def nanmean(row_scores):
    return float(np.nanmean(row_scores)) if row_scores and not np.all(np.isnan(row_scores)) else 0.0


def meta_columns(path) -> dict:
    """llm_file / source / row_index for a rows DataFrame."""
    name = Path(path).name
    return {"llm_file": name, "source": models.source(models.canonical_id(name)),
            "row_index": range(len(labels()))}


def conditioned_summary(rows_df, score_col, dims):
    """Per file: the dimension score on puzzles whose first move was correct
    and wrong, its overall mean, and each sub-metric's mean."""
    key = ["source", "llm_file"]
    grouped = (rows_df[rows_df["first_move_accuracy"].notna()]
               .groupby(key + ["first_move_accuracy"], dropna=False)[score_col].mean().reset_index())
    overall = rows_df.groupby(key, dropna=False)[score_col].mean().reset_index(name=f"avg_{score_col}_overall")
    summary = (grouped.pivot_table(index=key, columns="first_move_accuracy", values=score_col, aggfunc="first")
               .reset_index()
               .rename(columns={1: f"{score_col}_first_move_correct", 0: f"{score_col}_first_move_incorrect"}))
    for col in (f"{score_col}_first_move_correct", f"{score_col}_first_move_incorrect"):
        if col not in summary.columns:      # a model with only one class
            summary[col] = np.nan
    dim_means = (rows_df.groupby(key, dropna=False)[list(dims)].mean().reset_index()
                 .rename(columns={c: f"avg_{c}" for c in dims}))
    summary = summary.merge(dim_means, on=key, how="left").merge(overall, on=key, how="left")
    return summary[key + [f"{score_col}_first_move_correct", f"{score_col}_first_move_incorrect",
                          f"avg_{score_col}_overall"] + [f"avg_{c}" for c in dims]]


def cli(dimension: str, description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--files", nargs="+", help="parsed JSONL files to score")
    src.add_argument("--parsed-dir", default=None, help=f"score every .jsonl here (default: {PARSED_DIR})")
    ap.add_argument("--output-rows", default=str(SCORES_DIR / f"{dimension}_rows.csv"))
    ap.add_argument("--output-summary", default=str(SCORES_DIR / f"{dimension}_summary.csv"))
    return ap


def input_files(args) -> list[Path]:
    files = [Path(f) for f in args.files] if args.files else parsed_files(args.parsed_dir)
    if not files:
        raise SystemExit(f"No parsed files found in {args.parsed_dir or PARSED_DIR}")
    print(f"Processing {len(files)} files")
    return files


def write_outputs(rows_df, summary_df, args):
    for path in (args.output_rows, args.output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(args.output_rows, index=False)
    summary_df.to_csv(args.output_summary, index=False)
    print(f"Wrote {args.output_rows} ({len(rows_df)} rows)")
    print(f"Wrote {args.output_summary}")
    print(summary_df.to_string(index=False))
