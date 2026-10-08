"""Build data/puzzle_labels.csv.

Per puzzle, from the starting position: legal_moves (UCI), initial_eval (the
side to move is better), mobility (which side has more legal moves),
material_balance (piece counts and side ahead, 1/3/3/5/9), and king_safety and
pawn_structure from the Stockfish 11 terms in data/stockfish_terms.csv. Each
Stockfish term is a (middlegame, endgame) pair blended by game phase; the label
is the white-minus-black difference at +-EQUAL_THRESHOLD pawns.
"""
import argparse

import chess
import pandas as pd

from ..paths import PUZZLE_LABELS_CSV, PUZZLES_CSV, STOCKFISH_TERMS_CSV
from ..puzzles import starting_fen

# Equal band in pawns: under Lichess's cp-to-win% fit a 20 cp edge is ~51.8%.
EQUAL_THRESHOLD = 0.2

# Stockfish 11 phase interpolation (src/material.cpp): non-pawn material in MG
# piece values, clamped to [EndgameLimit, MidgameLimit].
_MG_PIECE_VALUE = {chess.KNIGHT: 781, chess.BISHOP: 825, chess.ROOK: 1276, chess.QUEEN: 2538}
_MIDGAME_LIMIT = 15258
_ENDGAME_LIMIT = 3915


def game_phase(fen):
    """Phase factor in [0, 1]: 1 = middlegame, 0 = endgame."""
    board = chess.Board(fen)
    npm = sum(value * (len(board.pieces(piece, chess.WHITE)) + len(board.pieces(piece, chess.BLACK)))
              for piece, value in _MG_PIECE_VALUE.items())
    npm = max(_ENDGAME_LIMIT, min(_MIDGAME_LIMIT, npm))
    return (npm - _ENDGAME_LIMIT) / (_MIDGAME_LIMIT - _ENDGAME_LIMIT)


def legal_moves(fen):
    return [m.uci() for m in chess.Board(fen).legal_moves]


def initial_eval(fen):
    return "white_better" if chess.Board(fen).turn == chess.WHITE else "black_better"


def mobility(fen):
    board = chess.Board(fen)
    counts = {chess.WHITE: 0, chess.BLACK: 0}
    counts[board.turn] += board.legal_moves.count()
    board.turn = not board.turn
    counts[board.turn] += board.legal_moves.count()
    if counts[chess.WHITE] > counts[chess.BLACK]:
        return "white"
    if counts[chess.BLACK] > counts[chess.WHITE]:
        return "black"
    return "equal"


def material_balance(fen):
    placement = fen.split()[0]
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
    counts = {"white": {p: placement.count(p) for p in values},
              "black": {p: placement.count(p.lower()) for p in values}}
    white = sum(values[p] * n for p, n in counts["white"].items())
    black = sum(values[p] * n for p, n in counts["black"].items())
    counts["material_diff"] = "white" if white > black else "black" if black > white else "equal"
    return counts


def stockfish_term_label(fish_df, term):
    """3-way label for one term from its white/black MG/EG columns."""
    cols = {f"{term}_{side}_{ph}": pd.to_numeric(fish_df[f"{term}_{side}_{ph}"], errors="coerce")
            for side in ("white", "black") for ph in ("mg", "eg")}
    phase = fish_df["starting_position_fen"].apply(game_phase)
    white = phase * cols[f"{term}_white_mg"] + (1 - phase) * cols[f"{term}_white_eg"]
    black = phase * cols[f"{term}_black_mg"] + (1 - phase) * cols[f"{term}_black_eg"]
    diff = white - black
    label = pd.Series("equal", index=fish_df.index, dtype="object")
    label[diff > EQUAL_THRESHOLD] = "white_better"
    label[diff < -EQUAL_THRESHOLD] = "black_better"
    label[white.isna() | black.isna()] = pd.NA
    return label


def add_stockfish_labels(df, fish_csv=STOCKFISH_TERMS_CSV):
    fish = pd.read_csv(fish_csv)
    for term, column in (("king_safety", "king_safety"), ("pawn_structure", "pawn_structure")):
        needed = ["starting_position_fen"] + [f"{term}_{s}_{p}" for s in ("white", "black") for p in ("mg", "eg")]
        missing = [c for c in needed if c not in fish.columns]
        if missing:
            raise KeyError(f"Missing columns in {fish_csv}: {missing}")
        sub = fish[needed].copy()
        sub[column] = stockfish_term_label(sub, term)
        df = df.merge(sub[["starting_position_fen", column]], on="starting_position_fen", how="left")
    return df


def build(puzzles_csv=PUZZLES_CSV, fish_csv=STOCKFISH_TERMS_CSV) -> pd.DataFrame:
    df = pd.read_csv(puzzles_csv)
    df["starting_position_fen"] = [starting_fen(f, m) for f, m in zip(df["FEN"], df["Moves"])]
    df["legal_moves"] = df["starting_position_fen"].apply(legal_moves)
    df["initial_eval"] = df["starting_position_fen"].apply(initial_eval)
    df["mobility"] = df["starting_position_fen"].apply(mobility)
    df["material_balance"] = df["starting_position_fen"].apply(material_balance)
    return add_stockfish_labels(df, fish_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default=str(PUZZLES_CSV))
    ap.add_argument("--stockfish-terms", default=str(STOCKFISH_TERMS_CSV))
    ap.add_argument("--output", default=str(PUZZLE_LABELS_CSV))
    args = ap.parse_args()
    df = build(args.puzzles, args.stockfish_terms)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
