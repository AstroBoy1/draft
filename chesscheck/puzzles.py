"""Puzzle loading and prompt construction.

A Lichess puzzle row gives the position before the opponent's last move (FEN)
and the move list (Moves), whose first move is that opponent move. The model is
asked about the position after it.
"""
import csv
from dataclasses import dataclass
from typing import Iterator

import chess

from .paths import PUZZLES_CSV

SYSTEM_INSTRUCTION = "You are a chess master. You are given a chess board in FEN format."


@dataclass(frozen=True)
class PuzzleRequest:
    index: int      # 0-based row index in puzzles.csv
    key: str        # PuzzleId (the row key every raw output file carries)
    fen: str        # the position the model is asked about


def starting_fen(fen: str, moves: str) -> str | None:
    """FEN after the first move of `moves`; None if it is illegal."""
    moves_list = (moves or "").split()
    if not moves_list:
        return fen
    try:
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(moves_list[0]))
        return board.fen()
    except Exception:
        return None


def user_content(fen: str, prompt_text: str, provide_legal_moves: bool = False) -> str:
    """The user turn: the FEN sentence, optionally the true legal moves, then the prompt."""
    text = f'The FEN of the given chess board is "{fen}".'
    if provide_legal_moves:
        legal = " ".join(sorted(m.uci() for m in chess.Board(fen).legal_moves))
        text += f" The legal moves in this position are: {legal}."
    return text + " " + prompt_text


def iter_requests(start: int = 0, end: int | None = None,
                  path=PUZZLES_CSV) -> Iterator[PuzzleRequest]:
    """Puzzles with row index in [start, end). Indices count every CSV row."""
    with open(path, newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            if index < start:
                continue
            if end is not None and index >= end:
                break
            fen = (row.get("FEN") or "").strip()
            if not fen:
                continue
            fen = starting_fen(fen, row.get("Moves") or "")
            if fen is None:
                print(f"Row {index}: could not apply the first move, skipping")
                continue
            key = (row.get("PuzzleId") or "").strip() or f"request-{index}"
            yield PuzzleRequest(index, key, fen)
