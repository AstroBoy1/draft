"""Run Stockfish 11 `eval` on every puzzle position and keep the Pawns and King
safety rows (white/black/total, middlegame/endgame) in data/stockfish_terms.csv.
Needs a Stockfish 11 binary (--stockfish or $STOCKFISH); later versions use
NNUE and do not print the term table.
"""
import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

from ..paths import PUZZLES_CSV, STOCKFISH_TERMS_CSV
from ..puzzles import starting_fen

TERMS = {"pawns": "pawn_structure", "king safety": "king_safety"}
COLUMNS = [f"{name}_{part}_{ph}" for name in TERMS.values()
           for part in ("white", "black", "total") for ph in ("mg", "eg")]


def parse_eval_rows(eval_text: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {c: None for c in COLUMNS}
    for raw_line in eval_text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        name = TERMS.get(parts[0].lower())
        if name is None:
            continue
        for part, cell in zip(("white", "black", "total"), parts[1:4]):
            values = cell.split()
            if len(values) >= 2:
                result[f"{name}_{part}_mg"], result[f"{name}_{part}_eg"] = values[0], values[1]
    return result


class Stockfish:
    def __init__(self, binary):
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.read_until("uciok")
        self.send("isready")
        self.read_until("readyok")

    def send(self, command):
        self.proc.stdin.write(f"{command}\n")
        self.proc.stdin.flush()

    def read_until(self, terminator):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("Stockfish process ended unexpectedly while reading output")
            lines.append(line.rstrip("\n"))
            if lines[-1] == terminator:
                return lines

    def eval_terms(self, fen):
        self.send(f"position fen {fen}")
        self.send("eval")
        self.send("isready")
        lines = self.read_until("readyok")
        return parse_eval_rows("\n".join(lines[:-1]))

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", type=Path, default=PUZZLES_CSV)
    ap.add_argument("--output", type=Path, default=STOCKFISH_TERMS_CSV)
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH", "stockfish11_eval"),
                    help="Stockfish 11 binary (default: $STOCKFISH or ./stockfish11_eval)")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print-results", action="store_true", help="print the first rows instead of only saving")
    args = ap.parse_args()

    if not Path(args.stockfish).is_file():
        raise FileNotFoundError(f"Missing Stockfish executable: {args.stockfish}")
    df = pd.read_csv(args.puzzles).iloc[args.start_index:args.end_index]
    if args.limit is not None:
        df = df.head(args.limit)
    fens = [starting_fen(f, m) for f, m in zip(df["FEN"], df["Moves"])]

    engine = Stockfish(args.stockfish)
    try:
        rows = []
        for i, fen in enumerate(fens):
            print(f"Processing index {i}: {fen}")
            rows.append(engine.eval_terms(fen))
    finally:
        engine.close()

    out = pd.concat([pd.DataFrame({"starting_position_fen": fens}), pd.DataFrame(rows)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    if args.print_results:
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
