"""Normalise move tokens to UCI before scoring.

Legal-move metrics compare `all_legal_moves_uci` to ground truth by exact string
match, but some models answer in SAN or long algebraic (Mistral-7B and
Llama-3.1-8B mostly do). Scored raw they get a legal-move F1 near 0 for
notation alone.

Tokens that cannot be resolved to a legal move are kept in the prediction set
as sentinels, so they still count as false positives. Dropping them would turn
precision into "fraction convertible" (for Mistral-7B, 0.137 instead of 0.076).

Conversion is a lookup against the position's legal moves rather than a
`parse_san` call per token; a legal SAN token converts either way, and the
lookup is much cheaper over 2,917 puzzles x 14 models.
"""

import re
import chess

UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?\Z")
_ZERO_CASTLE_RE = re.compile(r"0-0(-0)?[+#]?\Z")
_PROMO_NO_EQ_RE = re.compile(r"([a-h]?x?[a-h][1-8])([QRBN])[+#]?\Z")
_PIECE_PREFIX_RE = re.compile(r"([KQRBNkqrbn])([a-h][1-8]-?[a-h][1-8][qrbnQRBN]?[+#]?)\Z")
_BOARDS = {}


def _board_for(fen):
    b = _BOARDS.get(fen)
    if b is None:
        if len(_BOARDS) > 4096:
            _BOARDS.clear()
        b = _BOARDS[fen] = chess.Board(fen)
    return b

# Prefix for a token that is not a legal move; never equal to a real UCI move.
UNCONVERTIBLE = "__unconvertible__:"

_CACHE = {}


def _alias_map(fen):
    """{spelling -> uci} for every legal move: UCI, SAN, long algebraic, each
    with and without a trailing check/mate marker."""
    if fen in _CACHE:
        return _CACHE[fen]
    board = chess.Board(fen)
    out = {}
    for mv in board.legal_moves:
        uci = mv.uci()
        spellings = {uci, board.san(mv), board.lan(mv)}
        for s in list(spellings):
            spellings.add(s.rstrip("+#"))
        for s in spellings:
            out.setdefault(s, uci)
            out.setdefault(s.lower(), uci)
    if len(_CACHE) > 4096:
        _CACHE.clear()
    _CACHE[fen] = out
    return out


def normalise_move(token, fen):
    """One token -> UCI, or None if it is not a legal move in this position."""
    if not isinstance(token, str):
        return None
    t = token.strip()
    if not t:
        return None
    aliases = _alias_map(fen)

    # Already UCI (case-folded: models write e7e8Q).
    if UCI_RE.match(t) and t.lower() in aliases:
        return t.lower()

    cands = [t, t.rstrip("+#"), t.replace("-", ""), t.replace("-", "").rstrip("+#")]

    # Piece letter glued onto a UCI move (re2e1, Kf8g8, nb1-c3), accepted only
    # when that piece stands on the from-square.
    m = _PIECE_PREFIX_RE.match(t)
    if m:
        piece, rest = m.group(1).lower(), m.group(2).replace("-", "").rstrip("+#").lower()
        board = _board_for(fen)
        sq = chess.parse_square(rest[:2])
        pc = board.piece_at(sq)
        if pc is not None and pc.symbol().lower() == piece:
            cands.append(rest)

    # Castling written with zeros.
    if _ZERO_CASTLE_RE.match(t):
        cands.append(t.replace("0", "O"))

    # Promotion without the '=': a8Q.
    m = _PROMO_NO_EQ_RE.match(t)
    if m:
        cands.append(f"{m.group(1)}={m.group(2)}")

    for cand in cands:
        if cand in aliases:
            return aliases[cand]
        if cand.lower() in aliases:
            return aliases[cand.lower()]

    # Capture SAN without the 'x' (Ra3 for Rxa3): not in the alias map, but
    # parse_san accepts it. Tried last; it only ever returns legal moves.
    board = _board_for(fen)
    for cand in cands:
        try:
            return board.parse_san(cand).uci()
        except ValueError:
            continue
    return None


def normalise_moves(tokens, fen):
    """Tokens -> UCI, unconvertible ones kept as sentinels. Never drops a token."""
    return [normalise_or_sentinel(tok, fen)
            for tok in (tokens or []) if isinstance(tok, str)]


def normalise_or_sentinel(token, fen):
    """UCI if the token is a legal move, else a sentinel derived from the token,
    so the consistency checks (best_move in the model's own list) still match
    an illegal token against itself."""
    u = normalise_move(token, fen)
    return u if u is not None else f"{UNCONVERTIBLE}{token.strip()}"


def uci_native_rate(tokens):
    """Fraction of tokens already written as UCI."""
    toks = [t for t in (tokens or []) if isinstance(t, str)]
    if not toks:
        return float("nan")
    return sum(1 for t in toks if UCI_RE.match(t.strip())) / len(toks)


def push_move(board, token):
    """Play one token onto `board`, any notation. True if it was legal."""
    uci = normalise_move(token, board.fen())
    if uci is None:
        return False
    board.push_uci(uci)
    return True


def normalise_sequence(moves, fen):
    """A move sequence -> UCI, each token resolved against the board as it
    stands. Stops at the first illegal move and keeps the rest verbatim."""
    board = chess.Board(fen)
    out = []
    for i, tok in enumerate(moves or []):
        if not isinstance(tok, str):
            continue
        uci = normalise_move(tok, board.fen())
        if uci is None:
            out.extend(m for m in moves[i:] if isinstance(m, str))
            break
        out.append(uci)
        board.push_uci(uci)
    return out
