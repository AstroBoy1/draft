"""Reference explanations for the similarity metric.

Each explanation is scored against each master separately: `similarity_nm`
(the national master, who annotated all 2,917 puzzles; also exposed as
`similarity`) and `similarity_fm` (the FIDE master, a subset). The FM column is
scored only on the puzzles the FM annotated, so its n differs.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import COMMENTARY_FM_CSV, COMMENTARY_NM_CSV, PUZZLE_LABELS_CSV

# False: national master only.
USE_FM_REFERENCE = True
SOURCES = ("nm", "fm")
CROSS_ENCODER_NAME = "cross-encoder/stsb-roberta-base"


def _clean(text):
    return text.strip() if isinstance(text, str) and text.strip() else None


@lru_cache(maxsize=None)
def _by_id():
    """{PuzzleId: [(source, text), ...]} over both masters."""
    nm = pd.read_csv(COMMENTARY_NM_CSV)
    out = {p: [] for p in nm.PuzzleId}
    for p, d in zip(nm.PuzzleId, nm.Description):
        if (t := _clean(d)):
            out[p].append(("nm", t))
    if USE_FM_REFERENCE and COMMENTARY_FM_CSV.is_file():
        fm = pd.read_csv(COMMENTARY_FM_CSV)
        for p, d in zip(fm.PuzzleId, fm.Description):
            if p in out and (t := _clean(d)):
                out[p].append(("fm", t))
    return out


def load_labeled_reference_lists(puzzle_ids=None):
    """Per-puzzle list of (source, text), aligned to `puzzle_ids` (default:
    puzzle_labels.csv row order). Empty where neither master wrote anything."""
    if puzzle_ids is None:
        puzzle_ids = pd.read_csv(PUZZLE_LABELS_CSV, usecols=["PuzzleId"]).PuzzleId
    by_id = _by_id()
    return [list(by_id.get(p, [])) for p in puzzle_ids]


def load_reference_lists(puzzle_ids=None):
    """Texts only."""
    return [[t for _, t in refs] for refs in load_labeled_reference_lists(puzzle_ids)]


def coverage(puzzle_ids=None):
    """(n_puzzles, n_with_two_or_more_references)."""
    refs = load_reference_lists(puzzle_ids)
    return len(refs), sum(1 for r in refs if len(r) > 1)


_cross_encoder = None


def cross_encoder():
    """The cross-encoder, loaded once."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("sentence-transformers is required for explanation similarity: "
                              "pip install sentence-transformers") from e
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=device)
    return _cross_encoder


def similarity_by_source(model, explanations, labeled_reference_lists,
                         batch_size=16, chunk_size=256):
    """{source: [score or NaN per row]}; NaN where the model wrote nothing or
    that master did not annotate the puzzle. One pass over the cross-encoder."""
    n = len(explanations)
    out = {src: [np.nan] * n for src in SOURCES}
    pairs, owners = [], []
    for i, expl in enumerate(explanations):
        if i >= len(labeled_reference_lists):
            break
        text = _clean(expl)
        if text is None:
            continue
        for src, ref in labeled_reference_lists[i]:
            pairs.append((text, ref))
            owners.append((i, src))
    if not pairs:
        return out

    raw = []
    for start in range(0, len(pairs), chunk_size):
        raw.extend(model.predict(pairs[start:start + chunk_size], batch_size=batch_size))
    for (i, src), s in zip(owners, raw):
        out.setdefault(src, [np.nan] * n)[i] = float(s)
    return out


def annotated_masks(n=None):
    """{source: [bool per puzzle]} in row_index order."""
    refs = load_labeled_reference_lists()
    masks = {src: [any(s == src for s, _ in r) for r in refs] for src in SOURCES}
    if n is not None:
        masks = {src: (m + [False] * n)[:n] for src, m in masks.items()}
    return masks


def apply_failure_rule(df, index_col="row_index"):
    """Apply the failure rule to each similarity column, in place: NaN becomes 0
    on puzzles that master annotated and stays NaN elsewhere. `similarity` is a
    copy of the NM column. Idempotent."""
    masks = annotated_masks()
    for src in SOURCES:
        col = f"similarity_{src}"
        if col not in df.columns:
            continue
        covered = df[index_col].map(
            lambda i: masks[src][int(i)] if 0 <= int(i) < len(masks[src]) else False
        ).astype(bool)
        df[col] = df[col].where(~covered | df[col].notna(), 0.0)
        df.loc[~covered, col] = np.nan
    if "similarity_nm" in df.columns:
        df["similarity"] = df["similarity_nm"]
    return df


if __name__ == "__main__":
    total, both = coverage()
    print(f"{COMMENTARY_NM_CSV}: national master reference")
    print(f"{COMMENTARY_FM_CSV}: FIDE master reference "
          f"({'enabled' if USE_FM_REFERENCE else 'DISABLED'})")
    print(f"{total} puzzles, {both} with two references, {total - both} with one")
