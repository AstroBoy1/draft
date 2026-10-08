"""Locating the parsed files. A directory entry may be a symlink, or on a Windows
checkout without symlink support a text stub holding the target path; `resolve`
follows both.
"""
from pathlib import Path

from .models import is_baseline_file
from .paths import PARSED_DIR

_MAX_STUB_BYTES = 4096


def resolve(path) -> Path:
    """The file behind a directory entry."""
    path = Path(path)
    try:
        if path.stat().st_size >= _MAX_STUB_BYTES:
            return path
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return path
    if not text or "\n" in text:
        return path
    target = (path.parent / text).resolve()
    return target if target.is_file() else path


def parsed_files(parsed_dir=None, baseline_only=False) -> list[Path]:
    """Every .jsonl in the parsed directory, sorted."""
    parsed_dir = Path(parsed_dir) if parsed_dir else PARSED_DIR
    files = sorted(parsed_dir.glob("*.jsonl"))
    if baseline_only:
        files = [f for f in files if is_baseline_file(f.name)]
    return [resolve(f) for f in files]
