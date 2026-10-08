"""Input and output locations. Inputs under data/ and prompts/ are committed;
outputs under results/ are not.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"
KEYS = ROOT / "keys"

PUZZLES_CSV = DATA / "puzzles.csv"                       # the 2,917 Lichess puzzles
PUZZLE_LABELS_CSV = DATA / "puzzle_labels.csv"           # ground truth per puzzle
COMMENTARY_NM_CSV = DATA / "puzzle_commentary.csv"       # national master's explanations
COMMENTARY_FM_CSV = DATA / "puzzle_commentary_fm.csv"    # FIDE master's explanations
COMMENTARY_DECOMPOSITION_CSV = DATA / "puzzle_commentary_decomposition.csv"  # NM claims
STOCKFISH_TERMS_CSV = DATA / "stockfish_terms.csv"       # raw Stockfish 11 eval terms

LLM_PROMPT = PROMPTS / "llm_prompt.txt"
LLM_PROMPT_PARAPHRASE = PROMPTS / "llm_prompt_paraphrase.txt"
ENUMERATION_PROMPT = PROMPTS / "enumeration_prompt.txt"
LLM_PROMPT_SCHEMA = PROMPTS / "llm_prompt_schema.json"
ENUMERATION_PROMPT_SCHEMA = PROMPTS / "enumeration_prompt_schema.json"
DECOMPOSITION_PROMPT = PROMPTS / "explanation_decomposition.txt"
COVERAGE_JUDGE_PROMPT = PROMPTS / "concept_coverage_judge.txt"

RAW_API_DIR = RESULTS / "raw" / "api"          # closed-model responses
RAW_LOCAL_DIR = RESULTS / "raw" / "local"      # open-weight responses
# Overridable with the PARSED_DIR environment variable or --parsed-dir.
PARSED_DIR = Path(os.environ.get("PARSED_DIR", RESULTS / "parsed"))
SCORES_DIR = RESULTS / "scores"                # per-puzzle rows and summaries
TABLES_DIR = RESULTS / "tables"                # bootstrap CIs and the paper tables
COVERAGE_DIR = RESULTS / "concept_coverage"    # decomposition inputs, claims, judged files


def api_key(name: str) -> str | None:
    """Contents of keys/<name> (plain text), or None if the file is absent."""
    path = KEYS / name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return None
