"""LLM fallback for a response whose JSON fails to parse (parser --repair).

The model is instructed to extract only what is present in the text, never to
fill in missing content. Results are cached next to the source file
(repair_cache/), keyed by row key.
"""
import json
import os

from .. import clients

REPAIR_MODEL = "gpt-5.6-sol"

REPAIR_SYSTEM_INSTRUCTION = (
    "You are a strict JSON extraction tool. You will be given a chess-analysis "
    "response that failed to parse as valid JSON, plus the target schema. "
    "Extract ONLY the information that is already explicitly present in the "
    "source text into the exact JSON schema below. Do NOT invent, infer, guess, "
    "or fill in any value that is not explicitly present in the source text -- "
    "leave those fields null (or an empty list, matching the schema's shape). "
    "Do not correct, complete, or improve any chess content -- only recover and "
    "reformat what is already there. Return ONLY valid JSON, no other text."
)

TARGET_SCHEMA = """{
  "material_count": {"white": {"Q":int,"R":int,"B":int,"N":int,"P":int}, "black": {"Q":int,"R":int,"B":int,"N":int,"P":int}, "material_diff": {"side_ahead": "white or black or equal"}},
  "all_legal_moves_uci": ["uci move strings"],
  "candidate_moves_uci": ["uci move strings"],
  "positional_evaluation": {"material_summary": "white_better or black_better or equal", "mobility": "white_better or black_better or equal", "king_safety": "white_better or black_better or equal", "pawn_structure": "white_better or black_better or equal", "initial_eval_label": "white_better or black_better or equal"},
  "principal_variation_analysis": [{"candidate_move": "uci move", "pv_uci": ["uci move strings"], "eval_label": "white_better or black_better or equal", "explanation": "text"}],
  "final_explanation_summary": "text",
  "best_move": "uci move"
}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = clients.openai_client()
    return _client


def _cache_path_for(source_file: str) -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(source_file)), "repair_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, os.path.basename(source_file) + ".repair_cache.jsonl")


def _load_cache(cache_path: str) -> dict:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row
    return cache


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def repair_json_with_llm(key: str, raw_text: str, source_file: str):
    """(dict or None, status: cached_repaired | cached_failed | llm_repaired | llm_failed)."""
    cache_path = _cache_path_for(source_file)
    cache = _load_cache(cache_path)
    if key in cache:
        cached = cache[key]
        status = "cached_repaired" if cached.get("repaired_json") is not None else "cached_failed"
        return cached.get("repaired_json"), status

    repaired_json = None
    try:
        response = _get_client().responses.create(
            model=REPAIR_MODEL,
            input=[
                {"role": "system", "content": REPAIR_SYSTEM_INSTRUCTION},
                {"role": "user",
                 "content": f"Target schema:\n{TARGET_SCHEMA}\n\nSource text to extract from:\n{raw_text}"},
            ],
        )
        repaired_json = json.loads(_strip_markdown_fence(response.output_text))
        status = "llm_repaired"
    except Exception:
        repaired_json = None
        status = "llm_failed"

    with open(cache_path, "a", encoding="utf-8") as f:
        json.dump({"key": key, "raw_text": raw_text, "repaired_json": repaired_json, "status": status}, f)
        f.write("\n")
    return repaired_json, status
