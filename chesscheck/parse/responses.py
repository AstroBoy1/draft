"""Parse raw model responses into one JSONL of prompt fields per model.

    python -m chesscheck.parse.responses --input-dir results/raw/api
    python -m chesscheck.parse.responses --input-file results/raw/local/outputs_llm_prompt_Qwen3-8B.jsonl

Raw rows may be {"key", "response"}, {PuzzleId: text}, or Claude batch result
entries; the shape is detected per line. The text is then stripped of any
<think> block and markdown fence and parsed as JSON, falling back to the first
balanced {...} object.

Output: <output-dir>/parsed_<raw file name>, one row per input row with the
seven prompt fields, `error` (null on success) and `repair_status`. Parsing is
native unless --repair is given, which sends failed rows to an LLM
(llm_repair.py).
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from ..paths import PARSED_DIR

FIELDS = [
    "material_count",
    "all_legal_moves_uci",
    "candidate_moves_uci",
    "positional_evaluation",
    "principal_variation_analysis",
    "final_explanation_summary",
    "best_move",
]
LIST_FIELDS = {"all_legal_moves_uci", "candidate_moves_uci", "principal_variation_analysis"}


def extract_key_and_text(obj: dict[str, Any], index: int) -> tuple[str, str | None, str | None]:
    """(key, text, error) for a raw row of any known shape; text is None when
    the row records a failed request."""
    if "custom_id" in obj and "result" in obj:            # Claude Message Batches entry
        key = str(obj["custom_id"])
        result = obj.get("result") or {}
        message = result.get("message")
        if not isinstance(message, dict):
            return key, None, f"missing_message: result_type={result.get('type', 'unknown')}"
        content = message.get("content")
        text = content[0].get("text") if isinstance(content, list) and content and isinstance(content[0], dict) else None
        if not isinstance(text, str):
            return key, None, "missing_text_content"
        return key, text, None
    if "key" in obj and "response" in obj:                # gpt / gemini / claude rows
        key = str(obj.get("key") or f"request-{index}")
        text = obj.get("response")
        if text is None:
            return key, None, f"missing_response: {obj.get('error') or 'no response'}"
        return key, text if isinstance(text, str) else json.dumps(text), None
    if len(obj) == 1:                                     # {PuzzleId: text} local rows
        key, text = next(iter(obj.items()))
        return str(key), text if isinstance(text, str) else json.dumps(text), None
    for key, value in obj.items():
        if isinstance(value, str):
            return str(key), value, None
    raise ValueError("Could not locate response text in line object")


def strip_think_block(text: str) -> str:
    """Drop everything through the last </think>."""
    idx = text.rfind("</think>")
    return text if idx == -1 else text[idx + len("</think>"):].strip()


def strip_markdown_json_fence(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def find_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_structured_json(text: str) -> dict[str, Any]:
    text = strip_think_block(text)
    cleaned = strip_markdown_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = find_balanced_json_object(cleaned) or find_balanced_json_object(text)
        if not candidate:
            raise ValueError("No valid JSON object found in response text")
        data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Parsed response JSON is not an object")
    return data


def normalize_fields(key: str, parsed: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"key": key}
    for field in FIELDS:
        out[field] = parsed.get(field, [] if field in LIST_FIELDS else None)
    return out


def error_row(key: str, error: str, repair_status=None) -> dict[str, Any]:
    return {"key": key, **{f: None for f in FIELDS}, "error": error, "repair_status": repair_status}


def parse_file(input_file: str, output_file: str, limit: int | None = None, repair: bool = False):
    rows: list[dict[str, Any]] = []
    errors = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for row_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("JSONL row is not a JSON object")
                key, text, error = extract_key_and_text(obj, row_num)
            except Exception as e:
                rows.append(error_row(f"request-{row_num}", str(e)))
                errors += 1
                continue
            if error is not None:
                rows.append(error_row(key, error))
                errors += 1
                continue
            try:
                parsed = normalize_fields(key, parse_structured_json(text))
                parsed["error"], parsed["repair_status"] = None, None
                rows.append(parsed)
            except Exception as e:
                if not repair:
                    rows.append(error_row(key, str(e)))
                    errors += 1
                    continue
                from .llm_repair import repair_json_with_llm
                repaired, status = repair_json_with_llm(key, text, input_file)
                if repaired is not None:
                    parsed = normalize_fields(key, repaired)
                    parsed["error"], parsed["repair_status"] = None, status
                    rows.append(parsed)
                else:
                    rows.append(error_row(key, str(e), status))
                    errors += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")

    repaired = sum(1 for r in rows if r.get("repair_status") in ("llm_repaired", "cached_repaired"))
    print(f"Parsed {len(rows)} rows from {input_file}")
    print(f"Native parse failures: {errors + repaired} ({repaired} recovered via LLM repair, {errors} still unresolved)")
    print(f"Wrote: {output_file}")
    return len(rows), errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", nargs="+", help="raw JSONL file(s)")
    src.add_argument("--input-dir", help="parse every .jsonl in this directory")
    ap.add_argument("--output-dir", default=str(PARSED_DIR))
    ap.add_argument("--output-file", help="output path (single --input-file only)")
    ap.add_argument("--limit", type=int, default=None, help="max rows per input file")
    ap.add_argument("--repair", action="store_true", help="LLM repair for rows that fail native parsing")
    args = ap.parse_args()

    if args.input_file:
        inputs = [Path(p) for p in args.input_file]
    else:
        inputs = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix == ".jsonl")
        if not inputs:
            print(f"No .jsonl files found in {args.input_dir}")
            return
    if args.output_file and len(inputs) != 1:
        ap.error("--output-file needs exactly one --input-file")

    total = total_errors = 0
    for path in inputs:
        out = args.output_file or os.path.join(args.output_dir, f"parsed_{path.name}")
        n, e = parse_file(str(path), out, args.limit, args.repair)
        total += n
        total_errors += e
    if len(inputs) > 1:
        print(f"Total parsed rows: {total}; total errors: {total_errors}")


if __name__ == "__main__":
    main()
