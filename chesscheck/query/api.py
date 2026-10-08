"""Query GPT, Gemini or Claude on the puzzle set.

    python -m chesscheck.query.api --provider gpt    --model-id gpt-5.4         --end-index 2917
    python -m chesscheck.query.api --provider claude --model-id claude-opus-4-6 --end-index 2917

GPT and Gemini: one request per puzzle, concurrent, each row appended as it
returns; resume with --start-index. Claude: the Message Batches API with a JSON
schema (--schema), polled until the batch ends.

Output rows are {"key": PuzzleId, "response": text}. A failed request is
recorded with "error" and a null response; --retry-errors re-requests those.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import clients
from ..paths import LLM_PROMPT, LLM_PROMPT_SCHEMA, RAW_API_DIR
from ..puzzles import SYSTEM_INSTRUCTION, iter_requests, user_content

PROVIDERS = ("gpt", "gemini", "claude")
DEFAULT_MODEL = {"gpt": "gpt-5-mini", "gemini": "gemini-2.5-flash", "claude": "claude-sonnet-4-6"}


def output_path(output_dir, prompt_file, model_id) -> Path:
    return Path(output_dir) / f"outputs_{Path(prompt_file).stem}_{model_id}.jsonl"


def drop_error_rows(output_file: Path) -> set[str]:
    """Drop error rows from an existing output file; return the keys kept."""
    kept, done, failed = [], set(), 0
    if output_file.is_file():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("error") or not isinstance(rec.get("response"), str):
                    failed += 1
                else:
                    kept.append(line)
                    done.add(rec.get("key"))
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(kept)
    print(f"--retry-errors: kept {len(kept)} rows, dropped {failed} error rows", flush=True)
    return done


def make_caller(provider, model_id):
    if provider == "gpt":
        client = clients.openai_client()

        def call(content):
            r = client.responses.create(model=model_id, input=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": content}])
            return r.output_text
    elif provider == "gemini":
        from google.genai import types
        client = clients.gemini_client()

        def call(content):
            r = client.models.generate_content(
                model=model_id, contents=[content],
                config=types.GenerateContentConfig(temperature=1, system_instruction=SYSTEM_INSTRUCTION))
            return r.text
    else:
        raise ValueError(provider)
    return call


def run_concurrent(call, requests, prompt_text, output_file, workers, provide_legal_moves):
    def process(req):
        try:
            return {"key": req.key, "response": call(user_content(req.fen, prompt_text, provide_legal_moves))}
        except Exception as e:
            print(f"Row {req.index} ({req.key}) failed: {e}")
            return {"key": req.key, "response": None, "error": str(e)}

    lock = threading.Lock()
    with open(output_file, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, req): req for req in requests}
        for fut in as_completed(futures):
            print(f"Finished row {futures[fut].index}...")
            with lock:
                out.write(json.dumps(fut.result()) + "\n")
                out.flush()


def run_claude_batch(model_id, requests, prompt_text, output_file, schema_file, max_tokens, provide_legal_moves):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    schema = json.loads(Path(schema_file).read_text(encoding="utf-8"))
    client = clients.anthropic_client()
    batch_requests = [
        Request(custom_id=req.key, params=MessageCreateParamsNonStreaming(
            model=model_id, max_tokens=max_tokens, system=SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": user_content(req.fen, prompt_text, provide_legal_moves)}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        ))
        for req in requests
    ]
    if not batch_requests:
        raise SystemExit("No puzzle rows to send")

    start = time.time()
    batch = client.messages.batches.create(requests=batch_requests)
    print(batch)
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        print(batch.processing_status)
        if batch.processing_status == "ended":
            break
        print(f"Batch {batch.id} is still processing...")
        time.sleep(60)

    with open(output_file, "a", encoding="utf-8") as out:
        for entry in client.messages.batches.results(batch.id):
            row = {"key": entry.custom_id, "response": None}
            if entry.result.type == "succeeded":
                msg = entry.result.message
                row["response"] = "".join(getattr(b, "text", "") for b in msg.content
                                          if getattr(b, "type", None) == "text")
                row["stop_reason"] = msg.stop_reason
                row["usage"] = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else None
                print(f"[{entry.custom_id}] succeeded")
            else:
                row["error"] = f"{entry.result.type}: {getattr(entry.result, 'error', '')}"
                print(f"[{entry.custom_id}] {row['error']}")
            out.write(json.dumps(row) + "\n")
    print(f"Appended results to {output_file} ({(time.time() - start) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, required=True)
    ap.add_argument("--model-id", help="default: gpt-5-mini / gemini-2.5-flash / claude-sonnet-4-6")
    ap.add_argument("--prompt", default=str(LLM_PROMPT), help="prompt file")
    ap.add_argument("--start-index", type=int, default=0, help="first puzzle row (0-based, inclusive)")
    ap.add_argument("--end-index", type=int, default=None, help="end puzzle row (exclusive)")
    ap.add_argument("--output-dir", default=str(RAW_API_DIR))
    ap.add_argument("--max-workers", type=int, default=20, help="concurrent requests (gpt, gemini)")
    ap.add_argument("--provide-legal-moves", action="store_true",
                    help="prepend the position's true legal moves (python-chess) to the prompt")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-request only the rows in [start, end) whose output row is an error or missing")
    ap.add_argument("--schema", default=str(LLM_PROMPT_SCHEMA), help="JSON schema for Claude structured output")
    ap.add_argument("--max-tokens", type=int, default=16000, help="Claude output budget per request")
    args = ap.parse_args()

    model_id = args.model_id or DEFAULT_MODEL[args.provider]
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    output_file = output_path(args.output_dir, args.prompt, model_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    requests = list(iter_requests(args.start_index, args.end_index))
    if args.retry_errors:
        done = drop_error_rows(output_file)
        requests = [r for r in requests if r.key not in done]
        print(f"--retry-errors: re-requesting {len(requests)} rows", flush=True)

    if args.provider == "claude":
        run_claude_batch(model_id, requests, prompt_text, output_file, args.schema,
                         args.max_tokens, args.provide_legal_moves)
    else:
        run_concurrent(make_caller(args.provider, model_id), requests, prompt_text,
                       output_file, args.max_workers, args.provide_legal_moves)


if __name__ == "__main__":
    main()
