"""Concept coverage: the share of the master's claims a model's explanation covers.

The reference is data/puzzle_commentary_decomposition.csv (the NM explanations
as atomic claims). Steps:

    inputs     parsed explanations -> results/concept_coverage/inputs/<model>_explanations.json
    decompose  split each explanation into claims (prompts/explanation_decomposition.txt)
               -> results/concept_coverage/decompositions/<model>_explanation_decompositions.csv
    judge      count the reference claims covered (prompts/concept_coverage_judge.txt)
               -> results/concept_coverage/judged/chess_explanation_recall_<token>.csv
    summary    all judged files -> results/scores/decomposition_recall_rows.csv

    python -m chesscheck.score.concept_coverage <step> [--models ...] [--limit N]

decompose and judge use the OpenAI API, resume by PuzzleId, and write a sidecar
JSON with model, date, prompt hash and call counts. Default models: the six
open models of the re-run; the other eight were judged by hand with the same
prompts and their files dropped into judged/.

Rules shared with the manual round:
  * an empty explanation decomposes to zero claims without a call
  * a puzzle whose reference has zero claims is omitted from the recall file
  * a puzzle whose candidate has zero claims scores 0 without a call
  * the judge sees only atomic_claim text
"""
import argparse
import csv
import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from .. import clients, models
from ..parsed import parsed_files
from ..paths import (COMMENTARY_DECOMPOSITION_CSV, COVERAGE_DIR, COVERAGE_JUDGE_PROMPT,
                     DECOMPOSITION_PROMPT, PARSED_DIR, PUZZLE_LABELS_CSV, SCORES_DIR)

INPUTS_DIR = COVERAGE_DIR / "inputs"
DECOMPOSITIONS_DIR = COVERAGE_DIR / "decompositions"
JUDGED_DIR = COVERAGE_DIR / "judged"
JUDGED_PREFIX = "chess_explanation_recall_"
ROWS_CSV = SCORES_DIR / "decomposition_recall_rows.csv"
SUMMARY_CSV = SCORES_DIR / "decomposition_recall_summary.csv"

JUDGE_MODEL = "gpt-5.6-sol"     # same as the manual round


def build_inputs(model_ids, parsed_dir, out_dir):
    labels = pd.read_csv(PUZZLE_LABELS_CSV)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_model = {}
    for f in parsed_files(parsed_dir, baseline_only=True):
        mid = models.canonical_id(f.name)
        if mid and mid not in by_model:
            by_model[mid] = f
    for mid in model_ids:
        src = by_model.get(mid)
        if src is None:
            print(f"{mid:32} SKIP (no parsed file in {parsed_dir or PARSED_DIR})")
            continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        if len(rows) != len(labels):
            raise SystemExit(f"{mid}: {len(rows)} parsed rows vs {len(labels)} labels -- "
                             "order alignment would be wrong, refusing to write")
        out, n_text = [], 0
        for (pid, fen), row in zip(zip(labels.PuzzleId, labels.starting_position_fen), rows):
            # A failed response gets an empty explanation (recall 0).
            text = "" if row.get("error") else (row.get("final_explanation_summary") or "")
            text = text if isinstance(text, str) else json.dumps(text)
            n_text += bool(text.strip())
            out.append({"PuzzleId": pid, "starting_position_fen": fen, "final_explanation_summary": text})
        dst = out_dir / f"{mid}_explanations.json"
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{mid:32} {n_text:4d}/{len(out)} non-empty explanations -> {dst}")


def _strip_fence(t):
    return re.sub(r"\s*```$", "", re.sub(r"^```(?:json|csv)?\s*", "", t.strip()))


def _call(client, system, user, retries=4):
    delay = 2.0
    for attempt in range(retries):
        try:
            r = client.responses.create(model=JUDGE_MODEL, input=[
                {"role": "system", "content": system}, {"role": "user", "content": user}])
            return _strip_fence(r.output_text)
        except Exception:                 # rate limit / transient
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _sidecar(path, **fields):
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2))


def decompose(mid, client, limit, workers):
    src = INPUTS_DIR / f"{mid}_explanations.json"
    if not src.exists():
        print(f"{mid:32} SKIP decompose (no {src.name}; run the `inputs` step)")
        return
    rows = json.load(open(src, encoding="utf-8"))
    if limit:
        rows = rows[:limit]
    DECOMPOSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = DECOMPOSITIONS_DIR / f"{mid}_explanation_decompositions.csv"
    meta_path = DECOMPOSITIONS_DIR / f"{mid}_meta.json"
    template = DECOMPOSITION_PROMPT.read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()[:12]

    done = {}
    if out.exists():
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done[r["PuzzleId"]] = r["decomposition"]
    lock = threading.Lock()
    calls = failures = 0

    def work(row):
        pid, fen, text = row["PuzzleId"], row["starting_position_fen"], row["final_explanation_summary"]
        if not text.strip():
            return pid, json.dumps({"fen": fen, "claims": []}), False, False
        # str.replace, not str.format: the template is full of JSON braces.
        user = template.replace("{FEN}", fen).replace("{EXPLANATION}", text)
        for _ in range(3):
            try:
                raw = _call(client, "You are a chess-explanation decomposition system. Return only valid JSON.", user)
                d = json.loads(raw)
                if isinstance(d, dict) and isinstance(d.get("claims"), list):
                    d.setdefault("fen", fen)
                    return pid, json.dumps(d, ensure_ascii=False), True, False
            except Exception:
                pass
        return pid, json.dumps({"fen": fen, "claims": [], "error": "decomposition_failed"}), True, True

    todo = [r for r in rows if r["PuzzleId"] not in done]
    print(f"{mid:32} decompose: {len(done)} cached, {len(todo)} to do "
          f"({sum(1 for r in todo if r['final_explanation_summary'].strip())} need a call)")
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["PuzzleId", "decomposition"])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(work, r) for r in todo]):
                pid, dec, called, failed = fut.result()
                with lock:
                    w.writerow([pid, dec])
                    f.flush()
                    done[pid] = dec
                    calls += called
                    failures += failed
                    if len(done) % 250 == 0:
                        print(f"  {mid}: {len(done)}/{len(rows)}")
    # Rewrite in labels order.
    ordered = [(r["PuzzleId"], done[r["PuzzleId"]]) for r in rows if r["PuzzleId"] in done]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PuzzleId", "decomposition"])
        w.writerows(ordered)
    _sidecar(meta_path, stage="decomposition", model=JUDGE_MODEL, date=str(date.today()),
             prompt_sha256_12=prompt_sha, rows=len(ordered), api_calls_this_run=calls,
             failures_this_run=failures, temperature="api default (not set)")
    print(f"{mid:32} decompose DONE: {len(ordered)} rows, {calls} calls, {failures} failures -> {out.name}")


def _claims(cell):
    try:
        return [c["atomic_claim"] for c in json.loads(cell)["claims"] if isinstance(c, dict) and c.get("atomic_claim")]
    except Exception:
        return []


def _judge_rules():
    """The judging rules from the prompt file."""
    text = COVERAGE_JUDGE_PROMPT.read_text(encoding="utf-8")
    start = text.index("Otherwise, for each reference claim")
    end = text.index("Output requirements")
    return text[start:end].strip()


def judge(mid, client, limit, batch, workers):
    dec_path = DECOMPOSITIONS_DIR / f"{mid}_explanation_decompositions.csv"
    if not dec_path.exists():
        print(f"{mid:32} SKIP judge (no decompositions yet)")
        return
    token = models.BY_ID[mid].token
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)
    final = JUDGED_DIR / f"{JUDGED_PREFIX}{token}.csv"
    partial = JUDGED_DIR / f"{JUDGED_PREFIX}{token}.partial.csv"
    meta_path = JUDGED_DIR / f"{JUDGED_PREFIX}{token}_meta.json"

    ref = pd.read_csv(COMMENTARY_DECOMPOSITION_CSV)[["PuzzleId", "decomposition"]]
    cand = pd.read_csv(dec_path).set_index("PuzzleId")["decomposition"]
    rules = _judge_rules()
    prompt_sha = hashlib.sha256(COVERAGE_JUDGE_PROMPT.read_bytes()).hexdigest()[:12]

    items = []
    for pid, rcell in zip(ref.PuzzleId, ref.decomposition):
        rc = _claims(rcell)
        if not rc:
            continue                      # omitted per the manual round's rule
        if pid not in cand.index:
            continue                      # --limit on decomposition
        items.append((pid, rc, _claims(cand[pid])))
    if limit:
        items = items[:limit]

    done = {}
    if partial.exists():
        for r in csv.DictReader(open(partial, encoding="utf-8")):
            done[r["PuzzleId"]] = (int(r["num_ref_claims"]), int(r["covered_count"]))
    free = [(pid, rc) for pid, rc, cc in items if pid not in done and not cc]
    for pid, rc in free:
        done[pid] = (len(rc), 0)          # empty candidate: recall 0, no call
    todo = [(pid, rc, cc) for pid, rc, cc in items if pid not in done]
    print(f"{mid:32} judge: {len(items)} scoreable, {len(free)} empty-candidate (free), "
          f"{len(done) - len(free)} cached, {len(todo)} to judge in batches of {batch}")

    system = ("You are a strict chess-explanation recall judge. " + rules +
              "\nOutput requirements:\n- Return ONLY CSV, no prose or code fences.\n"
              "- Header row exactly: PuzzleId,num_ref_claims,covered_count\n"
              "- One data row per PuzzleId given, in the order given. Integers only.")

    def fmt(group):
        parts = []
        for pid, rc, cc in group:
            parts.append(f"PuzzleId: {pid}\nReference claims ({len(rc)}):\n" +
                         "\n".join(f"  R{i+1}. {c}" for i, c in enumerate(rc)) +
                         f"\nCandidate claims ({len(cc)}):\n" +
                         "\n".join(f"  C{i+1}. {c}" for i, c in enumerate(cc)))
        return "\n\n".join(parts)

    def parse(raw, group):
        got = {r["PuzzleId"].strip(): r for r in csv.DictReader(io.StringIO(raw.strip()))}
        out = {}
        for pid, rc, cc in group:
            r = got.get(pid)
            n, c = int(r["num_ref_claims"]), int(r["covered_count"])
            if n != len(rc) or not (0 <= c <= n):
                raise ValueError(f"bad row for {pid}: {r}")
            out[pid] = (n, c)
        return out

    lock = threading.Lock()
    stats = {"calls": 0, "failed_puzzles": 0}

    def work(group):
        for _ in range(2):
            try:
                raw = _call(client, system, fmt(group))
                with lock:
                    stats["calls"] += 1
                return parse(raw, group)
            except Exception:
                continue
        if len(group) > 1:                # split and retry singly
            out = {}
            for g in group:
                out.update(work([g]))
            return out
        with lock:
            stats["failed_puzzles"] += 1
        pid, rc, _ = group[0]
        return {pid: (len(rc), 0)}        # conservative: unjudged counts as uncovered

    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with open(partial, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["PuzzleId", "num_ref_claims", "covered_count", "recall"])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(work, g) for g in groups]):
                res = fut.result()
                with lock:
                    for pid, (n, c) in res.items():
                        done[pid] = (n, c)
                        w.writerow([pid, n, c, f"{c / n:.4f}"])
                    f.flush()

    if limit:
        print(f"{mid:32} judge (limited): {len(done)} scored ({len(free)} free); final file NOT written")
        return
    if len(done) < len(items):
        print(f"{mid:32} judge INCOMPLETE ({len(done)}/{len(items)}); rerun to resume")
        return
    with open(final, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PuzzleId", "num_ref_claims", "covered_count", "recall"])
        for pid, rc, cc in items:
            n, c = done[pid]
            w.writerow([pid, n, c, f"{c / n:.4f}"])
    partial.unlink()
    _sidecar(meta_path, stage="judge", model=JUDGE_MODEL, date=str(date.today()), prompt_sha256_12=prompt_sha,
             batch_size=batch, rows=len(items), api_calls_this_run=stats["calls"],
             failed_puzzles_scored_as_zero=stats["failed_puzzles"], temperature="api default (not set)")
    print(f"{mid:32} judge DONE: {len(items)} rows, {stats['calls']} calls, "
          f"{stats['failed_puzzles']} unjudged->0 -> {final.name}")


REQUIRED_COLUMNS = ["PuzzleId", "num_ref_claims", "covered_count", "recall"]


def _model_of(path: Path) -> str:
    token = path.stem[len(JUDGED_PREFIX):] if path.stem.startswith(JUDGED_PREFIX) else path.stem
    return models.BY_TOKEN[token].id if token in models.BY_TOKEN else token


def _load_judged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Skipping {path.name}: missing columns {missing}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = df[REQUIRED_COLUMNS].copy()
    for c in REQUIRED_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = (df["PuzzleId"].notna() & df["num_ref_claims"].notna() & df["covered_count"].notna()
             & df["recall"].notna() & (df["recall"] >= 0) & (df["recall"] <= 1)
             & (df["covered_count"] <= df["num_ref_claims"]) & (df["num_ref_claims"] > 0))
    if (~valid).sum():
        print(f"{path.name}: skipped {int((~valid).sum())}/{len(df)} invalid rows")
    return df[valid]


def summary(input_dir, output_rows, output_summary):
    files = sorted(Path(input_dir).glob(f"{JUDGED_PREFIX}*.csv"))
    files = [f for f in files if not f.name.endswith(".partial.csv")]
    if not files:
        print(f"No {JUDGED_PREFIX}*.csv files in {input_dir}")
        return
    parts = []
    for path in files:
        df = _load_judged(path)
        if not df.empty:
            df.insert(1, "model", _model_of(path))
            parts.append(df)
    if not parts:
        print("No valid rows found across any input file.")
        return
    rows = pd.concat(parts, ignore_index=True)
    Path(output_rows).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_rows, index=False)
    print(f"Wrote {output_rows} ({len(rows)} rows across {rows['model'].nunique()} models)")

    summ = (rows.groupby("model")
            .agg(n_scored=("recall", "size"), recall_mean=("recall", "mean"), recall_median=("recall", "median"))
            .reset_index().sort_values("recall_mean", ascending=False))
    summ["recall_mean"] = summ["recall_mean"].round(4)
    summ["recall_median"] = summ["recall_median"].round(4)
    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(output_summary, index=False)
    print(f"Wrote {output_summary}")
    print(summ.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)

    p = sub.add_parser("inputs", help="explanations per model from the parsed files")
    p.add_argument("--models", nargs="+", default=models.RERUN_MODELS, help="model ids")
    p.add_argument("--parsed-dir", default=None)
    p.add_argument("--out-dir", default=str(INPUTS_DIR))

    for step in ("decompose", "judge"):
        p = sub.add_parser(step)
        p.add_argument("--models", nargs="+", default=models.RERUN_MODELS, help="model ids")
        p.add_argument("--limit", type=int, default=0, help="smoke test: first N puzzles only")
        p.add_argument("--workers", type=int, default=6)
        if step == "judge":
            p.add_argument("--batch", type=int, default=25, help="puzzles per judge call")

    p = sub.add_parser("summary", help="aggregate the judged files")
    p.add_argument("--input-dir", default=str(JUDGED_DIR))
    p.add_argument("--output-rows", default=str(ROWS_CSV))
    p.add_argument("--output-summary", default=str(SUMMARY_CSV))

    args = ap.parse_args()
    if args.step == "inputs":
        build_inputs(args.models, args.parsed_dir, Path(args.out_dir))
    elif args.step == "decompose":
        client = clients.openai_client()
        for m in args.models:
            decompose(m, client, args.limit, args.workers)
    elif args.step == "judge":
        client = clients.openai_client()
        for m in args.models:
            judge(m, client, args.limit, args.batch, args.workers)
    else:
        summary(args.input_dir, args.output_rows, args.output_summary)


if __name__ == "__main__":
    main()
