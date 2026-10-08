"""Bootstrap 95% CIs over puzzles for every metric and model.

Reads results/scores/*_rows.csv and the judged concept-coverage rows. Writes
results/tables/bootstrap_conditioned_metrics.csv (one row per model and metric,
with its dimension in `category`) and bootstrap_category_summary.csv (the five
dimension scores side by side). explanation_quality_score is the mean of
similarity and concept coverage; the reasoning category holds first-move
accuracy and whole-PV correctness.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import models, references
from ..paths import PUZZLE_LABELS_CSV, SCORES_DIR, TABLES_DIR

ROWS_FILES = {
    "factuality": SCORES_DIR / "factuality_rows.csv",
    "consistency": SCORES_DIR / "consistency_rows.csv",
    "strategic": SCORES_DIR / "strategic_rows.csv",
    "explanation_quality": SCORES_DIR / "explanation_rows.csv",
    "reasoning": SCORES_DIR / "correctness_rows.csv",
}
CONCEPT_COVERAGE_ROWS_CSV = SCORES_DIR / "decomposition_recall_rows.csv"
NON_METRIC_COLUMNS = {"llm_file", "source", "row_index", "model"}

# Minimal-prompt first-move predictions, one line per puzzle. Optional.
BASELINE_DIRS = [Path("results_baseline/api_results/parsed"), Path("results_baseline/local_results/parsed_local")]

CONDITIONED_OUTPUT_CSV = TABLES_DIR / "bootstrap_conditioned_metrics.csv"
CATEGORY_SUMMARY_CSV = TABLES_DIR / "bootstrap_category_summary.csv"

N_RESAMPLES = 10000
SEED = 0


def bootstrap_ci(scores: np.ndarray, n_resamples: int, rng: np.random.Generator, alpha: float = 0.05):
    n = len(scores)
    idx = rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)
    means = scores[idx].mean(axis=1)
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return scores.mean(), lower, upper, means.std(ddof=1)


def _ci_row(category, model, metric, values, rng):
    mean, lower, upper, std = bootstrap_ci(values, N_RESAMPLES, rng)
    return {"category": category, "model": model, "metric": metric, "n": len(values),
            "mean": mean, "ci_lower_95": lower, "ci_upper_95": upper, "bootstrap_std": std}


def load_rows(csv_path: Path) -> pd.DataFrame:
    """A rows file restricted to baseline files, with a `model` column."""
    df = pd.read_csv(csv_path)
    df = df[df["llm_file"].apply(models.is_baseline_file)]
    df["model"] = df["llm_file"].apply(models.canonical_id)
    df = df[df["model"].notna()]
    if "similarity_nm" in df.columns or "similarity_fm" in df.columns:
        df = references.apply_failure_rule(df.copy())
    return df


def bootstrap_rows(df: pd.DataFrame, category: str, rng) -> list[dict]:
    metric_columns = [c for c in df.columns if c not in NON_METRIC_COLUMNS]
    rows = []
    for model, group in df.groupby("model"):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values):
                rows.append(_ci_row(category, model, metric, values, rng))
    return rows


def load_ablation_rows(labels_df: pd.DataFrame, expected_models: set[str]) -> pd.DataFrame:
    true_first = labels_df["Moves"].astype(str).str.split().str[1].str.lower()
    rows = []
    for baseline_dir in BASELINE_DIRS:
        if not baseline_dir.exists():
            continue
        for fn in sorted(baseline_dir.glob("*.txt")):
            model = models.canonical_id(fn.name)
            if model is None or model not in expected_models:
                continue
            with fn.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= len(labels_df):
                        break
                    pred = line.strip().lower()
                    ok = pred and pred != "error" and pred == true_first.iat[idx]
                    rows.append((model, idx, 1.0 if ok else 0.0))
    df = pd.DataFrame(rows, columns=["model", "row_index", "ablation_accuracy"])
    if df.empty:
        return df
    return df.groupby(["model", "row_index"], as_index=False)["ablation_accuracy"].mean()


def explanation_quality_rows(explanation: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, puzzle): similarity, concept coverage and their mean."""
    combined = explanation[["model", "row_index", "similarity"]].copy()
    if CONCEPT_COVERAGE_ROWS_CSV.exists():
        # Joined on PuzzleId; put on similarity's scale.
        cov = pd.read_csv(CONCEPT_COVERAGE_ROWS_CSV)[["PuzzleId", "model", "recall"]]
        pid_to_row = {pid: i for i, pid in enumerate(labels_df["PuzzleId"])}
        cov["row_index"] = cov["PuzzleId"].map(pid_to_row)
        cov = cov.dropna(subset=["row_index"]).astype({"row_index": int})
        scale = 100.0 if combined["similarity"].max() > 1.0 else 1.0
        cov["concept_coverage"] = cov["recall"] * scale
        combined = combined.merge(cov[["model", "row_index", "concept_coverage"]], on=["model", "row_index"], how="left")
        # Puzzles whose reference has zero claims are absent by design.
        for model, frac in combined.groupby("model")["concept_coverage"].apply(lambda s: s.isna().mean()).items():
            if frac > 0:
                print(f"concept_coverage: {model} has no coverage row for {frac:.1%} of puzzles")
    else:
        print(f"concept_coverage: {CONCEPT_COVERAGE_ROWS_CSV} not found; Explanation Quality = similarity only")
    eq_cols = [c for c in ("similarity", "concept_coverage") if c in combined.columns]
    combined["explanation_quality_score"] = combined[eq_cols].mean(axis=1)
    return combined


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    labels_df = pd.read_csv(PUZZLE_LABELS_CSV)
    rng = np.random.default_rng(SEED)

    rows, frames = [], {}
    for category, csv_path in ROWS_FILES.items():
        if not csv_path.exists():
            print(f"Skipping {category}: {csv_path} not found")
            continue
        frames[category] = load_rows(csv_path)
        rows.extend(bootstrap_rows(frames[category], category, rng))

    if "explanation_quality" in frames:
        eq = explanation_quality_rows(frames["explanation_quality"], labels_df)
        for metric in ("concept_coverage", "explanation_quality_score"):
            if metric in eq.columns:
                for model, group in eq.groupby("model"):
                    values = group[metric].dropna().to_numpy(dtype=float)
                    if len(values):
                        rows.append(_ci_row("explanation_quality", model, metric, values, rng))
    if "reasoning" in frames:
        ablation = load_ablation_rows(labels_df, set(frames["reasoning"]["model"]))
        if not ablation.empty:
            for model, group in ablation.groupby("model"):
                rows.append(_ci_row("reasoning", model, "ablation_accuracy",
                                    group["ablation_accuracy"].dropna().to_numpy(dtype=float), rng))

    metrics = pd.DataFrame(rows)
    print(metrics.to_string(index=False))
    metrics.to_csv(CONDITIONED_OUTPUT_CSV, index=False)
    print(f"\nWrote {CONDITIONED_OUTPUT_CSV}")

    # Dimension scores side by side, ordered by first-move accuracy.
    def pick(category, metric):
        sub = metrics[(metrics["category"] == category) & (metrics["metric"] == metric)].set_index("model")
        return sub[["mean", "ci_lower_95", "ci_upper_95"]].add_prefix(category + "_")

    parts = [pick("factuality", "factuality_score"), pick("consistency", "consistency_score"),
             pick("strategic", "strategic_score"), pick("explanation_quality", "explanation_quality_score"),
             pick("reasoning", "first_move_accuracy")]
    parts = [p for p in parts if not p.empty]
    summary = parts[0]
    for part in parts[1:]:
        summary = summary.join(part, how="outer")
    order_col = "reasoning_mean" if "reasoning_mean" in summary.columns else summary.columns[0]
    summary = summary.sort_values(order_col, ascending=False).reset_index().rename(columns={"index": "model"})
    print(f"\n{summary.to_string(index=False)}")
    summary.to_csv(CATEGORY_SUMMARY_CSV, index=False)
    print(f"\nWrote {CATEGORY_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
