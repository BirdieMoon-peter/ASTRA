"""Prioritize gold workset rows for annotation, oversampling rare phenomena."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from astra.config.task_schema import (
    ANNOTATIONS_DIR,
    PHENOMENON_LABELS,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
)
from astra.evaluation.baselines import rule_based_prediction
from astra.scoring.report_scorer import (
    HEDGE_LEXICON,
    OPTIMISM_LEXICON,
    RISK_LEXICON,
    _count_hits,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLD_WORKSET_PATH = ANNOTATIONS_DIR / "stratreportzh_gold_workset.jsonl"
DEV_IDS_PATH = ANNOTATIONS_DIR / "splits" / "dev_ids.json"
TEST_IDS_PATH = ANNOTATIONS_DIR / "splits" / "test_ids.json"
ALREADY_ANNOTATED_PATH = (
    ANNOTATIONS_DIR / "stratreportzh_dev_aligned_workset_annotated_normalized.jsonl"
)
OUTPUT_PATH = ANNOTATIONS_DIR / "annotation_priority_queue.jsonl"

# Rare-phenomenon weights: higher weight = more priority for annotation.
PHENOMENON_WEIGHT: dict[str, float] = {
    "euphemistic_risk": 3.0,
    "omitted_downside_context": 4.0,
    "title_body_mismatch": 2.0,
    "hedged_downside": 1.5,
    "none": 0.0,
}

# Method labels used in prediction JSONL files.
_METHOD_FILE_MAP: dict[str, str] = {
    "direct": "direct_baseline_predictions.jsonl",
    "cot": "cot_baseline_predictions.jsonl",
    "react": "react_baseline_predictions.jsonl",
    "astra": "astra_predictions.jsonl",
    "rule": "rule_baseline_predictions.jsonl",
}

# Annotation template matching the schema in annotation_guideline.md.
_ANNOTATION_TEMPLATE: dict[str, Any] = {
    "fundamental_sentiment": None,
    "strategic_optimism": None,
    "phenomenon": None,
    "evidence_spans": [],
    "annotation_confidence": None,
    "notes": "",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_predictions(predictions_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all prediction files, keyed by report_id then method name."""
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method_key, filename in _METHOD_FILE_MAP.items():
        path = predictions_dir / filename
        for row in _load_jsonl(path):
            rid = row.get("report_id")
            if rid:
                result[rid][method_key] = row
    return dict(result)


def _compute_lexicon_heuristic_score(title: str, summary: str) -> tuple[float, str]:
    """Use lexicon heuristics to estimate phenomenon and return (bonus, phenomenon).

    Returns a bonus score and the predicted phenomenon based on lexicon patterns.
    """
    title_risk = _count_hits(title, RISK_LEXICON)
    title_optimism = _count_hits(title, OPTIMISM_LEXICON)
    summary_risk = _count_hits(summary, RISK_LEXICON)
    summary_hedge = _count_hits(summary, HEDGE_LEXICON)
    summary_optimism = _count_hits(summary, OPTIMISM_LEXICON)

    # High risk + high optimism in title -> likely euphemistic_risk
    if title_risk > 0 and title_optimism > 0 and summary_risk > 0:
        return (PHENOMENON_WEIGHT["euphemistic_risk"] * 0.5, "euphemistic_risk")

    # Risk in summary but not in title, with hedge -> likely hedged_downside
    if summary_risk > 0 and title_risk == 0 and summary_hedge > 0:
        return (PHENOMENON_WEIGHT["hedged_downside"] * 0.5, "hedged_downside")

    # Strong optimism in title + risk in summary -> likely title_body_mismatch
    if title_optimism >= 2 and summary_risk > 0:
        return (PHENOMENON_WEIGHT["title_body_mismatch"] * 0.5, "title_body_mismatch")

    # Weaker version: some optimism in title + risk in summary
    if title_optimism > 0 and summary_risk > 0 and summary_optimism < title_optimism:
        return (PHENOMENON_WEIGHT["title_body_mismatch"] * 0.3, "title_body_mismatch")

    return (0.0, "none")


def _compute_priority_score(
    row: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> tuple[float, list[str]]:
    """Compute a priority score for a single row.

    Returns (priority_score, list_of_predicted_phenomena).
    """
    report_id = row["report_id"]
    title = row.get("title", "")
    summary = row.get("summary", "")

    # 1. Rule-based prediction
    rule_pred = rule_based_prediction(title, summary)
    rule_phenomenon = str(rule_pred.get("phenomenon", "none"))

    # 2. Gather LLM predictions if available
    report_preds = predictions.get(report_id, {})

    # Collect all predicted phenomena from different methods
    method_phenomena: dict[str, str] = {"rule": rule_phenomenon}
    for method_key in ("direct", "cot", "react", "astra"):
        pred = report_preds.get(method_key, {})
        phen = str(pred.get("phenomenon", "none"))
        method_phenomena[method_key] = phen

    # 3. Count how many methods predict each non-none phenomenon
    phenomenon_votes: Counter[str] = Counter()
    for _method, phen in method_phenomena.items():
        if phen != "none":
            phenomenon_votes[phen] += 1

    # 4. Weighted score from prediction votes
    vote_score = 0.0
    for phen, count in phenomenon_votes.items():
        weight = PHENOMENON_WEIGHT.get(phen, 1.0)
        vote_score += weight * count

    # 5. Lexicon heuristic bonus
    lexicon_bonus, lexicon_phenomenon = _compute_lexicon_heuristic_score(title, summary)
    vote_score += lexicon_bonus
    if lexicon_phenomenon != "none" and lexicon_phenomenon not in method_phenomena.values():
        method_phenomena["lexicon"] = lexicon_phenomenon

    # Collect unique non-none predicted phenomena for output
    predicted_phenomena = sorted(
        {phen for phen in method_phenomena.values() if phen != "none"}
    )

    return (round(vote_score, 4), predicted_phenomena)


def _build_split_lookup(
    dev_ids_path: Path,
    test_ids_path: Path,
) -> dict[str, str]:
    """Build a report_id -> target_split mapping from split ID files."""
    lookup: dict[str, str] = {}
    for rid in _load_json(dev_ids_path):
        lookup[str(rid)] = "dev"
    for rid in _load_json(test_ids_path):
        lookup[str(rid)] = "test"
    return lookup


def _load_already_annotated_ids(path: Path) -> set[str]:
    """Load report IDs that are already annotated."""
    return {
        str(row.get("report_id"))
        for row in _load_jsonl(path)
        if row.get("report_id")
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_priority_queue(
    *,
    gold_workset_path: Path = GOLD_WORKSET_PATH,
    predictions_dir: Path = PREDICTIONS_DIR,
    dev_ids_path: Path = DEV_IDS_PATH,
    test_ids_path: Path = TEST_IDS_PATH,
    already_annotated_path: Path = ALREADY_ANNOTATED_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Build the annotation priority queue and write it to output_path.

    Returns the list of priority-ranked rows.
    """
    # Load data
    gold_workset = _load_jsonl(gold_workset_path)
    predictions = _load_predictions(predictions_dir)
    split_lookup = _build_split_lookup(dev_ids_path, test_ids_path)
    already_annotated = _load_already_annotated_ids(already_annotated_path)

    # Build queue
    queue: list[dict[str, Any]] = []
    for row in gold_workset:
        report_id = str(row.get("report_id", ""))
        if not report_id:
            continue

        # Exclude already-annotated reports
        if report_id in already_annotated:
            continue

        title = row.get("title", "")
        summary = row.get("summary", "")
        stock_code = row.get("stock_code", "")
        report_date = row.get("report_date", "")

        priority_score, predicted_phenomena = _compute_priority_score(row, predictions)
        target_split = split_lookup.get(report_id, row.get("split", "train"))

        queue.append({
            "report_id": report_id,
            "title": title,
            "summary": summary,
            "stock_code": stock_code,
            "report_date": report_date,
            "target_split": target_split,
            "priority_score": priority_score,
            "predicted_phenomena": predicted_phenomena,
            "annotation_template": {
                "fundamental_sentiment": None,
                "strategic_optimism": None,
                "phenomenon": None,
                "evidence_spans": [],
                "annotation_confidence": None,
                "notes": "",
            },
        })

    # Sort by priority score descending
    queue.sort(key=lambda r: r["priority_score"], reverse=True)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for entry in queue:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Print summary
    _print_summary(queue)

    return queue


def _print_summary(queue: list[dict[str, Any]]) -> None:
    """Print summary statistics about the priority queue."""
    total = len(queue)
    split_counter: Counter[str] = Counter()
    phenomenon_counter: Counter[str] = Counter()

    for entry in queue:
        split_counter[entry["target_split"]] += 1
        for phen in entry["predicted_phenomena"]:
            phenomenon_counter[phen] += 1

    # Count rows with no predicted phenomena
    no_phenomenon_count = sum(
        1 for entry in queue if not entry["predicted_phenomena"]
    )
    phenomenon_counter["none (no prediction)"] = no_phenomenon_count

    print(f"\n{'=' * 60}")
    print(f"Annotation Priority Queue Summary")
    print(f"{'=' * 60}")
    print(f"Total rows in queue: {total}")
    print()
    print("Rows per target split:")
    for split in sorted(split_counter):
        print(f"  {split}: {split_counter[split]}")
    print()
    print("Predicted phenomenon distribution:")
    for phen, count in phenomenon_counter.most_common():
        print(f"  {phen}: {count}")
    print()

    if total > 0:
        top_score = queue[0]["priority_score"]
        bottom_score = queue[-1]["priority_score"]
        print(f"Score range: {bottom_score:.4f} - {top_score:.4f}")

    print(f"{'=' * 60}\n")


def main() -> None:
    build_priority_queue()


if __name__ == "__main__":
    main()
