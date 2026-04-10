from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from astra.config.task_schema import (
    ANNOTATION_CONFIDENCE_LABELS,
    PHENOMENON_LABELS,
    SENTIMENT_LABELS,
    STRATEGIC_OPTIMISM_LABELS,
)

REQUIRED_ANNOTATION_FIELDS = (
    "fundamental_sentiment",
    "strategic_optimism",
    "phenomenon",
    "evidence_spans",
    "annotation_confidence",
    "notes",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_evidence_spans(text: str, evidence_spans: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, span in enumerate(evidence_spans):
        if not isinstance(span, dict):
            errors.append(f"invalid evidence_spans item at index={index}")
            continue
        start = span.get("start")
        end = span.get("end")
        snippet = span.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            errors.append(f"invalid span boundary at index={index}")
            continue
        if end > len(text):
            errors.append(f"span out of range at index={index}")
            continue
        if snippet is not None and text[start:end] != snippet:
            errors.append(f"span text mismatch at index={index}")
    return errors


def _is_complete_annotation(annotation: dict[str, Any]) -> bool:
    return (
        annotation.get("fundamental_sentiment") is not None
        and annotation.get("strategic_optimism") is not None
        and annotation.get("phenomenon") is not None
        and isinstance(annotation.get("evidence_spans"), list)
        and annotation.get("annotation_confidence") is not None
    )


def validate_annotation_file(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    report_id_counter = Counter(row.get("report_id", "<missing>") for row in rows)
    duplicate_report_ids = sorted(
        report_id
        for report_id, count in report_id_counter.items()
        if count > 1 and report_id != "<missing>"
    )
    errors: list[dict[str, str]] = []
    sentiment_counter = Counter()
    optimism_counter = Counter()
    phenomenon_counter = Counter()
    confidence_counter = Counter()
    incomplete_annotation_count = 0

    for row in rows:
        report_id = row.get("report_id", "<missing>")
        annotation_value = row.get("annotation")
        annotation = annotation_value if isinstance(annotation_value, dict) else {}
        text = f"{row.get('title', '')}\n{row.get('summary', '')}"

        missing_fields = [field for field in REQUIRED_ANNOTATION_FIELDS if field not in annotation]
        if missing_fields:
            errors.append({
                "report_id": report_id,
                "error": f"missing annotation fields: {', '.join(missing_fields)}",
            })

        if not _is_complete_annotation(annotation):
            incomplete_annotation_count += 1

        sentiment = annotation.get("fundamental_sentiment")
        if sentiment is not None:
            if sentiment not in SENTIMENT_LABELS:
                errors.append({"report_id": report_id, "error": f"invalid sentiment: {sentiment}"})
            else:
                sentiment_counter[sentiment] += 1

        optimism = annotation.get("strategic_optimism")
        if optimism is not None:
            if optimism not in STRATEGIC_OPTIMISM_LABELS:
                errors.append({"report_id": report_id, "error": f"invalid strategic_optimism: {optimism}"})
            else:
                optimism_counter[optimism] += 1

        phenomenon = annotation.get("phenomenon")
        if phenomenon is not None:
            if phenomenon not in PHENOMENON_LABELS:
                errors.append({"report_id": report_id, "error": f"invalid phenomenon: {phenomenon}"})
            else:
                phenomenon_counter[phenomenon] += 1

        confidence = annotation.get("annotation_confidence")
        if confidence is not None:
            if confidence not in ANNOTATION_CONFIDENCE_LABELS:
                errors.append({"report_id": report_id, "error": f"invalid annotation_confidence: {confidence}"})
            else:
                confidence_counter[confidence] += 1

        evidence_spans = annotation.get("evidence_spans", [])
        if not isinstance(evidence_spans, list):
            errors.append({"report_id": report_id, "error": "evidence_spans must be a list"})
        else:
            for message in _validate_evidence_spans(text, evidence_spans):
                errors.append({"report_id": report_id, "error": message})

    for report_id in duplicate_report_ids:
        errors.append({"report_id": report_id, "error": "duplicate report_id"})

    return {
        "file": str(path),
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors[:100],
        "duplicate_report_id_count": len(duplicate_report_ids),
        "incomplete_annotation_count": incomplete_annotation_count,
        "ready_to_freeze": len(errors) == 0 and incomplete_annotation_count == 0 and not duplicate_report_ids,
        "sentiment_distribution": dict(sentiment_counter),
        "strategic_optimism_distribution": dict(optimism_counter),
        "phenomenon_distribution": dict(phenomenon_counter),
        "annotation_confidence_distribution": dict(confidence_counter),
    }


_DEFAULT_PHENOMENON_THRESHOLDS: dict[str, int] = {
    "hedged_downside": 30,
    "euphemistic_risk": 15,
    "title_body_mismatch": 15,
    "omitted_downside_context": 10,
}


def check_phenomenon_coverage(
    path: Path,
    *,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Check whether an annotated JSONL file has sufficient annotations per phenomenon.

    Parameters
    ----------
    path:
        Path to an annotated JSONL file where each row has an ``annotation``
        dict containing a ``phenomenon`` field.
    thresholds:
        Minimum required count per phenomenon label.  Defaults to
        ``hedged_downside=30, euphemistic_risk=15, title_body_mismatch=15,
        omitted_downside_context=10``.

    Returns
    -------
    dict with keys:

    - ``counts``: ``dict[str, int]`` — observed count per phenomenon label.
    - ``thresholds``: ``dict[str, int]`` — the thresholds used.
    - ``results``: ``dict[str, dict]`` — per-phenomenon dict with ``count``,
      ``threshold``, and ``pass`` (bool).
    - ``all_pass``: ``bool`` — True iff every phenomenon meets its threshold.
    """
    if thresholds is None:
        thresholds = dict(_DEFAULT_PHENOMENON_THRESHOLDS)

    rows = _load_jsonl(path)
    phenomenon_counter: Counter[str] = Counter()
    for row in rows:
        annotation_value = row.get("annotation")
        annotation = annotation_value if isinstance(annotation_value, dict) else {}
        phenomenon = annotation.get("phenomenon")
        if phenomenon is not None:
            phenomenon_counter[phenomenon] += 1

    results: dict[str, dict[str, Any]] = {}
    all_pass = True
    for label, min_count in thresholds.items():
        observed = phenomenon_counter.get(label, 0)
        passed = observed >= min_count
        if not passed:
            all_pass = False
        results[label] = {
            "count": observed,
            "threshold": min_count,
            "pass": passed,
        }

    return {
        "counts": dict(phenomenon_counter),
        "thresholds": thresholds,
        "results": results,
        "all_pass": all_pass,
    }
