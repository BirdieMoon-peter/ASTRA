from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from astra.config.task_schema import CALIBRATION_PATH, NLP_METRICS_PATH, PHENOMENA_METRICS_PATH


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _macro_f1(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], label_key: str) -> float:
    gold_by_id = {row["report_id"]: row for row in gold_rows}
    gold_labels = []
    pred_labels = []
    labels = set()
    for pred in pred_rows:
        gold = gold_by_id.get(pred["report_id"])
        if not gold:
            continue
        gold_label = (gold.get("annotation") or {}).get(label_key)
        pred_label = pred.get(label_key)
        if gold_label is None or pred_label is None:
            continue
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        labels.add(gold_label)
        labels.add(pred_label)
    if not gold_labels:
        return 0.0
    f1_scores = []
    for label in sorted(labels):
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p != label)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(round(2 * precision * recall / (precision + recall), 4))
    return round(sum(f1_scores) / len(f1_scores), 4)


def _span_set(spans: list[Any]) -> set[str]:
    normalized: set[str] = set()
    for span in spans:
        if isinstance(span, str):
            text = span.strip()
        elif isinstance(span, dict):
            text = str(span.get("text") or "").strip()
        else:
            text = str(span).strip()
        if text:
            normalized.add(text)
    return normalized


def evidence_span_prf(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, float]:
    gold_by_id = {row["report_id"]: row for row in gold_rows}
    tp = fp = fn = 0
    for pred in pred_rows:
        gold = gold_by_id.get(pred["report_id"])
        if not gold:
            continue
        gold_spans = _span_set((gold.get("annotation") or {}).get("evidence_spans", []))
        pred_spans = _span_set(pred.get("evidence_spans", []))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)
    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
    }


def build_calibration_bins(pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_by_id = {row["report_id"]: row for row in gold_rows}
    bins: dict[int, list[int]] = defaultdict(list)
    for pred in pred_rows:
        gold = gold_by_id.get(pred["report_id"])
        if not gold:
            continue
        gold_label = (gold.get("annotation") or {}).get("fundamental_sentiment")
        if gold_label is None:
            continue
        confidence = 1.0 - float(pred.get("uncertainty", 0.5))
        bucket = min(9, max(0, int(confidence * 10)))
        bins[bucket].append(1 if pred.get("fundamental_sentiment") == gold_label else 0)
    results = []
    for bucket in sorted(bins):
        values = bins[bucket]
        center = round((bucket + 0.5) / 10, 2)
        results.append({"bin": center, "count": len(values), "accuracy": round(sum(values) / len(values), 4)})
    return results


def expected_calibration_error(calibration_bins: list[dict[str, Any]]) -> float:
    total = sum(int(item["count"]) for item in calibration_bins)
    if total == 0:
        return 0.0
    weighted_error = 0.0
    for item in calibration_bins:
        confidence = float(item["bin"])
        accuracy = float(item["accuracy"])
        count = int(item["count"])
        weighted_error += abs(confidence - accuracy) * count
    return round(weighted_error / total, 4)


def build_phenomena_metrics(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    gold_by_id = {row["report_id"]: row for row in gold_rows}
    per_label_total = Counter()
    per_label_match = Counter()
    for pred in pred_rows:
        gold = gold_by_id.get(pred["report_id"])
        if not gold:
            continue
        gold_label = (gold.get("annotation") or {}).get("phenomenon")
        if gold_label is None:
            continue
        per_label_total[gold_label] += 1
        if pred.get("phenomenon") == gold_label:
            per_label_match[gold_label] += 1
    details = {
        label: {
            "accuracy": _safe_ratio(per_label_match[label], total),
            "count": total,
        }
        for label, total in per_label_total.items()
    }
    macro = round(sum(item["accuracy"] for item in details.values()) / len(details), 4) if details else 0.0
    return {"per_label": details, "macro_accuracy": macro}


def evaluate_predictions(
    gold_path: Path,
    prediction_paths: dict[str, Path],
    *,
    metrics_output_path: Path = NLP_METRICS_PATH,
    calibration_output_path: Path = CALIBRATION_PATH,
    phenomena_output_path: Path = PHENOMENA_METRICS_PATH,
) -> dict[str, Any]:
    if not gold_path.exists():
        return {
            "status": "missing_gold_annotations",
            "gold_path": str(gold_path),
            "available_prediction_paths": {key: str(value) for key, value in prediction_paths.items()},
        }

    gold_rows = _load_jsonl(gold_path)
    gold_ids = {row.get("report_id") for row in gold_rows}
    metrics = {}
    calibration = {}
    phenomena = {}
    overlap_by_method: dict[str, int] = {}
    for method, path in prediction_paths.items():
        if not path.exists():
            continue
        pred_rows = _load_jsonl(path)
        overlap_count = sum(1 for row in pred_rows if row.get("report_id") in gold_ids)
        overlap_by_method[method] = overlap_count
        span_metrics = evidence_span_prf(gold_rows, pred_rows)
        calibration_bins = build_calibration_bins(pred_rows, gold_rows)
        metrics[method] = {
            "fundamental_sentiment_macro_f1": _macro_f1(gold_rows, pred_rows, "fundamental_sentiment"),
            "strategic_optimism_macro_f1": _macro_f1(gold_rows, pred_rows, "strategic_optimism"),
            "phenomenon_macro_f1": _macro_f1(gold_rows, pred_rows, "phenomenon"),
            **span_metrics,
            "ece": expected_calibration_error(calibration_bins),
            "prediction_count": len(pred_rows),
            "matched_gold_count": overlap_count,
        }
        calibration[method] = calibration_bins
        phenomena[method] = build_phenomena_metrics(gold_rows, pred_rows)

    if overlap_by_method and all(count == 0 for count in overlap_by_method.values()):
        return {
            "status": "no_overlapping_report_ids",
            "gold_path": str(gold_path),
            "gold_count": len(gold_rows),
            "prediction_paths": {key: str(value) for key, value in prediction_paths.items()},
            "overlap_by_method": overlap_by_method,
        }

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with calibration_output_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, ensure_ascii=False, indent=2)
    with phenomena_output_path.open("w", encoding="utf-8") as handle:
        json.dump(phenomena, handle, ensure_ascii=False, indent=2)
    return {"metrics": metrics, "calibration": calibration, "phenomena": phenomena}


def continuous_score_deltas(
    full_predictions: list[dict[str, Any]],
    ablation_predictions: list[dict[str, Any]],
    score_keys: tuple[str, ...] = (
        "strategic_optimism_gap",
        "astra_composite_score",
        "astra_uncertainty_gated_score",
        "uncertainty",
        "penalty_confidence",
    ),
) -> dict[str, float]:
    """Compute mean absolute delta between full model and ablation continuous scores."""
    full_by_id = {row["report_id"]: row for row in full_predictions}
    deltas: dict[str, list[float]] = {key: [] for key in score_keys}
    for ablation in ablation_predictions:
        full = full_by_id.get(ablation["report_id"])
        if not full:
            continue
        for key in score_keys:
            full_val = float(full.get(key, 0.0) or 0.0)
            abl_val = float(ablation.get(key, 0.0) or 0.0)
            deltas[key].append(abs(full_val - abl_val))
    return {
        f"mean_abs_delta_{key}": round(sum(vals) / len(vals), 4) if vals else 0.0
        for key, vals in deltas.items()
    }


def label_distribution_diff(
    full_predictions: list[dict[str, Any]],
    ablation_predictions: list[dict[str, Any]],
    label_keys: tuple[str, ...] = ("fundamental_sentiment", "strategic_optimism", "phenomenon"),
) -> dict[str, Any]:
    """Compare categorical label distributions between full model and ablation."""
    full_by_id = {row["report_id"]: row for row in full_predictions}
    results: dict[str, Any] = {}
    for key in label_keys:
        flipped = 0
        matched = 0
        for ablation in ablation_predictions:
            full = full_by_id.get(ablation["report_id"])
            if not full:
                continue
            if full.get(key) != ablation.get(key):
                flipped += 1
            else:
                matched += 1
        total = flipped + matched
        results[key] = {
            "flipped_count": flipped,
            "total": total,
            "flip_rate": round(flipped / total, 4) if total > 0 else 0.0,
        }
    return results


def evaluate_split(
    gold_path: Path,
    prediction_paths: dict[str, Path],
    split: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate predictions filtered to a specific split."""
    if not gold_path.exists():
        return {"status": "missing_gold_annotations", "gold_path": str(gold_path)}

    gold_rows = _load_jsonl(gold_path)
    gold_rows = [row for row in gold_rows if row.get("split") == split]
    if not gold_rows:
        return {"status": "no_gold_rows_for_split", "split": split}

    filtered_paths: dict[str, Path] = {}
    for method, path in prediction_paths.items():
        if not path.exists():
            continue
        pred_rows = _load_jsonl(path)
        pred_rows = [row for row in pred_rows if row.get("split") == split]
        if pred_rows:
            tmp_path = path.parent / f"_tmp_split_{split}_{path.name}"
            with tmp_path.open("w", encoding="utf-8") as handle:
                for row in pred_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            filtered_paths[method] = tmp_path

    tmp_gold = gold_path.parent / f"_tmp_split_{split}_gold.jsonl"
    with tmp_gold.open("w", encoding="utf-8") as handle:
        for row in gold_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = evaluate_predictions(tmp_gold, filtered_paths, **kwargs)

    tmp_gold.unlink(missing_ok=True)
    for tmp_path in filtered_paths.values():
        tmp_path.unlink(missing_ok=True)

    return result
