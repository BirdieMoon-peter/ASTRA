"""Error analysis: per-phenomenon confusion matrices, FP/FN analysis, patterns."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL file, one JSON object per line."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _gold_label(row: dict[str, Any], key: str) -> Any:
    """Extract a label from a gold row (nested under ``annotation``)."""
    ann = row.get("annotation")
    if isinstance(ann, dict):
        return ann.get(key)
    return row.get(key)


def _pred_label(row: dict[str, Any], key: str) -> Any:
    """Extract a label from a prediction row (flat)."""
    return row.get(key)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round(num / den, 4)


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------


def build_confusion_matrix(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    label_key: str = "phenomenon",
) -> dict[str, Any]:
    """Build a full confusion matrix with per-label precision / recall / F1.

    Parameters
    ----------
    gold_rows:
        Gold annotation rows (JSONL with nested ``annotation`` dict).
    pred_rows:
        Prediction rows (JSONL with flat fields).
    label_key:
        Which label field to evaluate.

    Returns
    -------
    dict with keys ``matrix`` (nested dict ``{gold: {pred: count}}``),
    ``labels`` (sorted unique labels), and ``per_label`` metrics.
    """
    gold_by_id: dict[str, dict] = {r["report_id"]: r for r in gold_rows}
    pred_by_id: dict[str, dict] = {r["report_id"]: r for r in pred_rows}
    common_ids = sorted(set(gold_by_id) & set(pred_by_id))

    # Collect label pairs.
    pairs: list[tuple[str, str]] = []
    for rid in common_ids:
        g = _gold_label(gold_by_id[rid], label_key)
        p = _pred_label(pred_by_id[rid], label_key)
        if g is None or p is None:
            continue
        pairs.append((str(g), str(p)))

    labels = sorted({lab for pair in pairs for lab in pair})

    # Build matrix.
    matrix: dict[str, dict[str, int]] = {
        gl: {pl: 0 for pl in labels} for gl in labels
    }
    for g, p in pairs:
        matrix[g][p] += 1

    # Per-label precision / recall / F1.
    per_label: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = matrix.get(lab, {}).get(lab, 0)
        fp = sum(matrix.get(other, {}).get(lab, 0) for other in labels if other != lab)
        fn = sum(matrix.get(lab, {}).get(other, 0) for other in labels if other != lab)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            0.0
            if (precision + recall) == 0
            else round(2 * precision * recall / (precision + recall), 4)
        )
        per_label[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return {
        "matrix": matrix,
        "labels": labels,
        "per_label": per_label,
        "total_evaluated": len(pairs),
    }


# ---------------------------------------------------------------------------
# 2. Classify errors
# ---------------------------------------------------------------------------


def classify_errors(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict]:
    """Classify every mismatched prediction.

    Returns a list of error dicts with ``report_id``, ``gold_label``,
    ``pred_label``, ``error_type``, gold/predicted fields, and the
    report ``title`` and ``summary`` for manual review.
    """
    gold_by_id: dict[str, dict] = {r["report_id"]: r for r in gold_rows}
    pred_by_id: dict[str, dict] = {r["report_id"]: r for r in pred_rows}
    common_ids = sorted(set(gold_by_id) & set(pred_by_id))

    label_keys = ["fundamental_sentiment", "strategic_optimism", "phenomenon"]
    errors: list[dict] = []

    for rid in common_ids:
        gold = gold_by_id[rid]
        pred = pred_by_id[rid]

        for key in label_keys:
            g = _gold_label(gold, key)
            p = _pred_label(pred, key)
            if g is None or p is None:
                continue
            g_str, p_str = str(g), str(p)
            if g_str == p_str:
                continue

            # Determine error type.
            if key == "phenomenon":
                if g_str == "none" and p_str != "none":
                    error_type = "false_positive"
                elif g_str != "none" and p_str == "none":
                    error_type = "false_negative"
                else:
                    error_type = "misclassification"
            else:
                # For sentiment / optimism, treat as misclassification.
                error_type = "misclassification"

            gold_ann = gold.get("annotation") or {}
            errors.append({
                "report_id": rid,
                "label_key": key,
                "gold_label": g_str,
                "pred_label": p_str,
                "error_type": error_type,
                "gold_annotation": {
                    "fundamental_sentiment": gold_ann.get("fundamental_sentiment"),
                    "strategic_optimism": gold_ann.get("strategic_optimism"),
                    "phenomenon": gold_ann.get("phenomenon"),
                    "annotation_confidence": gold_ann.get("annotation_confidence"),
                },
                "predicted_fields": {
                    "fundamental_sentiment": pred.get("fundamental_sentiment"),
                    "strategic_optimism": pred.get("strategic_optimism"),
                    "phenomenon": pred.get("phenomenon"),
                    "uncertainty": pred.get("uncertainty"),
                },
                "title": gold.get("title", ""),
                "summary": gold.get("summary", "")[:300],
            })

    return errors


# ---------------------------------------------------------------------------
# 3. Error-confidence correlation
# ---------------------------------------------------------------------------


def error_confidence_correlation(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict:
    """Group predictions by ``annotation_confidence`` and compute accuracy.

    Returns per-confidence-level accuracy and overall correlation statistics.
    """
    gold_by_id: dict[str, dict] = {r["report_id"]: r for r in gold_rows}
    pred_by_id: dict[str, dict] = {r["report_id"]: r for r in pred_rows}
    common_ids = sorted(set(gold_by_id) & set(pred_by_id))

    conf_levels = ["low", "medium", "high"]
    conf_correct: dict[str, int] = defaultdict(int)
    conf_total: dict[str, int] = defaultdict(int)
    # For correlation: map confidence to numeric, compute with accuracy.
    conf_numeric = {"low": 1, "medium": 2, "high": 3}
    xs: list[float] = []
    ys: list[float] = []

    label_keys = ["fundamental_sentiment", "strategic_optimism", "phenomenon"]

    for rid in common_ids:
        gold = gold_by_id[rid]
        pred = pred_by_id[rid]
        conf = _gold_label(gold, "annotation_confidence")
        if conf is None:
            conf = "medium"  # fallback
        conf_str = str(conf).strip().lower()
        if conf_str not in conf_numeric:
            conf_str = "medium"

        # Check if all labels match.
        all_correct = True
        any_evaluated = False
        for key in label_keys:
            g = _gold_label(gold, key)
            p = _pred_label(pred, key)
            if g is None or p is None:
                continue
            any_evaluated = True
            if str(g) != str(p):
                all_correct = False

        if not any_evaluated:
            continue

        conf_total[conf_str] += 1
        correct_int = 1 if all_correct else 0
        conf_correct[conf_str] += correct_int
        xs.append(float(conf_numeric[conf_str]))
        ys.append(float(correct_int))

    per_confidence: dict[str, dict[str, Any]] = {}
    for level in conf_levels:
        total = conf_total.get(level, 0)
        correct = conf_correct.get(level, 0)
        per_confidence[level] = {
            "total": total,
            "correct": correct,
            "accuracy": _safe_div(correct, total),
        }

    # Pearson correlation between confidence level and correctness.
    pearson_r = 0.0
    if len(xs) >= 2:
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(var_x * var_y)
        if denom > 0:
            pearson_r = round(cov / denom, 4)

    return {
        "per_confidence": per_confidence,
        "pearson_r_confidence_accuracy": pearson_r,
        "total_evaluated": len(xs),
    }


# ---------------------------------------------------------------------------
# 4. Per-phenomenon error summary
# ---------------------------------------------------------------------------


def phenomenon_error_summary(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> dict:
    """Per-phenomenon summary: most common confusion, example IDs, FP/FN rates.

    Also returns macro-averaged precision / recall / F1 for phenomenon
    detection.
    """
    cm = build_confusion_matrix(gold_rows, pred_rows, label_key="phenomenon")
    matrix = cm["matrix"]
    labels = cm["labels"]
    per_label = cm["per_label"]

    gold_by_id: dict[str, dict] = {r["report_id"]: r for r in gold_rows}
    pred_by_id: dict[str, dict] = {r["report_id"]: r for r in pred_rows}
    common_ids = sorted(set(gold_by_id) & set(pred_by_id))

    # Per-phenomenon detail.
    per_phenomenon: dict[str, dict[str, Any]] = {}
    for lab in labels:
        # Most common confusion pair: the label most often confused *with*.
        row = matrix.get(lab, {})
        confusions = {k: v for k, v in row.items() if k != lab and v > 0}
        most_common_confusion = (
            max(confusions, key=confusions.get)  # type: ignore[arg-type]
            if confusions
            else None
        )

        # Example report IDs for errors involving this phenomenon.
        example_ids: list[str] = []
        for rid in common_ids:
            g = str(_gold_label(gold_by_id[rid], "phenomenon") or "")
            p = str(_pred_label(pred_by_id[rid], "phenomenon") or "")
            if (g == lab or p == lab) and g != p:
                example_ids.append(rid)
                if len(example_ids) >= 5:
                    break

        # FP / FN rates.
        tp = row.get(lab, 0)
        fn = sum(v for k, v in row.items() if k != lab)
        fp = sum(matrix.get(other, {}).get(lab, 0) for other in labels if other != lab)
        support = tp + fn

        per_phenomenon[lab] = {
            "most_common_confusion": most_common_confusion,
            "example_error_report_ids": example_ids,
            "fp_rate": _safe_div(fp, fp + tp) if (fp + tp) > 0 else 0.0,
            "fn_rate": _safe_div(fn, support) if support > 0 else 0.0,
            "precision": per_label.get(lab, {}).get("precision", 0.0),
            "recall": per_label.get(lab, {}).get("recall", 0.0),
            "f1": per_label.get(lab, {}).get("f1", 0.0),
            "support": support,
        }

    # Macro-averaged P / R / F1.
    if per_label:
        macro_precision = round(
            sum(v["precision"] for v in per_label.values()) / len(per_label), 4
        )
        macro_recall = round(
            sum(v["recall"] for v in per_label.values()) / len(per_label), 4
        )
        macro_f1 = (
            0.0
            if (macro_precision + macro_recall) == 0
            else round(
                2 * macro_precision * macro_recall / (macro_precision + macro_recall),
                4,
            )
        )
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    return {
        "per_phenomenon": per_phenomenon,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix,
        "labels": labels,
        "total_evaluated": cm["total_evaluated"],
    }


# ---------------------------------------------------------------------------
# 5. Export CSV error table
# ---------------------------------------------------------------------------


def export_error_table(summary: dict, output_path: Path | str) -> None:
    """Write a CSV table suitable for paper inclusion.

    Columns: phenomenon, precision, recall, F1, FP_rate, FN_rate,
    support, most_common_confusion.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_phenom = summary.get("per_phenomenon", {})

    fieldnames = [
        "phenomenon",
        "precision",
        "recall",
        "f1",
        "fp_rate",
        "fn_rate",
        "support",
        "most_common_confusion",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for phenom in sorted(per_phenom.keys()):
            stats = per_phenom[phenom]
            writer.writerow({
                "phenomenon": phenom,
                "precision": stats.get("precision", 0.0),
                "recall": stats.get("recall", 0.0),
                "f1": stats.get("f1", 0.0),
                "fp_rate": stats.get("fp_rate", 0.0),
                "fn_rate": stats.get("fn_rate", 0.0),
                "support": stats.get("support", 0),
                "most_common_confusion": stats.get("most_common_confusion", ""),
            })

        # Macro row.
        writer.writerow({
            "phenomenon": "MACRO",
            "precision": summary.get("macro_precision", 0.0),
            "recall": summary.get("macro_recall", 0.0),
            "f1": summary.get("macro_f1", 0.0),
            "fp_rate": "",
            "fn_rate": "",
            "support": summary.get("total_evaluated", 0),
            "most_common_confusion": "",
        })
