"""Human evaluation protocol for factual preservation, faithfulness, and tone removal."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
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


def _get_annotation_field(row: dict[str, Any], field: str) -> Any:
    """Read a field from either a nested ``annotation`` dict (gold) or a flat row (pred)."""
    ann = row.get("annotation")
    if isinstance(ann, dict) and field in ann:
        return ann[field]
    return row.get(field)


def _normalize_spans(value: Any) -> list[dict[str, Any]]:
    """Normalise evidence_spans to a list of dicts with at least a ``text`` key."""
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item.strip():
            result.append({"text": item.strip()})
    return result


# ---------------------------------------------------------------------------
# 1. Sample selection
# ---------------------------------------------------------------------------

PHENOMENA = [
    "hedged_downside",
    "euphemistic_risk",
    "title_body_mismatch",
    "omitted_downside_context",
    "none",
]


def select_eval_sample(
    predictions_path: str | Path,
    gold_path: str | Path,
    n: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Select a stratified evaluation sample from ASTRA predictions and gold annotations.

    Ensures each non-``none`` phenomenon gets at least 5 examples when available,
    then fills the remainder proportionally across all phenomena.

    Parameters
    ----------
    predictions_path:
        JSONL of ASTRA predictions (flat fields).
    gold_path:
        JSONL of gold annotations (with nested ``annotation`` dict).
    n:
        Total number of examples to select.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Each dict contains ``report_id``, ``title``, ``summary``,
        ``gold_annotation``, ``astra_prediction``, ``neutralized_text``,
        ``removed_rhetoric``, and ``evidence_spans``.
    """
    predictions_path = Path(predictions_path)
    gold_path = Path(gold_path)

    preds_raw = _load_jsonl(predictions_path)
    golds_raw = _load_jsonl(gold_path)

    preds_by_id: dict[str, dict] = {r["report_id"]: r for r in preds_raw}
    golds_by_id: dict[str, dict] = {r["report_id"]: r for r in golds_raw}

    # Only keep IDs that appear in both files.
    common_ids = sorted(set(preds_by_id) & set(golds_by_id))

    # Group by gold phenomenon.
    by_phenomenon: dict[str, list[str]] = defaultdict(list)
    for rid in common_ids:
        gold = golds_by_id[rid]
        phenomenon = _get_annotation_field(gold, "phenomenon") or "none"
        by_phenomenon[phenomenon].append(rid)

    rng = random.Random(seed)
    for ids in by_phenomenon.values():
        rng.shuffle(ids)

    selected_ids: list[str] = []
    remaining_budget = n

    # Phase 1 -- guarantee minimum per non-none phenomenon.
    min_per_phenomenon = 5
    non_none_phenomena = [p for p in PHENOMENA if p != "none"]
    for phenom in non_none_phenomena:
        available = by_phenomenon.get(phenom, [])
        take = min(min_per_phenomenon, len(available), remaining_budget)
        selected_ids.extend(available[:take])
        by_phenomenon[phenom] = available[take:]
        remaining_budget -= take
        if remaining_budget <= 0:
            break

    # Phase 2 -- fill the rest proportionally from all phenomena.
    if remaining_budget > 0:
        pool: list[str] = []
        for phenom in PHENOMENA:
            pool.extend(by_phenomenon.get(phenom, []))
        # Also include any phenomena not in the canonical list.
        for phenom, ids in by_phenomenon.items():
            if phenom not in PHENOMENA:
                pool.extend(ids)
        # Remove already-selected.
        already = set(selected_ids)
        pool = [rid for rid in pool if rid not in already]
        rng.shuffle(pool)
        selected_ids.extend(pool[:remaining_budget])

    # Build output records.
    sample: list[dict] = []
    for rid in selected_ids:
        gold = golds_by_id[rid]
        pred = preds_by_id[rid]
        gold_ann = gold.get("annotation") or {}

        sample.append({
            "report_id": rid,
            "title": gold.get("title", ""),
            "summary": gold.get("summary", ""),
            "gold_annotation": {
                "fundamental_sentiment": gold_ann.get("fundamental_sentiment"),
                "strategic_optimism": gold_ann.get("strategic_optimism"),
                "phenomenon": gold_ann.get("phenomenon"),
                "annotation_confidence": gold_ann.get("annotation_confidence"),
                "evidence_spans": _normalize_spans(gold_ann.get("evidence_spans", [])),
            },
            "astra_prediction": {
                "fundamental_sentiment": pred.get("fundamental_sentiment"),
                "strategic_optimism": pred.get("strategic_optimism"),
                "phenomenon": pred.get("phenomenon"),
            },
            "neutralized_text": pred.get("neutralized_text", ""),
            "removed_rhetoric": (
                pred.get("neutralization", {}).get("removed_rhetoric")
                or pred.get("removed_rhetoric")
                or []
            ),
            "evidence_spans": _normalize_spans(pred.get("evidence_spans", [])),
        })

    return sample


# ---------------------------------------------------------------------------
# 2. Prepare evaluation packets
# ---------------------------------------------------------------------------


def prepare_eval_packet(sample: list[dict]) -> list[dict]:
    """Create evaluation packets for human annotators.

    Each packet contains the original and neutralized texts together with
    empty evaluation fields for the annotator to fill in.
    """
    packets: list[dict] = []
    for item in sample:
        title = item.get("title", "")
        summary = item.get("summary", "")
        original = f"{title}\n{summary}".strip()

        packets.append({
            "report_id": item["report_id"],
            "original_text": original,
            "neutralized_text": item.get("neutralized_text", ""),
            "removed_rhetoric": item.get("removed_rhetoric", []),
            "gold_phenomenon": (
                item.get("gold_annotation", {}).get("phenomenon", "")
            ),
            "pred_phenomenon": (
                item.get("astra_prediction", {}).get("phenomenon", "")
            ),
            "eval_dimensions": {
                "factual_preservation": None,   # "yes" or "no"
                "faithfulness": None,           # 1-5 Likert
                "tone_removal": None,           # 1-5 Likert
            },
            "evaluator_notes": "",
        })

    return packets


# ---------------------------------------------------------------------------
# 3. Export evaluation batch
# ---------------------------------------------------------------------------


def export_eval_batch(packets: list[dict], output_path: Path | str) -> None:
    """Write evaluation packets as JSONL for human annotators.

    All evaluation fields are left empty (``None`` / ``""``) so annotators
    can fill them in.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for packet in packets:
            fh.write(json.dumps(packet, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 4. Aggregate completed evaluations
# ---------------------------------------------------------------------------


def _krippendorff_alpha_ordinal(ratings_by_item: list[list[int | None]]) -> float:
    """Krippendorff's alpha for ordinal data (Likert scales).

    Falls back to nominal distance (0/1) but uses squared-difference for
    ordinal data to weight larger disagreements more heavily.

    Parameters
    ----------
    ratings_by_item:
        ``ratings_by_item[i]`` is a list of ratings (or ``None`` for
        missing) from each evaluator for item *i*.
    """
    if not ratings_by_item:
        return 0.0

    # Build coincidence matrix.
    coincidences: dict[tuple[int, int], float] = Counter()
    total_pairable = 0.0

    for row in ratings_by_item:
        values = [v for v in row if v is not None]
        m = len(values)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for a, b in combinations(values, 2):
            coincidences[(a, b)] += weight
            coincidences[(b, a)] += weight
        val_counts = Counter(values)
        for v, cnt in val_counts.items():
            coincidences[(v, v)] += cnt * (cnt - 1) * weight
        total_pairable += m

    if total_pairable < 2:
        return 0.0

    categories = sorted({
        v for row in ratings_by_item for v in row if v is not None
    })
    n_c: dict[int, float] = {}
    for c in categories:
        n_c[c] = sum(coincidences.get((c, k), 0.0) for k in categories)

    n_total = sum(n_c.values())
    if math.isclose(n_total, 0.0):
        return 0.0

    # Ordinal distance: squared difference.
    def _dist(a: int, b: int) -> float:
        return float((a - b) ** 2)

    d_o = sum(
        coincidences.get((c, k), 0.0) * _dist(c, k)
        for c in categories
        for k in categories
    ) / n_total

    d_e = sum(
        n_c.get(c, 0.0) * n_c.get(k, 0.0) * _dist(c, k)
        for c in categories
        for k in categories
    ) / (n_total ** 2)

    if math.isclose(d_e, 0.0):
        return 1.0 if math.isclose(d_o, 0.0) else 0.0
    return round(1.0 - d_o / d_e, 4)


def aggregate_eval_results(completed_paths: list[Path | str]) -> dict:
    """Aggregate completed human evaluations from multiple evaluators.

    Parameters
    ----------
    completed_paths:
        List of JSONL file paths, each from a different evaluator.  Each
        line has the same schema as ``prepare_eval_packet`` output but
        with ``eval_dimensions`` filled in.

    Returns
    -------
    dict
        Summary with factual-preservation rate, mean faithfulness /
        tone-removal scores (with std), Krippendorff's alpha for Likert
        dimensions, and per-phenomenon breakdowns.
    """
    completed_paths = [Path(p) for p in completed_paths]

    # evaluator_idx -> report_id -> eval record
    evals_per_evaluator: list[dict[str, dict]] = []
    for path in completed_paths:
        rows = _load_jsonl(path)
        by_id = {r["report_id"]: r for r in rows}
        evals_per_evaluator.append(by_id)

    # Collect all report_ids across evaluators.
    all_ids: set[str] = set()
    for by_id in evals_per_evaluator:
        all_ids.update(by_id.keys())
    all_ids_sorted = sorted(all_ids)

    # ---- Factual preservation rate ----
    factual_yes = 0
    factual_total = 0
    for rid in all_ids_sorted:
        for by_id in evals_per_evaluator:
            rec = by_id.get(rid)
            if rec is None:
                continue
            dims = rec.get("eval_dimensions") or {}
            fp_val = dims.get("factual_preservation")
            if fp_val is not None:
                factual_total += 1
                if str(fp_val).strip().lower() == "yes":
                    factual_yes += 1

    factual_preservation_rate = (
        round(factual_yes / factual_total, 4) if factual_total > 0 else 0.0
    )

    # ---- Faithfulness & tone removal scores ----
    faithfulness_scores: list[float] = []
    tone_removal_scores: list[float] = []
    for rid in all_ids_sorted:
        for by_id in evals_per_evaluator:
            rec = by_id.get(rid)
            if rec is None:
                continue
            dims = rec.get("eval_dimensions") or {}
            faith = dims.get("faithfulness")
            tone = dims.get("tone_removal")
            if faith is not None:
                try:
                    faithfulness_scores.append(float(faith))
                except (ValueError, TypeError):
                    pass
            if tone is not None:
                try:
                    tone_removal_scores.append(float(tone))
                except (ValueError, TypeError):
                    pass

    def _mean_std(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
            "n": len(values),
        }

    # ---- Krippendorff's alpha for Likert dimensions ----
    def _collect_ratings(dimension: str) -> list[list[int | None]]:
        """Return ratings_by_item for a given Likert dimension."""
        ratings: list[list[int | None]] = []
        for rid in all_ids_sorted:
            row_ratings: list[int | None] = []
            for by_id in evals_per_evaluator:
                rec = by_id.get(rid)
                if rec is None:
                    row_ratings.append(None)
                    continue
                dims = rec.get("eval_dimensions") or {}
                val = dims.get(dimension)
                if val is not None:
                    try:
                        row_ratings.append(int(val))
                    except (ValueError, TypeError):
                        row_ratings.append(None)
                else:
                    row_ratings.append(None)
            ratings.append(row_ratings)
        return ratings

    alpha_faithfulness = _krippendorff_alpha_ordinal(
        _collect_ratings("faithfulness")
    )
    alpha_tone_removal = _krippendorff_alpha_ordinal(
        _collect_ratings("tone_removal")
    )

    # ---- Per-phenomenon breakdown ----
    phenomenon_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "factual_yes": 0,
            "factual_total": 0,
            "faithfulness": [],
            "tone_removal": [],
        }
    )

    for rid in all_ids_sorted:
        # Determine phenomenon from the first evaluator file that has this ID.
        phenomenon = "unknown"
        for by_id in evals_per_evaluator:
            rec = by_id.get(rid)
            if rec is not None:
                phenomenon = rec.get("gold_phenomenon") or rec.get("pred_phenomenon") or "unknown"
                break

        for by_id in evals_per_evaluator:
            rec = by_id.get(rid)
            if rec is None:
                continue
            dims = rec.get("eval_dimensions") or {}

            fp_val = dims.get("factual_preservation")
            if fp_val is not None:
                phenomenon_stats[phenomenon]["factual_total"] += 1
                if str(fp_val).strip().lower() == "yes":
                    phenomenon_stats[phenomenon]["factual_yes"] += 1

            faith = dims.get("faithfulness")
            if faith is not None:
                try:
                    phenomenon_stats[phenomenon]["faithfulness"].append(float(faith))
                except (ValueError, TypeError):
                    pass

            tone = dims.get("tone_removal")
            if tone is not None:
                try:
                    phenomenon_stats[phenomenon]["tone_removal"].append(float(tone))
                except (ValueError, TypeError):
                    pass

    per_phenomenon: dict[str, dict[str, Any]] = {}
    for phenom, stats in sorted(phenomenon_stats.items()):
        per_phenomenon[phenom] = {
            "factual_preservation_rate": (
                round(stats["factual_yes"] / stats["factual_total"], 4)
                if stats["factual_total"] > 0
                else 0.0
            ),
            "faithfulness": _mean_std(stats["faithfulness"]),
            "tone_removal": _mean_std(stats["tone_removal"]),
        }

    return {
        "factual_preservation_rate": factual_preservation_rate,
        "faithfulness": _mean_std(faithfulness_scores),
        "tone_removal": _mean_std(tone_removal_scores),
        "inter_annotator_agreement": {
            "krippendorff_alpha_faithfulness": alpha_faithfulness,
            "krippendorff_alpha_tone_removal": alpha_tone_removal,
        },
        "per_phenomenon": per_phenomenon,
        "evaluator_count": len(evals_per_evaluator),
        "report_count": len(all_ids_sorted),
    }
