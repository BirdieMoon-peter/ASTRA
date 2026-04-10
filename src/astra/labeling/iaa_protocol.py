"""Inter-annotator agreement (IAA) metrics for ASTRA benchmark annotations.

Computes Cohen's kappa, Fleiss' kappa, Krippendorff's alpha (nominal),
token-overlap F1 for evidence spans, a full IAA report between two annotator
files, and an adjudication helper.

Annotation schema (see data/annotations/annotation_guideline.md):
    fundamental_sentiment : negative | neutral | positive
    strategic_optimism    : low | balanced | high
    phenomenon            : hedged_downside | euphemistic_risk |
                            title_body_mismatch | omitted_downside_context | none
    annotation_confidence : low | medium | high          (ordinal)
    evidence_spans        : list[str]
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

# ── schema constants ──────────────────────────────────────────────────────────

LABEL_FIELDS: list[str] = [
    "fundamental_sentiment",
    "strategic_optimism",
    "phenomenon",
    "annotation_confidence",
]

# ── helpers ───────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, one JSON object per line."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _normalize_spans(value: Any) -> list[str]:
    """Extract a flat list of span texts, handling both dict and str items."""
    spans: list[str] = []
    if not isinstance(value, list):
        return spans
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            spans.append(text)
    return spans


# ── Cohen's kappa ─────────────────────────────────────────────────────────────


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two annotators on a categorical field.

    Parameters
    ----------
    labels_a, labels_b : list[str]
        Parallel lists of categorical labels from annotator A and B.
        Must be the same length.

    Returns
    -------
    float
        Kappa coefficient in [-1, 1].  1 = perfect agreement,
        0 = agreement expected by chance, negative = worse than chance.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length.")
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)

    if math.isclose(1.0 - expected, 0.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return round((observed - expected) / (1.0 - expected), 4)


# ── Fleiss' kappa ─────────────────────────────────────────────────────────────


def fleiss_kappa(annotations: list[list[str]]) -> float:
    """Fleiss' kappa for 3+ annotators.

    Parameters
    ----------
    annotations : list[list[str]]
        annotations[i] is a list of labels from each annotator for item *i*.
        Every inner list must have the same length (number of annotators).

    Returns
    -------
    float
        Fleiss' kappa coefficient.
    """
    if not annotations:
        return 0.0

    n_raters = len(annotations[0])
    if n_raters < 2:
        raise ValueError("Fleiss' kappa requires at least 2 raters.")
    n_items = len(annotations)

    categories = sorted({label for row in annotations for label in row})
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build the n_items x k count matrix.
    matrix = [[0] * k for _ in range(n_items)]
    for i, row in enumerate(annotations):
        if len(row) != n_raters:
            raise ValueError(
                f"Item {i} has {len(row)} raters; expected {n_raters}."
            )
        for label in row:
            matrix[i][cat_idx[label]] += 1

    # Per-item agreement P_i.
    p_items = []
    for row in matrix:
        p_i = (sum(nij * nij for nij in row) - n_raters) / (
            n_raters * (n_raters - 1)
        )
        p_items.append(p_i)

    p_bar = sum(p_items) / n_items

    # Category proportions p_j.
    total_assignments = n_items * n_raters
    p_cats = [
        sum(matrix[i][j] for i in range(n_items)) / total_assignments
        for j in range(k)
    ]
    p_e = sum(p_j * p_j for p_j in p_cats)

    if math.isclose(1.0 - p_e, 0.0):
        return 1.0 if math.isclose(p_bar, 1.0) else 0.0
    return round((p_bar - p_e) / (1.0 - p_e), 4)


# ── Krippendorff's alpha (nominal) ───────────────────────────────────────────


def krippendorff_alpha_nominal(annotations: list[list[str | None]]) -> float:
    """Krippendorff's alpha for nominal data, handling missing values (None).

    Parameters
    ----------
    annotations : list[list[str | None]]
        annotations[i] is a list of labels (or None for missing) from each
        annotator for item *i*.  Inner lists must be the same length.

    Returns
    -------
    float
        Alpha coefficient.  1 = perfect reliability, 0 = chance-level.
    """
    if not annotations:
        return 0.0

    n_coders = len(annotations[0])

    # Build coincidence matrix across all items.
    coincidences: dict[tuple[str, str], float] = Counter()
    total_pairable = 0.0

    for row in annotations:
        # Collect non-missing values for this item.
        values = [v for v in row if v is not None]
        m = len(values)
        if m < 2:
            continue
        # Each pair contributes 1/(m-1) to the coincidence matrix.
        weight = 1.0 / (m - 1)
        for a, b in combinations(values, 2):
            coincidences[(a, b)] += weight
            coincidences[(b, a)] += weight
        # Diagonal: each value paired with itself across other raters.
        val_counts = Counter(values)
        for v, cnt in val_counts.items():
            coincidences[(v, v)] += cnt * (cnt - 1) * weight
        total_pairable += m

    if total_pairable < 2:
        return 0.0

    # Marginals: n_c = sum of coincidences involving category c.
    categories = sorted({
        v
        for row in annotations
        for v in row
        if v is not None
    })
    n_c: dict[str, float] = {}
    for c in categories:
        n_c[c] = sum(coincidences.get((c, k), 0.0) for k in categories)

    n_total = sum(n_c.values())
    if math.isclose(n_total, 0.0):
        return 0.0

    # Observed disagreement.
    d_o = 1.0 - sum(coincidences.get((c, c), 0.0) for c in categories) / n_total

    # Expected disagreement (nominal metric: delta=1 if c!=c').
    d_e = 1.0 - sum(n_c[c] ** 2 for c in categories) / (n_total ** 2)

    if math.isclose(d_e, 0.0):
        return 1.0 if math.isclose(d_o, 0.0) else 0.0
    return round(1.0 - d_o / d_e, 4)


# ── Token-overlap F1 ─────────────────────────────────────────────────────────


def token_overlap_f1(
    spans_a: list[str], spans_b: list[str]
) -> dict[str, float]:
    """Token-level overlap F1 between two sets of evidence spans.

    Tokenisation is a simple whitespace split; Chinese characters are each
    treated as individual tokens.

    Parameters
    ----------
    spans_a, spans_b : list[str]
        Evidence span texts from annotator A and B respectively.

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``.
    """

    def _tokenize(spans: list[str]) -> set[str]:
        tokens: list[str] = []
        for span in spans:
            for ch in span:
                if ch.isspace():
                    continue
                tokens.append(ch)
        return set(tokens)

    tokens_a = _tokenize(spans_a)
    tokens_b = _tokenize(spans_b)

    if not tokens_a and not tokens_b:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(tokens_a & tokens_b)
    fp = len(tokens_b - tokens_a)
    fn = len(tokens_a - tokens_b)

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = (
        0.0
        if (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── Full IAA report ──────────────────────────────────────────────────────────


def compute_iaa(
    annotator_a_path: Path,
    annotator_b_path: Path,
) -> dict[str, Any]:
    """Compute a full IAA report between two annotator JSONL files.

    Returns
    -------
    dict
        Keys:
        - ``annotator_a``, ``annotator_b``: file paths.
        - ``common_report_count``: number of shared report IDs.
        - ``label_metrics``: per-field ``{accuracy, cohen_kappa}``.
        - ``overall_kappa``: micro-averaged Cohen's kappa across all label
          fields (concatenated label pairs).
        - ``span_metrics``: ``{precision, recall, f1}`` token-overlap.
        - ``disagreements``: list of ``{report_id, fields}`` dicts.
        - ``disagreement_count``: total.
    """
    rows_a = {row["report_id"]: row for row in _load_jsonl(annotator_a_path)}
    rows_b = {row["report_id"]: row for row in _load_jsonl(annotator_b_path)}
    common_ids = sorted(set(rows_a) & set(rows_b))

    result: dict[str, Any] = {
        "annotator_a": str(annotator_a_path),
        "annotator_b": str(annotator_b_path),
        "common_report_count": len(common_ids),
        "label_metrics": {},
        "overall_kappa": 0.0,
        "span_metrics": {},
        "disagreements": [],
    }

    # ---- per-field label metrics ----
    all_labels_a: list[str] = []
    all_labels_b: list[str] = []

    for field in LABEL_FIELDS:
        field_a: list[str] = []
        field_b: list[str] = []
        for rid in common_ids:
            a_val = str((rows_a[rid].get("annotation") or {}).get(field))
            b_val = str((rows_b[rid].get("annotation") or {}).get(field))
            field_a.append(a_val)
            field_b.append(b_val)
        accuracy = (
            0.0
            if not field_a
            else round(
                sum(1 for a, b in zip(field_a, field_b) if a == b) / len(field_a),
                4,
            )
        )
        result["label_metrics"][field] = {
            "accuracy": accuracy,
            "cohen_kappa": cohen_kappa(field_a, field_b),
        }
        all_labels_a.extend(field_a)
        all_labels_b.extend(field_b)

    # Overall (micro) kappa across all label fields concatenated.
    result["overall_kappa"] = cohen_kappa(all_labels_a, all_labels_b)

    # ---- span overlap ----
    span_tokens_a_all: set[str] = set()
    span_tokens_b_all: set[str] = set()
    span_f1_per_item: list[dict[str, float]] = []

    for rid in common_ids:
        a_spans = _normalize_spans(
            (rows_a[rid].get("annotation") or {}).get("evidence_spans", [])
        )
        b_spans = _normalize_spans(
            (rows_b[rid].get("annotation") or {}).get("evidence_spans", [])
        )
        span_f1_per_item.append(token_overlap_f1(a_spans, b_spans))

        # Track disagreements.
        disagreement_fields: list[str] = []
        for field in LABEL_FIELDS:
            a_val = (rows_a[rid].get("annotation") or {}).get(field)
            b_val = (rows_b[rid].get("annotation") or {}).get(field)
            if a_val != b_val:
                disagreement_fields.append(field)
        if a_spans != b_spans:
            disagreement_fields.append("evidence_spans")
        if disagreement_fields:
            result["disagreements"].append(
                {"report_id": rid, "fields": disagreement_fields}
            )

    # Macro-average span F1 over items.
    if span_f1_per_item:
        result["span_metrics"] = {
            "precision": round(
                sum(d["precision"] for d in span_f1_per_item)
                / len(span_f1_per_item),
                4,
            ),
            "recall": round(
                sum(d["recall"] for d in span_f1_per_item)
                / len(span_f1_per_item),
                4,
            ),
            "f1": round(
                sum(d["f1"] for d in span_f1_per_item) / len(span_f1_per_item),
                4,
            ),
        }
    else:
        result["span_metrics"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    result["disagreement_count"] = len(result["disagreements"])
    return result


# ── Adjudication ──────────────────────────────────────────────────────────────


def adjudicate(
    annotator_a: dict,
    annotator_b: dict,
    adjudicator: dict | None = None,
) -> dict:
    """Adjudication protocol for a single item.

    With three annotation dicts (``adjudicator`` provided): majority vote on
    each label field; evidence spans union from the majority-winning
    annotators.

    With two annotation dicts (``adjudicator is None``): if they agree the
    value is kept, otherwise ``adjudicator`` must be provided to break the
    tie -- in that case the adjudicator value is used directly.

    Parameters
    ----------
    annotator_a, annotator_b : dict
        Annotation dicts (the ``annotation`` sub-dict from the JSONL row).
    adjudicator : dict | None
        Optional third annotation dict used for tie-breaking.

    Returns
    -------
    dict
        Resolved annotation dict with the same schema.
    """
    resolved: dict[str, Any] = {}

    for field in LABEL_FIELDS:
        val_a = annotator_a.get(field)
        val_b = annotator_b.get(field)

        if adjudicator is not None:
            val_c = adjudicator.get(field)
            # Majority vote among three.
            votes = Counter([val_a, val_b, val_c])
            winner, _ = votes.most_common(1)[0]
            resolved[field] = winner
        else:
            if val_a == val_b:
                resolved[field] = val_a
            else:
                # Tie with no adjudicator -- cannot resolve; keep annotator_a
                # and flag.
                resolved[field] = val_a
                resolved.setdefault("_unresolved_fields", []).append(field)

    # Evidence spans: union from winning annotators, deduplicated.
    spans_a = _normalize_spans(annotator_a.get("evidence_spans", []))
    spans_b = _normalize_spans(annotator_b.get("evidence_spans", []))

    if adjudicator is not None:
        spans_c = _normalize_spans(adjudicator.get("evidence_spans", []))
        # Take union of all three.
        seen: set[str] = set()
        merged: list[str] = []
        for s in spans_a + spans_b + spans_c:
            if s not in seen:
                seen.add(s)
                merged.append(s)
        resolved["evidence_spans"] = merged
    else:
        if spans_a == spans_b:
            resolved["evidence_spans"] = spans_a
        else:
            # Union for two annotators when they disagree.
            seen = set()
            merged = []
            for s in spans_a + spans_b:
                if s not in seen:
                    seen.add(s)
                    merged.append(s)
            resolved["evidence_spans"] = merged

    # Confidence: take the lower of the two (conservative).
    conf_order = {"low": 0, "medium": 1, "high": 2}
    conf_a = conf_order.get(str(annotator_a.get("annotation_confidence")), 0)
    conf_b = conf_order.get(str(annotator_b.get("annotation_confidence")), 0)
    if adjudicator is not None:
        conf_c = conf_order.get(
            str(adjudicator.get("annotation_confidence")), 0
        )
        # Median of three.
        resolved["annotation_confidence"] = {
            0: "low",
            1: "medium",
            2: "high",
        }[sorted([conf_a, conf_b, conf_c])[1]]
    else:
        resolved["annotation_confidence"] = {0: "low", 1: "medium", 2: "high"}[
            min(conf_a, conf_b)
        ]

    return resolved
