from __future__ import annotations

from typing import Any

from astra.evaluation.baselines import simple_title_body_divergence
from astra.finance.signal_registry import signal_names

SENTIMENT_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
COMPOSITE_WEIGHT = 0.75
BODY_ONLY_WEIGHT = 0.25
FINANCE_PLUS_GAP_WEIGHT = 0.5
UNCERTAINTY_QUALITY_WEIGHT = 0.8
DEFAULT_UNCERTAINTY = 0.5


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_report_signal(prediction: dict[str, Any], title: str, summary: str) -> dict[str, float]:
    body_only = SENTIMENT_SCORE.get(prediction.get("fundamental_sentiment"), 0.0)
    title_body_divergence = simple_title_body_divergence(title, summary)
    cf_gap = _safe_float(prediction.get("strategic_optimism_gap", 0.0), 0.0)
    uncertainty = _clamp(
        _safe_float(prediction.get("uncertainty", DEFAULT_UNCERTAINTY), DEFAULT_UNCERTAINTY),
        0.0,
        1.0,
    )
    composite = _safe_float(prediction.get("astra_composite_score"), body_only - cf_gap)
    gap_signal = -cf_gap
    gated = composite * (1.0 - uncertainty)
    finance_blend = (COMPOSITE_WEIGHT * composite) + (BODY_ONLY_WEIGHT * body_only)
    finance_plus_gap = finance_blend + (0.15 * gap_signal)
    gap_scaled = COMPOSITE_WEIGHT * gap_signal
    quality_weighted = finance_blend * (1.0 - (UNCERTAINTY_QUALITY_WEIGHT * uncertainty))
    return {
        "body_only_signal": round(body_only, 6),
        "title_body_divergence": round(title_body_divergence, 6),
        "counterfactual_gap": round(gap_signal, 6),
        "astra_composite": round(composite, 6),
        "astra_uncertainty_gated": round(gated, 6),
        "astra_finance_blend": round(finance_blend, 6),
        "astra_finance_plus_gap": round(finance_plus_gap, 6),
        "astra_gap_scaled": round(gap_scaled, 6),
        "astra_quality_weighted": round(quality_weighted, 6),
    }
