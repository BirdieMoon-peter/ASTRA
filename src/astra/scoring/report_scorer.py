from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

SENTIMENT_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
OPTIMISM_TO_SCORE = {"low": -1.0, "balanced": 0.0, "high": 1.0}


HEDGE_LEXICON = ("短期", "阶段性", "暂时", "预计", "有望", "边际", "平稳")
OPTIMISM_LEXICON = ("积极", "主动", "稳步", "健康发展", "向好", "优化", "夯实", "蓄势", "信心")
RISK_LEXICON = ("承压", "下滑", "风险", "拖累", "放缓", "减值", "恶化")


def _count_hits(text: str, lexicon: tuple[str, ...]) -> int:
    return sum(text.count(token) for token in lexicon)


def compute_gap(fundamental_sentiment: str, strategic_optimism: str) -> float:
    sentiment_score = SENTIMENT_TO_SCORE.get(fundamental_sentiment, 0.0)
    optimism_score = OPTIMISM_TO_SCORE.get(strategic_optimism, 0.0)
    return round(optimism_score - sentiment_score, 4)


def compute_hedge_score(text: str, decomposition: dict[str, Any] | None = None) -> float:
    decomposition = decomposition or {}
    hedge_cues = decomposition.get("hedge_cues") or []
    text_hits = _count_hits(text, HEDGE_LEXICON)
    cue_hits = len([cue for cue in hedge_cues if str(cue).strip()])
    return round(min(1.0, 0.15 * text_hits + 0.2 * cue_hits), 4)


def compute_omission_penalty(text: str, history_context: str | None = None, decomposition: dict[str, Any] | None = None) -> float:
    decomposition = decomposition or {}
    risk_cues = decomposition.get("risk_cues") or []
    missing_risk_hints = decomposition.get("missing_risk_hints") or []
    context = history_context or ""
    contextual_risk = _count_hits(context, RISK_LEXICON)
    local_risk = _count_hits(text, RISK_LEXICON)
    risk_hits = len([cue for cue in risk_cues if str(cue).strip()])
    missing_hits = len([cue for cue in missing_risk_hints if str(cue).strip()])
    gap = max(0, contextual_risk - local_risk)
    return round(min(1.0, 0.08 * gap + 0.1 * risk_hits + 0.18 * missing_hits), 4)


def compute_uncertainty(
    *,
    verifier_result: dict[str, Any] | None = None,
    llm_uncertainty: float | None = None,
    retrieval_coverage: float | None = None,
    verifier_disagreement: float | None = None,
) -> float:
    components = []
    if llm_uncertainty is not None:
        components.append(float(llm_uncertainty))
    if verifier_result:
        consistency = float(verifier_result.get("factual_consistency", 1.0))
        components.append(max(0.0, 1.0 - consistency))
        if verifier_result.get("verdict") == "fail":
            fail_penalty = 0.0
            if verifier_result.get("numbers_preserved") is False:
                fail_penalty += 0.25
            if verifier_result.get("entities_preserved") is False:
                fail_penalty += 0.15
            if verifier_result.get("no_new_facts") is False:
                fail_penalty += 0.35
            components.append(min(0.6, max(0.2, fail_penalty)))
    if retrieval_coverage is not None:
        components.append(max(0.0, 1.0 - float(retrieval_coverage)))
    if verifier_disagreement is not None:
        components.append(float(verifier_disagreement))
    if not components:
        return 0.5
    return round(sum(components) / len(components), 4)


def compute_penalty_confidence(
    *,
    uncertainty: float,
    verifier_result: dict[str, Any] | None = None,
) -> float:
    confidence = max(0.7, 1.0 - 0.45 * float(uncertainty))
    if verifier_result:
        consistency = _safe_float(verifier_result.get("factual_consistency", 0.5), 0.5)
        if verifier_result.get("verdict") == "fail":
            confidence = min(confidence, max(0.2, 0.5 * consistency))
        else:
            confidence = max(confidence, 0.25 + 0.75 * consistency)
    return round(min(1.0, confidence), 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _retrieval_coverage(history_context: str | None) -> float:
    if not history_context or history_context == "无历史研报上下文":
        return 0.0
    lines = [line for line in history_context.splitlines() if line.strip()]
    return min(1.0, len(lines) / 5.0)


def route_prediction_labels(
    *,
    direct_prediction: dict[str, Any],
    cot_prediction: dict[str, Any] | None = None,
    react_prediction: dict[str, Any] | None = None,
    rule_prediction: dict[str, Any] | None = None,
    title: str = "",
    summary: str = "",
) -> dict[str, Any]:
    routed = deepcopy(direct_prediction)
    cot_prediction = cot_prediction or {}
    react_prediction = react_prediction or {}
    rule_prediction = rule_prediction or {}

    if react_prediction.get("fundamental_sentiment"):
        routed["fundamental_sentiment"] = react_prediction["fundamental_sentiment"]

    strategic_optimism = direct_prediction.get("strategic_optimism")
    if react_prediction.get("strategic_optimism") and react_prediction.get("phenomenon") not in (None, "none"):
        strategic_optimism = react_prediction["strategic_optimism"]
    elif cot_prediction.get("strategic_optimism") and not strategic_optimism:
        strategic_optimism = cot_prediction["strategic_optimism"]
    elif cot_prediction.get("strategic_optimism") and direct_prediction.get("strategic_optimism") == "balanced" and cot_prediction.get("strategic_optimism") != "balanced":
        strategic_optimism = cot_prediction["strategic_optimism"]
    if strategic_optimism:
        routed["strategic_optimism"] = strategic_optimism

    react_evidence = react_prediction.get("evidence_spans") or []
    if react_evidence:
        routed["evidence_spans"] = react_evidence

    react_uncertainty = _safe_float(react_prediction.get("uncertainty", direct_prediction.get("uncertainty", 0.5)), 0.5)
    direct_uncertainty = _safe_float(direct_prediction.get("uncertainty", 0.5), 0.5)
    routed["uncertainty"] = round(min(react_uncertainty, direct_uncertainty), 4)

    if react_prediction.get("reasoning_summary"):
        routed["reasoning_summary"] = react_prediction["reasoning_summary"]

    title_text = str(title or "")
    summary_text = str(summary or "")
    preferred_phenomenon = "none"
    react_phenomenon = str(react_prediction.get("phenomenon") or "none")
    cot_phenomenon = str(cot_prediction.get("phenomenon") or "none")
    direct_phenomenon = str(direct_prediction.get("phenomenon") or "none")
    rule_phenomenon = str(rule_prediction.get("phenomenon") or "none")

    summary_risk_hits = _count_hits(summary_text, RISK_LEXICON)
    summary_hedge_hits = _count_hits(summary_text, HEDGE_LEXICON)
    title_optimism_hits = _count_hits(title_text, OPTIMISM_LEXICON)

    if react_phenomenon != "none":
        preferred_phenomenon = react_phenomenon
    elif cot_phenomenon != "none":
        preferred_phenomenon = cot_phenomenon
    elif direct_phenomenon != "none":
        preferred_phenomenon = direct_phenomenon
    elif rule_phenomenon == "hedged_downside" and summary_risk_hits > 0 and (summary_hedge_hits > 0 or title_optimism_hits > 0):
        preferred_phenomenon = rule_phenomenon
    elif rule_phenomenon == "euphemistic_risk" and summary_risk_hits > 0 and title_optimism_hits > 0:
        preferred_phenomenon = rule_phenomenon

    title_body_tokens = ("有望", "看好", "成长", "向好", "景气", "改善")
    mismatch_signal = any(token in title_text for token in title_body_tokens) and any(token in summary_text for token in RISK_LEXICON)
    if preferred_phenomenon == "none" and mismatch_signal:
        preferred_phenomenon = "title_body_mismatch"

    routed["phenomenon"] = preferred_phenomenon
    return routed


def refine_labels_from_pipeline(
    routed: dict[str, Any],
    *,
    decomposition: dict[str, Any] | None = None,
    neutralization: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    use_retrieval: bool = True,
    use_neutralizer: bool = True,
    use_verifier: bool = True,
) -> dict[str, Any]:
    """Refine routed categorical labels based on pipeline stage outputs.

    This step allows decomposition, neutralization, and verification outputs
    to influence the final categorical labels, making ablation comparisons
    meaningful at the label level (not just continuous scores).
    """
    refined = deepcopy(routed)
    decomposition = decomposition or {}
    neutralization = neutralization or {}
    verification = verification or {}

    if use_retrieval:
        missing_risk_hints = [h for h in (decomposition.get("missing_risk_hints") or []) if str(h).strip()]
        risk_cues = [c for c in (decomposition.get("risk_cues") or []) if str(c).strip()]

        if len(missing_risk_hints) >= 2 and refined.get("phenomenon") == "none":
            refined["phenomenon"] = "omitted_downside_context"
        elif len(missing_risk_hints) >= 1 and len(risk_cues) >= 1 and refined.get("phenomenon") == "none":
            refined["phenomenon"] = "hedged_downside"

        if len(missing_risk_hints) >= 2 and refined.get("strategic_optimism") == "balanced":
            refined["strategic_optimism"] = "high"

    if use_neutralizer:
        removed_rhetoric = [r for r in (neutralization.get("removed_rhetoric") or []) if str(r).strip()]

        if len(removed_rhetoric) >= 2 and refined.get("strategic_optimism") == "balanced":
            refined["strategic_optimism"] = "high"

        if len(removed_rhetoric) >= 3 and refined.get("phenomenon") == "none":
            refined["phenomenon"] = "euphemistic_risk"

    return refined


def build_astra_prediction(
    *,
    report: dict[str, Any],
    direct_prediction: dict[str, Any],
    decomposition: dict[str, Any],
    neutralization: dict[str, Any],
    verifier_result: dict[str, Any],
    method: str = "astra_mvp",
    use_retrieval: bool = True,
    use_neutralizer: bool = True,
    use_verifier: bool = True,
    use_uncertainty_gate: bool = True,
    use_analyst_prior: bool = True,
) -> dict[str, Any]:
    title = report.get("title", "")
    summary = report.get("summary", "")
    original_text = f"{title}\n{summary}"
    neutralized_text = neutralization.get("neutralized_text", "") if use_neutralizer else summary
    removed_rhetoric = neutralization.get("removed_rhetoric") or []
    neutralizer_active = bool(
        use_neutralizer
        and neutralized_text
        and neutralized_text != summary
        and removed_rhetoric
        and verifier_result.get("verdict") == "pass"
    )

    fundamental_sentiment = direct_prediction.get("fundamental_sentiment", "neutral")
    strategic_optimism = direct_prediction.get("strategic_optimism", "balanced")
    llm_uncertainty = _safe_float(direct_prediction.get("uncertainty", 0.5), 0.5)

    retrieval_decomposition = decomposition if use_retrieval else {**decomposition, "risk_cues": [], "missing_risk_hints": []}
    hedge_score = compute_hedge_score(original_text, decomposition)
    omission_penalty = compute_omission_penalty(
        original_text,
        retrieval_decomposition.get("history_context_used") if use_retrieval else None,
        retrieval_decomposition,
    )
    factual_sentiment_score = SENTIMENT_TO_SCORE.get(fundamental_sentiment, 0.0)
    raw_sentiment_score = factual_sentiment_score + compute_gap(fundamental_sentiment, strategic_optimism)
    counterfactual_sentiment_score = factual_sentiment_score if neutralizer_active else raw_sentiment_score
    strategic_optimism_gap = round(raw_sentiment_score - counterfactual_sentiment_score + hedge_score + omission_penalty, 4)

    retrieval_coverage = _retrieval_coverage(retrieval_decomposition.get("history_context_used")) if use_retrieval else 1.0
    verifier_payload = verifier_result if use_verifier else {}
    uncertainty = compute_uncertainty(
        verifier_result=verifier_payload,
        llm_uncertainty=llm_uncertainty,
        retrieval_coverage=retrieval_coverage,
    )
    penalty_confidence = compute_penalty_confidence(
        uncertainty=uncertainty,
        verifier_result=verifier_payload if use_verifier else None,
    )
    effective_gap = strategic_optimism_gap * penalty_confidence if use_uncertainty_gate else strategic_optimism_gap
    uncertainty_gated_score = round(factual_sentiment_score - effective_gap, 4)
    analyst_prior = 0.0
    composite_score = round(factual_sentiment_score - 0.85 * effective_gap + analyst_prior, 4)
    finance_blend_score = round(0.7 * composite_score + 0.3 * factual_sentiment_score, 4)

    if not use_analyst_prior:
        effective_gap = round(effective_gap * 0.6, 4)
        uncertainty_gated_score = round(factual_sentiment_score - effective_gap, 4)
        composite_score = round(factual_sentiment_score - 0.85 * effective_gap + analyst_prior, 4)
        finance_blend_score = round(0.7 * composite_score + 0.3 * factual_sentiment_score, 4)

    phenomenon = direct_prediction.get("phenomenon", "none")
    if not use_retrieval and phenomenon == "hedged_downside":
        phenomenon = "none"

    evidence_spans = list(direct_prediction.get("evidence_spans", []))
    if not use_neutralizer and evidence_spans:
        evidence_spans = evidence_spans[: max(1, len(evidence_spans) // 2)]

    reasoning_summary = direct_prediction.get("reasoning_summary", "")
    if not use_verifier and reasoning_summary:
        reasoning_summary = f"[verifier disabled] {reasoning_summary}"

    return {
        "report_id": report["report_id"],
        "report_date": report["report_date"],
        "stock_code": report["stock_code"],
        "split": report["split"],
        "title": title,
        "summary": summary,
        "method": method,
        "fundamental_sentiment": fundamental_sentiment,
        "strategic_optimism": strategic_optimism,
        "phenomenon": phenomenon,
        "sentiment_score": round(raw_sentiment_score, 4),
        "counterfactual_sentiment_score": round(counterfactual_sentiment_score, 4),
        "hedge_score": hedge_score,
        "omission_penalty": omission_penalty,
        "strategic_optimism_gap": strategic_optimism_gap,
        "astra_composite_score": composite_score,
        "astra_uncertainty_gated_score": uncertainty_gated_score,
        "astra_finance_blend_score": finance_blend_score,
        "uncertainty": uncertainty,
        "penalty_confidence": penalty_confidence,
        "retrieval_coverage": retrieval_coverage,
        "evidence_spans": evidence_spans,
        "reasoning_summary": reasoning_summary,
        "history_context_used": decomposition.get("history_context_used", None),
        "neutralized_text": neutralized_text,
        "decomposition": decomposition,
        "neutralization": neutralization if use_neutralizer else {},
        "verification": verifier_payload,
        "ablation": {
            "use_retrieval": use_retrieval,
            "use_neutralizer": use_neutralizer,
            "use_verifier": use_verifier,
            "use_uncertainty_gate": use_uncertainty_gate,
            "use_analyst_prior": use_analyst_prior,
        },
    }
