from __future__ import annotations

import re
from typing import Any

POSITIVE_HINTS = ("增长", "提升", "改善", "修复", "亮眼", "稳健", "回升", "突破", "向好")
NEGATIVE_HINTS = ("下降", "承压", "放缓", "拖累", "收窄", "下滑", "风险", "减值", "恶化")
OPTIMISM_HINTS = ("积极", "主动", "稳步", "健康发展", "向好", "优化", "夯实", "蓄势", "信心")
HEDGE_HINTS = ("短期", "阶段性", "暂时", "预计", "有望", "边际", "平稳")


def _count_hits(text: str, lexicon: tuple[str, ...]) -> int:
    return sum(text.count(token) for token in lexicon)


def rule_based_prediction(title: str, summary: str) -> dict[str, Any]:
    text = f"{title}\n{summary}"
    positive_hits = _count_hits(text, POSITIVE_HINTS)
    negative_hits = _count_hits(text, NEGATIVE_HINTS)
    optimism_hits = _count_hits(text, OPTIMISM_HINTS)
    hedge_hits = _count_hits(text, HEDGE_HINTS)

    if positive_hits > negative_hits:
        sentiment = "positive"
    elif negative_hits > positive_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if optimism_hits >= max(2, hedge_hits + 1):
        strategic_optimism = "high"
    elif negative_hits > optimism_hits and hedge_hits == 0:
        strategic_optimism = "low"
    else:
        strategic_optimism = "balanced"

    phenomenon = "none"
    if hedge_hits > 0 and negative_hits > 0:
        phenomenon = "hedged_downside"
    elif optimism_hits > 0 and negative_hits > 0:
        phenomenon = "euphemistic_risk"
    elif title and summary and _count_hits(title, OPTIMISM_HINTS) > _count_hits(summary, OPTIMISM_HINTS) + 1:
        phenomenon = "title_body_mismatch"

    evidence = []
    for token in POSITIVE_HINTS + NEGATIVE_HINTS + OPTIMISM_HINTS + HEDGE_HINTS:
        if token in text:
            evidence.append({"text": token, "label": "lexicon_hit"})
    return {
        "fundamental_sentiment": sentiment,
        "strategic_optimism": strategic_optimism,
        "phenomenon": phenomenon,
        "uncertainty": 0.45,
        "evidence_spans": evidence[:8],
        "reasoning_summary": "rule_based_baseline",
    }


def simple_title_body_divergence(title: str, summary: str) -> float:
    title_score = _count_hits(title, POSITIVE_HINTS) - _count_hits(title, NEGATIVE_HINTS)
    summary_score = _count_hits(summary, POSITIVE_HINTS) - _count_hits(summary, NEGATIVE_HINTS)
    return float(title_score - summary_score)
