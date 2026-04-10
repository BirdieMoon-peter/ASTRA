from __future__ import annotations

"""Encoder-model baselines for comparison with the ASTRA pipeline.

Provides three baselines:
  1. FinBERTBaseline -- enhanced lexicon-based sentiment (zero-shot proxy)
  2. EncoderSentimentBaseline -- HuggingFace text-classification wrapper
  3. StrongLLMBaseline -- single-prompt LLM baseline with exhaustive instructions

All prediction dicts follow the standard ASTRA output schema (see
``outputs/predictions/*.jsonl``).
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extended Chinese financial lexicons (used by FinBERTBaseline)
# ---------------------------------------------------------------------------

_POSITIVE_TERMS: tuple[str, ...] = (
    "增长", "提升", "改善", "回升", "突破", "向好", "亮眼", "稳健",
    "超预期", "高增", "放量", "扩张", "盈利", "利好", "景气",
    "修复", "新高", "加速", "翻倍", "丰厚", "产销两旺", "量价齐升",
    "毛利率提升", "净利润增长",
)

_NEGATIVE_TERMS: tuple[str, ...] = (
    "下降", "承压", "放缓", "拖累", "收窄", "下滑", "风险", "减值",
    "恶化", "亏损", "萎缩", "低迷", "疲软", "缩量", "暴跌",
    "利空", "不及预期", "下修", "走弱", "下行", "毛利率下降",
    "计提减值", "业绩下滑", "营收下降",
)

_NEUTRAL_TERMS: tuple[str, ...] = (
    "持平", "平稳", "波动", "分化", "结构性", "常态化", "维持",
    "基本面稳定", "同比持平", "环比持平", "温和", "震荡",
    "正常波动", "窄幅", "中性", "符合预期", "区间运行",
    "保持稳定", "变化不大", "整体平稳", "基本一致",
)

_STRONG_OPTIMISM_TERMS: tuple[str, ...] = (
    "强烈推荐", "买入评级", "显著增长", "大幅提升", "极具潜力",
    "战略机遇", "龙头地位", "核心竞争力", "高成长", "爆发式增长",
    "黄金期", "确定性强", "稀缺标的", "首次覆盖买入", "空间广阔",
)

_HEDGE_TERMS: tuple[str, ...] = (
    "短期", "阶段性", "暂时", "预计", "有望", "边际", "或将",
    "不排除", "存在不确定性", "关注风险", "谨慎乐观", "需观察",
)


def _count_hits(text: str, lexicon: tuple[str, ...]) -> int:
    return sum(text.count(token) for token in lexicon)


def _weighted_score(text: str) -> float:
    """Return a continuous sentiment score in [-1, 1]."""
    pos = _count_hits(text, _POSITIVE_TERMS)
    neg = _count_hits(text, _NEGATIVE_TERMS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ---------------------------------------------------------------------------
# 1. FinBERTBaseline
# ---------------------------------------------------------------------------

class FinBERTBaseline:
    """Enhanced lexicon-based sentiment baseline (zero-shot FinBERT proxy).

    Uses an expanded Chinese financial lexicon with weighted scoring to
    produce a more nuanced sentiment estimate than the simple rule baseline
    in ``astra.evaluation.baselines``.
    """

    method_name: str = "finbert_lexicon"

    def predict(self, title: str, summary: str) -> dict[str, Any]:
        text = f"{title}\n{summary}"

        pos = _count_hits(text, _POSITIVE_TERMS)
        neg = _count_hits(text, _NEGATIVE_TERMS)
        neu = _count_hits(text, _NEUTRAL_TERMS)

        # Title-weighted scoring: title terms count double.
        title_pos = _count_hits(title, _POSITIVE_TERMS) * 2
        title_neg = _count_hits(title, _NEGATIVE_TERMS) * 2
        effective_pos = pos + title_pos
        effective_neg = neg + title_neg

        total = effective_pos + effective_neg + neu
        if total == 0:
            sentiment = "neutral"
            confidence = 0.5
        else:
            ratio = (effective_pos - effective_neg) / total
            if ratio > 0.15:
                sentiment = "positive"
            elif ratio < -0.15:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            confidence = min(1.0, 0.5 + abs(ratio))

        # Build minimal evidence spans from lexicon hits.
        evidence: list[dict[str, str]] = []
        for token in _POSITIVE_TERMS + _NEGATIVE_TERMS + _NEUTRAL_TERMS:
            if token in text:
                evidence.append({"text": token, "label": "finbert_lexicon_hit"})
        evidence = evidence[:10]

        return {
            "fundamental_sentiment": sentiment,
            "strategic_optimism": "balanced",
            "phenomenon": "none",
            "uncertainty": round(1.0 - confidence, 4),
            "evidence_spans": evidence,
            "reasoning_summary": (
                f"finbert_lexicon: pos={pos} neg={neg} neu={neu} "
                f"title_boost_pos={title_pos} title_boost_neg={title_neg}"
            ),
        }


# ---------------------------------------------------------------------------
# 2. EncoderSentimentBaseline
# ---------------------------------------------------------------------------

# Mapping from various model output labels to ASTRA's 3-class sentiment.
_LABEL_MAP_2CLASS: dict[str, str] = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "positive": "positive",
    "negative": "negative",
    "LABEL_0": "negative",
    "LABEL_1": "positive",
}

_LABEL_MAP_3CLASS: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class EncoderSentimentBaseline:
    """Wrapper for any HuggingFace text-classification model.

    Falls back to the FinBERT lexicon baseline when the model cannot be
    loaded (e.g. missing ``transformers`` dependency or model weights).
    """

    method_name: str = "encoder_sentiment"

    def __init__(self, model_name: str = "bert-base-chinese") -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None
        self._load_attempted: bool = False
        self._fallback = FinBERTBaseline()

    def _ensure_pipeline(self) -> bool:
        """Lazily load the HuggingFace pipeline. Returns True on success."""
        if self._load_attempted:
            return self._pipeline is not None
        self._load_attempted = True
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded HuggingFace model: %s", self.model_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load HuggingFace model %s (%s); falling back to lexicon.",
                self.model_name,
                exc,
            )
            return False

    def _map_label(self, raw_label: str) -> str:
        """Map a model output label to ASTRA sentiment classes."""
        mapped = _LABEL_MAP_3CLASS.get(raw_label) or _LABEL_MAP_2CLASS.get(raw_label)
        if mapped is not None:
            return mapped
        # Best-effort substring matching for unknown label schemes.
        lower = raw_label.lower()
        if "pos" in lower:
            return "positive"
        if "neg" in lower:
            return "negative"
        return "neutral"

    def predict(self, title: str, summary: str) -> dict[str, Any]:
        text = f"{title} {summary}"[:512]

        if not self._ensure_pipeline():
            result = self._fallback.predict(title, summary)
            result["reasoning_summary"] = (
                f"encoder_fallback({self.model_name}): {result['reasoning_summary']}"
            )
            return result

        outputs = self._pipeline(text)
        if isinstance(outputs, list) and len(outputs) > 0:
            top = outputs[0]
        else:
            top = outputs

        raw_label = top.get("label", "neutral")
        score = float(top.get("score", 0.5))
        sentiment = self._map_label(raw_label)

        return {
            "fundamental_sentiment": sentiment,
            "strategic_optimism": "balanced",
            "phenomenon": "none",
            "uncertainty": round(1.0 - score, 4),
            "evidence_spans": [],
            "reasoning_summary": (
                f"encoder({self.model_name}): raw_label={raw_label} score={score:.4f}"
            ),
        }


# ---------------------------------------------------------------------------
# 3. StrongLLMBaseline
# ---------------------------------------------------------------------------

_STRONG_LLM_SYSTEM_PROMPT = (
    "你是资深中文金融文本分析专家。你必须严格依据标题与摘要中的显式文本做判断，"
    "不补充外部背景，不猜测作者未写出的事实。输出严格 JSON，不要输出 JSON 之外的任何内容。"
)

_STRONG_LLM_USER_PROMPT = """\
请完整分析以下中文分析师研报，依据标题与摘要，逐步判断五个维度，最终只输出一个 JSON 对象。

=== 字段定义与详细示例 ===

1. fundamental_sentiment（基本面情绪）
   只看基本面、业绩、经营趋势本身，不看包装语气。
   - "negative": 明确承压/下滑/风险上升。例："营收同比下降15%""利润大幅缩水""不良率攀升"。
   - "neutral": 事实描述为主或正负平衡。例："营收略有波动，利润基本持平""结构分化明显"。
   - "positive": 明确改善/增长/向好。例："净利润同比增长30%""订单持续放量""ROE创新高"。
   注意：若正负面事实并存且无法分出主次，优先判 neutral。

2. strategic_optimism（策略性乐观程度）
   看表述层面的乐观程度，不只看事实方向。
   - "low": 明显保守、强调风险或弱化前景。例：多次提及"不确定性""下行风险""谨慎看待"。
   - "balanced": 语气克制、正反信息都交代。例：既说增长也提到了制约因素。
   - "high": 明显偏乐观、强化利好、淡化约束、使用前瞻积极表达。
     例："强烈推荐""极具投资价值""黄金发展期""空间广阔"。
   注意：基本面 positive 不等于 strategic_optimism high。

3. phenomenon（主要现象，只选一个）
   - "hedged_downside": 承认下行事实但用缓和措辞包裹。
     例：使用"短期承压但长期向好""阶段性调整""暂时性因素"来包装负面事实。
   - "euphemistic_risk": 对明确风险使用委婉弱化表达。
     例："存在一定不确定性"（实际是重大风险）、"或有阶段性波动"（实际是持续下滑）。
   - "title_body_mismatch": 标题明显更乐观或更强势，而正文支持不足或方向不一致。
     例：标题"业绩高增，强烈推荐"，正文实际利润下降。
   - "omitted_downside_context": 突出利好，但遗漏理解结论所需的重要下行背景。
     例：只提营收增长，不提毛利率大幅下降和负债飙升。
   - "none": 以上现象均不明显。

4. evidence_spans（证据片段）
   从标题或摘要中直接摘取支持判断的原文片段。每个片段必须是连续文本、最小充分片段。
   label 取值范围：positive_fact, negative_fact, optimistic_rhetoric, hedge_cue, risk_cue,
   title_body_gap, omitted_context_hint。

5. uncertainty（0-1）
   对整体判断的不确定度。0 = 完全确信，1 = 完全不确定。

输出 JSON schema:
{{
  "fundamental_sentiment": "positive" | "neutral" | "negative",
  "strategic_optimism": "high" | "balanced" | "low",
  "phenomenon": "hedged_downside" | "euphemistic_risk" | "title_body_mismatch" | "omitted_downside_context" | "none",
  "uncertainty": number,
  "evidence_spans": [{{"text": string, "label": string}}],
  "reasoning_summary": string
}}

=== 研报文本 ===

标题:
{title}

摘要:
{summary}
"""


class StrongLLMBaseline:
    """Single-prompt LLM baseline using a comprehensive prompt.

    This tests whether a single, well-crafted prompt can match the
    multi-stage ASTRA pipeline.
    """

    method_name: str = "strong_llm"

    def __init__(self, llm_client: Any, model_label: str = "strong_llm") -> None:
        """Initialise with an ``astra.llm.client.ClaudeJSONClient`` instance."""
        self.llm_client = llm_client
        self.model_label = model_label
        self.method_name = model_label

    def predict(self, title: str, summary: str) -> dict[str, Any]:
        prompt = _STRONG_LLM_USER_PROMPT.format(title=title, summary=summary)
        result = self.llm_client.create_json(
            system=_STRONG_LLM_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        return _normalize_prediction(result, method=self.model_label)


# ---------------------------------------------------------------------------
# Prediction normalisation helpers
# ---------------------------------------------------------------------------

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}
_VALID_OPTIMISMS = {"low", "balanced", "high"}
_VALID_PHENOMENA = {
    "hedged_downside",
    "euphemistic_risk",
    "title_body_mismatch",
    "omitted_downside_context",
    "none",
}


def _normalize_prediction(raw: dict[str, Any], *, method: str) -> dict[str, Any]:
    """Ensure a prediction dict conforms to the standard ASTRA schema."""
    sentiment = str(raw.get("fundamental_sentiment", "neutral")).strip().lower()
    if sentiment not in _VALID_SENTIMENTS:
        sentiment = "neutral"

    optimism = str(raw.get("strategic_optimism", "balanced")).strip().lower()
    if optimism not in _VALID_OPTIMISMS:
        optimism = "balanced"

    phenomenon = str(raw.get("phenomenon", "none")).strip().lower()
    if phenomenon not in _VALID_PHENOMENA:
        phenomenon = "none"

    uncertainty = raw.get("uncertainty", 0.5)
    try:
        uncertainty = float(uncertainty)
        uncertainty = max(0.0, min(1.0, uncertainty))
    except (TypeError, ValueError):
        uncertainty = 0.5

    evidence_spans = raw.get("evidence_spans") or []
    if not isinstance(evidence_spans, list):
        evidence_spans = []
    clean_spans: list[dict[str, str]] = []
    for span in evidence_spans:
        if isinstance(span, dict) and span.get("text"):
            clean_spans.append({
                "text": str(span["text"]),
                "label": str(span.get("label", "unknown")),
            })
    clean_spans = clean_spans[:15]

    reasoning = str(raw.get("reasoning_summary", f"{method}_baseline"))

    return {
        "fundamental_sentiment": sentiment,
        "strategic_optimism": optimism,
        "phenomenon": phenomenon,
        "uncertainty": round(uncertainty, 4),
        "evidence_spans": clean_spans,
        "reasoning_summary": reasoning,
    }


# ---------------------------------------------------------------------------
# run_all_baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    reports: list[dict[str, Any]],
    llm_client: Any | None = None,
    *,
    encoder_model: str = "bert-base-chinese",
) -> dict[str, list[dict[str, Any]]]:
    """Run all available baselines on a list of report dicts.

    Each report dict must have at least: ``report_id``, ``report_date``,
    ``stock_code``, ``split``, ``title``, ``summary``.

    Parameters
    ----------
    reports:
        List of report dicts (e.g. rows from the experiment CSV).
    llm_client:
        An ``astra.llm.client.ClaudeJSONClient`` instance.  If *None*,
        the ``StrongLLMBaseline`` is skipped.
    encoder_model:
        HuggingFace model name for the encoder baseline.

    Returns
    -------
    dict mapping method name to a list of prediction dicts.
    """
    finbert = FinBERTBaseline()
    encoder = EncoderSentimentBaseline(model_name=encoder_model)
    strong: StrongLLMBaseline | None = None
    if llm_client is not None:
        strong = StrongLLMBaseline(llm_client)

    results: dict[str, list[dict[str, Any]]] = {
        finbert.method_name: [],
        encoder.method_name: [],
    }
    if strong is not None:
        results[strong.method_name] = []

    meta_keys = ("report_id", "report_date", "stock_code", "split", "title", "summary")

    for i, report in enumerate(reports):
        title = report.get("title", "")
        summary = report.get("summary", "")
        meta = {k: report.get(k, "") for k in meta_keys}

        # FinBERT lexicon
        try:
            pred = finbert.predict(title, summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FinBERT failed on report %s: %s", report.get("report_id"), exc)
            pred = _fallback_prediction("finbert_lexicon_error")
        row = {**meta, "method": finbert.method_name, **pred}
        results[finbert.method_name].append(row)

        # Encoder
        try:
            pred = encoder.predict(title, summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Encoder failed on report %s: %s", report.get("report_id"), exc)
            pred = _fallback_prediction("encoder_error")
        row = {**meta, "method": encoder.method_name, **pred}
        results[encoder.method_name].append(row)

        # Strong LLM
        if strong is not None:
            try:
                pred = strong.predict(title, summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Strong LLM failed on report %s: %s", report.get("report_id"), exc)
                pred = _fallback_prediction("strong_llm_error")
            row = {**meta, "method": strong.method_name, **pred}
            results[strong.method_name].append(row)

        if (i + 1) % 20 == 0:
            logger.info("Processed %d / %d reports", i + 1, len(reports))

    return results


def _fallback_prediction(reason: str) -> dict[str, Any]:
    """Return a safe default prediction when a baseline errors."""
    return {
        "fundamental_sentiment": "neutral",
        "strategic_optimism": "balanced",
        "phenomenon": "none",
        "uncertainty": 0.95,
        "evidence_spans": [],
        "reasoning_summary": f"fallback: {reason}",
    }
