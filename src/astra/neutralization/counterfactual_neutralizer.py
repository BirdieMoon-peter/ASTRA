from __future__ import annotations

import json
import re
from typing import Any

from astra.config.prompts import NEUTRALIZER_SYSTEM_PROMPT, NEUTRALIZER_USER_PROMPT
from astra.llm.client import ClaudeJSONClient


_NUMERIC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_TOKEN_RE.findall(text))


def _fallback_output(summary: str) -> dict[str, Any]:
    return {
        "neutralized_text": _normalize_text(summary),
        "removed_rhetoric": [],
        "preserved_facts": [],
    }


def _sanitize_list(values: Any, source_text: str) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        if text in source_text:
            cleaned.append(text)
            seen.add(text)
    return cleaned[:8]


def _sanitize_neutralized_output(raw_output: Any, *, title: str, summary: str) -> dict[str, Any]:
    if not isinstance(raw_output, dict):
        return _fallback_output(summary)

    source_text = _normalize_text(f"{title}\n{summary}")
    source_summary = _normalize_text(summary)
    neutralized_text = _normalize_text(raw_output.get("neutralized_text"))
    if not neutralized_text:
        return _fallback_output(summary)

    source_numbers = _numeric_tokens(source_text)
    candidate_numbers = _numeric_tokens(neutralized_text)
    introduces_new_numbers = not candidate_numbers.issubset(source_numbers)
    source_length = max(len(source_summary), 1)
    too_long = len(neutralized_text) > max(len(source_summary) + 120, int(source_length * 1.6))

    removed_rhetoric = _sanitize_list(raw_output.get("removed_rhetoric"), source_text)
    preserved_facts = _sanitize_list(raw_output.get("preserved_facts"), source_text)
    summary_changed = neutralized_text != source_summary
    removed_any_rhetoric = any(text not in neutralized_text for text in removed_rhetoric)

    if introduces_new_numbers or too_long or (summary_changed and not removed_any_rhetoric):
        return _fallback_output(summary)

    return {
        "neutralized_text": neutralized_text,
        "removed_rhetoric": removed_rhetoric,
        "preserved_facts": preserved_facts,
    }


class CounterfactualNeutralizer:
    def __init__(self, llm_client: ClaudeJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, *, title: str, summary: str, decomposition: dict[str, Any]) -> dict[str, Any]:
        prompt = NEUTRALIZER_USER_PROMPT.format(
            title=title,
            summary=summary,
            decomposition_json=json.dumps(decomposition, ensure_ascii=False, indent=2),
        )
        raw_output = self.llm_client.create_json(system=NEUTRALIZER_SYSTEM_PROMPT, user_prompt=prompt)
        return _sanitize_neutralized_output(raw_output, title=title, summary=summary)
