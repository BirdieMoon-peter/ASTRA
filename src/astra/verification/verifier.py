from __future__ import annotations

import re
from typing import Any

from astra.config.prompts import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_PROMPT
from astra.llm.client import ClaudeJSONClient


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class Verifier:
    def __init__(self, llm_client: ClaudeJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, *, title: str, summary: str, neutralized_text: str) -> dict[str, Any]:
        if _normalize_text(neutralized_text) == _normalize_text(summary):
            return {
                "numbers_preserved": True,
                "entities_preserved": True,
                "no_new_facts": True,
                "factual_consistency": 1.0,
                "verdict": "pass",
                "issues": [],
            }

        prompt = VERIFIER_USER_PROMPT.format(
            title=title,
            summary=summary,
            neutralized_text=neutralized_text,
        )
        result = self.llm_client.create_json(system=VERIFIER_SYSTEM_PROMPT, user_prompt=prompt)
        return {
            "numbers_preserved": bool(result.get("numbers_preserved", False)),
            "entities_preserved": bool(result.get("entities_preserved", False)),
            "no_new_facts": bool(result.get("no_new_facts", False)),
            "factual_consistency": float(result.get("factual_consistency", 0.0)),
            "verdict": str(result.get("verdict", "fail")),
            "issues": result.get("issues", []) if isinstance(result.get("issues", []), list) else [],
        }
