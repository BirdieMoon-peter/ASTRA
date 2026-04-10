from __future__ import annotations

from typing import Any

from astra.config.prompts import DECOMPOSER_SYSTEM_PROMPT, DECOMPOSER_USER_PROMPT
from astra.llm.client import ClaudeJSONClient


class Decomposer:
    def __init__(self, llm_client: ClaudeJSONClient) -> None:
        self.llm_client = llm_client

    def run(self, *, title: str, summary: str, history_context: str) -> dict[str, Any]:
        prompt = DECOMPOSER_USER_PROMPT.format(
            title=title,
            summary=summary,
            history_context=history_context,
        )
        return self.llm_client.create_json(system=DECOMPOSER_SYSTEM_PROMPT, user_prompt=prompt)
