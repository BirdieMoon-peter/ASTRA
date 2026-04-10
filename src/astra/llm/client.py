from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    # Keep vendored packages as a fallback, but prefer the active Python
    # environment's site-packages first. Some vendored SDK wheels are incomplete
    # (e.g. missing compiled pydantic_core), which can disable the LLM client.
    sys.path.append(str(VENDOR_DIR))

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def _load_anthropic_sdk() -> Any:
    global anthropic
    if anthropic is not None:
        return anthropic
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        return None
    anthropic = anthropic_sdk
    return anthropic


def _load_openai_sdk() -> Any:
    global OpenAI
    if OpenAI is not None:
        return OpenAI
    try:
        from openai import OpenAI as OpenAIClient
    except ImportError:
        return None
    OpenAI = OpenAIClient
    return OpenAI


DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".local" / "llm_config.json"
DEFAULT_PROVIDER = "anthropic"
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible"}


@dataclass
class ClaudeClientConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = "claude-opus-4-6"
    max_tokens: int = 4000
    api_key: str | None = None
    base_url: str | None = None
    seed: int | None = None
    config_path: Path = DEFAULT_CONFIG_PATH


def _parse_max_tokens(raw_value: Any, source: str) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_tokens value {raw_value!r} from {source}; expected an integer") from exc


def _parse_seed(raw_value: Any, source: str) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seed value {raw_value!r} from {source}; expected an integer") from exc


def _normalize_provider(raw_value: Any) -> str:
    provider = str(raw_value or DEFAULT_PROVIDER).strip().lower()
    if provider == "":
        return DEFAULT_PROVIDER
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return "openai_compatible"
    return provider


def _provider_env_name(provider: str, suffix: str) -> str:
    prefix = "OPENAI" if provider == "openai_compatible" else "ANTHROPIC"
    return f"{prefix}_{suffix}"


def load_llm_settings(config_path: Path | None = None) -> ClaudeClientConfig:
    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    file_settings: dict[str, Any] = {}

    if resolved_config_path.exists():
        try:
            file_settings = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM config JSON at {resolved_config_path}") from exc

    provider = _normalize_provider(file_settings.get("provider") or os.getenv("LLM_PROVIDER"))
    model = file_settings.get("model") or os.getenv(_provider_env_name(provider, "MODEL")) or ClaudeClientConfig.model
    max_tokens_value = file_settings.get("max_tokens")
    if max_tokens_value is not None:
        max_tokens = _parse_max_tokens(max_tokens_value, "LLM config file")
    else:
        env_name = _provider_env_name(provider, "MAX_TOKENS")
        env_max_tokens = os.getenv(env_name)
        max_tokens = (
            _parse_max_tokens(env_max_tokens, f"{env_name} environment variable")
            if env_max_tokens is not None
            else ClaudeClientConfig.max_tokens
        )
    api_key = file_settings.get("api_key") or os.getenv(_provider_env_name(provider, "API_KEY"))
    base_url = file_settings.get("base_url") or os.getenv(_provider_env_name(provider, "BASE_URL"))
    seed = _parse_seed(file_settings.get("seed") or os.getenv(_provider_env_name(provider, "SEED")), "LLM config file or environment")

    return ClaudeClientConfig(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        seed=seed,
        config_path=resolved_config_path,
    )


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ClaudeJSONClient:
    def __init__(self, config: ClaudeClientConfig | None = None) -> None:
        self.config = config or load_llm_settings()
        self._client = None
        self._last_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        if not self.config.api_key:
            return

        if self.config.provider == "anthropic":
            anthropic_sdk = _load_anthropic_sdk()
            if anthropic_sdk is None:
                return
            client_kwargs: dict[str, Any] = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = anthropic_sdk.Anthropic(**client_kwargs)
            return

        if self.config.provider == "openai_compatible":
            openai_client_cls = _load_openai_sdk()
            if openai_client_cls is None:
                return
            client_kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = openai_client_cls(**client_kwargs)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def last_usage(self) -> dict[str, int]:
        """Return token usage from the most recent API call."""
        return dict(self._last_usage)

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        candidates = [stripped]
        fenced_match = FENCED_JSON_RE.search(stripped)
        if fenced_match:
            candidates.insert(0, fenced_match.group(1).strip())

        repaired_candidates = []
        for candidate in candidates:
            repaired = re.sub(r'(?m)(^\s*|[,\{]\s*)([A-Za-z_][A-Za-z0-9_ ]*?):', lambda m: f'{m.group(1)}"{m.group(2).strip()}":', candidate)
            repaired_candidates.append(repaired)

        for candidate in [*candidates, *repaired_candidates]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        object_start = stripped.find("{")
        if object_start != -1:
            depth = 0
            in_string = False
            escape = False
            for index in range(object_start, len(stripped)):
                char = stripped[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[object_start:index + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"LLM did not return valid JSON: {stripped[:200]}")

    def _anthropic_text(self, *, system: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        return "".join(block.text for block in response.content if block.type == "text")

    def _openai_compatible_text(self, *, system: str, user_prompt: str) -> str:
        request = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if self.config.base_url and "127.0.0.1:11434" not in self.config.base_url:
            request["response_format"] = {"type": "json_object"}
        if self.config.seed is not None:
            request["seed"] = self.config.seed

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**request)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self._last_usage = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    }
                content = response.choices[0].message.content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                parts.append(text)
                    return "".join(parts)
                return content or ""
            except Exception as exc:
                error_name = exc.__class__.__name__
                if not any(token in error_name for token in ("Connection", "Timeout", "Protocol")):
                    raise
                last_error = exc
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

        if last_error is not None:
            raise last_error
        return ""

    def create_json(self, *, system: str, user_prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError(
                f"LLM client is unavailable for provider={self.config.provider}; check API key and SDK installation"
            )

        if self.config.provider == "anthropic":
            text = self._anthropic_text(system=system, user_prompt=user_prompt)
        elif self.config.provider == "openai_compatible":
            text = self._openai_compatible_text(system=system, user_prompt=user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
        return self._parse_json_text(text)
