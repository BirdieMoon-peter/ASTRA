import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from astra.llm import client as llm_client
from astra.llm.client import ClaudeClientConfig, ClaudeJSONClient, load_llm_settings


class ApiConnectionError(Exception):
    pass


class LoadLlmSettingsTest(unittest.TestCase):
    def test_default_config_path_is_anchored_to_project_root(self) -> None:
        project_root = Path(llm_client.__file__).resolve().parents[3]
        expected_path = project_root / ".local" / "llm_config.json"

        with patch.dict(os.environ, {}, clear=True):
            settings = load_llm_settings()

        self.assertEqual(settings.config_path, expected_path)

    def test_local_config_file_takes_priority_over_environment_variables(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / ".local" / "llm_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "api_key": "file-key",
                        "base_url": "https://example.test",
                        "model": "claude-opus-4-6",
                        "max_tokens": 1234,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "env-key",
                    "ANTHROPIC_BASE_URL": "https://env.example.test",
                    "ANTHROPIC_MODEL": "claude-haiku-4-5",
                    "ANTHROPIC_MAX_TOKENS": "999",
                },
                clear=False,
            ):
                settings = load_llm_settings(config_path)

        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.base_url, "https://example.test")
        self.assertEqual(settings.model, "claude-opus-4-6")
        self.assertEqual(settings.max_tokens, 1234)
        self.assertEqual(settings.config_path, config_path)

    def test_missing_local_config_falls_back_to_environment_variables(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"

            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "env-key",
                    "ANTHROPIC_BASE_URL": "https://env.example.test",
                    "ANTHROPIC_MODEL": "claude-haiku-4-5",
                    "ANTHROPIC_MAX_TOKENS": "2048",
                },
                clear=False,
            ):
                settings = load_llm_settings(config_path)

        self.assertEqual(settings.api_key, "env-key")
        self.assertEqual(settings.base_url, "https://env.example.test")
        self.assertEqual(settings.model, "claude-haiku-4-5")
        self.assertEqual(settings.max_tokens, 2048)
        self.assertEqual(settings.config_path, config_path)

    def test_invalid_local_config_json_raises_clear_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("{invalid json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, rf"Failed to parse LLM config JSON at {config_path}"):
                load_llm_settings(config_path)

    def test_invalid_local_max_tokens_raises_clear_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"max_tokens": "not-an-int"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Invalid max_tokens value 'not-an-int' from LLM config file"):
                load_llm_settings(config_path)

    def test_invalid_environment_max_tokens_raises_clear_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"

            with patch.dict(os.environ, {"ANTHROPIC_MAX_TOKENS": "bad-value"}, clear=True):
                with self.assertRaisesRegex(ValueError, r"Invalid max_tokens value 'bad-value' from ANTHROPIC_MAX_TOKENS environment variable"):
                    load_llm_settings(config_path)

    def test_openai_compatible_config_loads_from_local_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "openai_compatible",
                        "api_key": "file-key",
                        "base_url": "https://api.example.test/v1",
                        "model": "deepseek-chat",
                        "max_tokens": 2345,
                    }
                ),
                encoding="utf-8",
            )

            settings = load_llm_settings(config_path)

        self.assertEqual(settings.provider, "openai_compatible")
        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.base_url, "https://api.example.test/v1")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertEqual(settings.max_tokens, 2345)

    def test_openai_compatible_environment_variables_are_used_when_provider_selected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".local" / "llm_config.json"

            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "openai_compatible",
                    "OPENAI_API_KEY": "openai-key",
                    "OPENAI_BASE_URL": "https://api.openai-compatible.test/v1",
                    "OPENAI_MODEL": "deepseek-chat",
                    "OPENAI_MAX_TOKENS": "3000",
                },
                clear=True,
            ):
                settings = load_llm_settings(config_path)

        self.assertEqual(settings.provider, "openai_compatible")
        self.assertEqual(settings.api_key, "openai-key")
        self.assertEqual(settings.base_url, "https://api.openai-compatible.test/v1")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertEqual(settings.max_tokens, 3000)


class ClaudeJsonClientParseTest(unittest.TestCase):
    def test_parse_json_text_accepts_fenced_json(self) -> None:
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)

        parsed = ClaudeJSONClient._parse_json_text(client, "```json\n{\"ok\": true}\n```")

        self.assertEqual(parsed, {"ok": True})


    def test_parse_json_text_repairs_unquoted_object_keys(self) -> None:
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)

        parsed = ClaudeJSONClient._parse_json_text(client, '{"neutralized_text": "标题", summary: "正文"}')

        self.assertEqual(parsed, {"neutralized_text": "标题", "summary": "正文"})

    def test_create_json_uses_anthropic_provider(self) -> None:
        config = ClaudeClientConfig(provider="anthropic", api_key="key", model="claude-opus-4-6")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = MagicMock()
        client._last_usage = {"input_tokens": 0, "output_tokens": 0}
        client._client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"ok": true}')],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )

        result = ClaudeJSONClient.create_json(client, system="sys", user_prompt="user")

        self.assertEqual(result, {"ok": True})
        client._client.messages.create.assert_called_once()
        self.assertEqual(client.last_usage, {"input_tokens": 100, "output_tokens": 50})

    def test_create_json_uses_openai_compatible_provider(self) -> None:
        config = ClaudeClientConfig(provider="openai_compatible", api_key="key", model="deepseek-chat")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = MagicMock()
        client._last_usage = {"input_tokens": 0, "output_tokens": 0}
        client._client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"ok": true}\n```'))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=80),
        )

        result = ClaudeJSONClient.create_json(client, system="sys", user_prompt="user")

        self.assertEqual(result, {"ok": True})
        client._client.chat.completions.create.assert_called_once()
        self.assertEqual(client.last_usage, {"input_tokens": 200, "output_tokens": 80})

    def test_create_json_retries_transient_openai_connection_errors(self) -> None:
        config = ClaudeClientConfig(provider="openai_compatible", api_key="key", model="deepseek-chat")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = MagicMock()
        client._last_usage = {"input_tokens": 0, "output_tokens": 0}
        client._client.chat.completions.create.side_effect = [
            ApiConnectionError("boom"),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            ),
        ]

        with patch("astra.llm.client.time.sleep") as sleep_mock:
            result = ClaudeJSONClient.create_json(client, system="sys", user_prompt="user")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(client._client.chat.completions.create.call_count, 2)
        sleep_mock.assert_called_once()

    def test_create_json_raises_clear_error_when_client_unavailable(self) -> None:
        config = ClaudeClientConfig(provider="openai_compatible", api_key=None, model="deepseek-chat")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = None
        client._last_usage = {"input_tokens": 0, "output_tokens": 0}

        with self.assertRaisesRegex(RuntimeError, r"provider=openai_compatible"):
            ClaudeJSONClient.create_json(client, system="sys", user_prompt="user")

    def test_last_usage_defaults_to_zero(self) -> None:
        config = ClaudeClientConfig(provider="anthropic", api_key=None, model="claude-opus-4-6")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = None
        client._last_usage = {"input_tokens": 0, "output_tokens": 0}

        self.assertEqual(client.last_usage, {"input_tokens": 0, "output_tokens": 0})

    def test_last_usage_returns_copy(self) -> None:
        config = ClaudeClientConfig(provider="anthropic", api_key="key", model="claude-opus-4-6")
        client = ClaudeJSONClient.__new__(ClaudeJSONClient)
        client.config = config
        client._client = MagicMock()
        client._last_usage = {"input_tokens": 42, "output_tokens": 7}

        usage = client.last_usage
        usage["input_tokens"] = 999
        self.assertEqual(client.last_usage["input_tokens"], 42)

