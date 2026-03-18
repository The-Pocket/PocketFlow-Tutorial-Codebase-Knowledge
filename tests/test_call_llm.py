"""Unit tests for utils/call_llm.py – focused on MiniMax provider support."""

import json
import os
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_env(monkeypatch):
    """Remove all LLM-related env vars to start with a blank slate."""
    for key in list(os.environ):
        if key.startswith(("LLM_PROVIDER", "GEMINI_", "MINIMAX_", "OPENAI_", "XAI_", "OLLAMA_")):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# get_llm_provider – auto-detection
# ---------------------------------------------------------------------------

class TestGetLlmProvider:
    """Tests for the provider auto-detection logic."""

    def test_explicit_provider_takes_priority(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "XAI")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        from utils.call_llm import get_llm_provider
        assert get_llm_provider() == "XAI"

    def test_gemini_auto_detected_before_minimax(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        from utils.call_llm import get_llm_provider
        assert get_llm_provider() == "GEMINI"

    def test_minimax_auto_detected_by_api_key(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        from utils.call_llm import get_llm_provider
        assert get_llm_provider() == "MINIMAX"

    def test_no_provider_detected(self, monkeypatch):
        _clean_env(monkeypatch)
        from utils.call_llm import get_llm_provider
        assert get_llm_provider() is None


# ---------------------------------------------------------------------------
# _PROVIDER_DEFAULTS – MiniMax defaults
# ---------------------------------------------------------------------------

class TestProviderDefaults:
    """Verify built-in defaults for the MINIMAX provider."""

    def test_minimax_defaults_exist(self):
        from utils.call_llm import _PROVIDER_DEFAULTS
        assert "MINIMAX" in _PROVIDER_DEFAULTS

    def test_minimax_base_url(self):
        from utils.call_llm import _PROVIDER_DEFAULTS
        assert _PROVIDER_DEFAULTS["MINIMAX"]["base_url"] == "https://api.minimax.io"

    def test_minimax_default_model(self):
        from utils.call_llm import _PROVIDER_DEFAULTS
        assert _PROVIDER_DEFAULTS["MINIMAX"]["model"] == "MiniMax-M2.7"

    def test_minimax_temperature_range(self):
        from utils.call_llm import _PROVIDER_DEFAULTS
        d = _PROVIDER_DEFAULTS["MINIMAX"]
        assert d["min_temperature"] > 0
        assert d["max_temperature"] <= 1.0


# ---------------------------------------------------------------------------
# _call_llm_provider – MiniMax request construction
# ---------------------------------------------------------------------------

class TestCallLlmProviderMiniMax:
    """Test that _call_llm_provider builds the correct request for MiniMax."""

    def test_uses_default_model_and_base_url(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "MINIMAX")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from MiniMax!"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response) as mock_post:
            from utils.call_llm import _call_llm_provider
            result = _call_llm_provider("test prompt")

        assert result == "Hello from MiniMax!"
        call_args = mock_post.call_args
        url = call_args[0][0]
        payload = call_args[1]["json"]
        headers = call_args[1]["headers"]
        assert "api.minimax.io" in url
        assert payload["model"] == "MiniMax-M2.7"
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_custom_model_overrides_default(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "MINIMAX")
        monkeypatch.setenv("MINIMAX_API_KEY", "key")
        monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "fast"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response) as mock_post:
            from utils.call_llm import _call_llm_provider
            _call_llm_provider("prompt")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "MiniMax-M2.7-highspeed"

    def test_custom_base_url_overrides_default(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "MINIMAX")
        monkeypatch.setenv("MINIMAX_API_KEY", "key")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://custom.endpoint.io")

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response) as mock_post:
            from utils.call_llm import _call_llm_provider
            _call_llm_provider("prompt")

        url = mock_post.call_args[0][0]
        assert url.startswith("https://custom.endpoint.io")

    def test_temperature_clamped_to_provider_range(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "MINIMAX")
        monkeypatch.setenv("MINIMAX_API_KEY", "key")

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response) as mock_post:
            from utils.call_llm import _call_llm_provider
            _call_llm_provider("prompt")

        payload = mock_post.call_args[1]["json"]
        temp = payload["temperature"]
        assert 0 < temp <= 1.0, f"Temperature {temp} out of MiniMax range"

    def test_no_defaults_for_unknown_provider(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "CUSTOM")
        # Without model and base_url, should raise ValueError
        with pytest.raises(ValueError, match="CUSTOM_MODEL"):
            from utils.call_llm import _call_llm_provider
            _call_llm_provider("prompt")


# ---------------------------------------------------------------------------
# call_llm – end-to-end with MiniMax auto-detection
# ---------------------------------------------------------------------------

class TestCallLlmMiniMaxIntegration:
    """Test call_llm() with MiniMax auto-detection (env var only, mocked HTTP)."""

    def test_auto_detect_and_call(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

        # Use a temp cache file so we don't pollute the repo
        monkeypatch.setattr("utils.call_llm.cache_file", str(tmp_path / "cache.json"))

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "MiniMax response"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response):
            from utils.call_llm import call_llm
            result = call_llm("hello", use_cache=False)

        assert result == "MiniMax response"

    def test_cache_works_with_minimax(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

        cache_path = str(tmp_path / "cache.json")
        monkeypatch.setattr("utils.call_llm.cache_file", cache_path)

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "first call"}}]
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("utils.call_llm.requests.post", return_value=mock_response) as mock_post:
            from utils.call_llm import call_llm
            r1 = call_llm("cached prompt", use_cache=True)
            r2 = call_llm("cached prompt", use_cache=True)

        # Second call should use cache, not make another HTTP request
        assert r1 == r2 == "first call"
        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Integration test – real API call (requires MINIMAX_API_KEY)
# ---------------------------------------------------------------------------

_LIVE_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


@pytest.mark.skipif(
    not _LIVE_API_KEY,
    reason="MINIMAX_API_KEY not set – skipping live integration test",
)
class TestMiniMaxLiveIntegration:
    """Live integration tests that call the real MiniMax API."""

    def test_simple_chat_completion(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", _LIVE_API_KEY)
        monkeypatch.setattr("utils.call_llm.cache_file", str(tmp_path / "cache.json"))

        from utils.call_llm import call_llm
        result = call_llm("Reply with exactly: HELLO", use_cache=False)
        assert "HELLO" in result.upper()

    def test_longer_response(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", _LIVE_API_KEY)
        monkeypatch.setattr("utils.call_llm.cache_file", str(tmp_path / "cache.json"))

        from utils.call_llm import call_llm
        result = call_llm(
            "List three colors, one per line. No other text.",
            use_cache=False,
        )
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        assert len(lines) >= 3

    def test_highspeed_model(self, monkeypatch, tmp_path):
        _clean_env(monkeypatch)
        monkeypatch.setenv("MINIMAX_API_KEY", _LIVE_API_KEY)
        monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")
        monkeypatch.setattr("utils.call_llm.cache_file", str(tmp_path / "cache.json"))

        from utils.call_llm import call_llm
        result = call_llm("Say OK", use_cache=False)
        assert len(result) > 0
