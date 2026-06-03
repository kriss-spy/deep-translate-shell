"""Tests for the OpenAI-compatible provider."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from dtrans.models import TranslationResult
from dtrans.providers.openai_compat import OpenAICompatibleProvider


@respx.mock
def test_translate_returns_translation_result() -> None:
    """Provider calls the API and returns a parsed TranslationResult."""
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "translation": "Bonjour",
                                "alternatives": [],
                                "examples": [],
                                "detected_source_language": "en",
                                "translation_phonetics": "",
                            }),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
        system_prompt="You are a translator.",
    )

    result = provider.translate("Hello", source_lang="en", target_lang="fr")

    assert isinstance(result, TranslationResult)
    assert result.translation == "Bonjour"
    assert result.detected_source_language == "en"
    assert route.called


@respx.mock
def test_translate_retries_on_transient_errors() -> None:
    """Retry logic: max 3 retries with exponential backoff on 502 errors."""
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        side_effect=[
            Response(502, text="Bad Gateway"),
            Response(502, text="Bad Gateway"),
            Response(502, text="Bad Gateway"),
            Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({
                                    "translation": "Hola",
                                    "alternatives": [],
                                    "examples": [],
                                    "detected_source_language": "en",
                                    "translation_phonetics": "",
                                }),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ),
        ]
    )

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
        system_prompt="You are a translator.",
    )

    result = provider.translate("Hello", source_lang="en", target_lang="es")

    assert result.translation == "Hola"
    assert route.call_count == 4  # 1 initial + 3 retries


@respx.mock
def test_translate_fails_after_max_retries() -> None:
    """After 3 retries (4 total attempts), a persistent error should raise."""
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=Response(502, text="Bad Gateway")
    )

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
        system_prompt="You are a translator.",
    )

    with pytest.raises(RuntimeError) as exc_info:
        provider.translate("Hello", source_lang="en", target_lang="es")

    msg = str(exc_info.value).lower()
    assert "failed after 3 retries" in msg or "502" in msg or "bad gateway" in msg
    assert route.call_count == 4  # 1 initial + 3 retries


@respx.mock
def test_translate_uses_json_mode() -> None:
    """The request should include response_format set to json_object."""
    captured_request = {}

    def capture_request(request):
        captured_request["body"] = json.loads(request.content)
        return Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "translation": "Ciao",
                                "alternatives": [],
                                "examples": [],
                                "detected_source_language": "en",
                                "translation_phonetics": "",
                            }),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=capture_request)

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
        system_prompt="You are a translator.",
    )

    provider.translate("Hello", source_lang="en", target_lang="it")

    assert captured_request["body"]["response_format"]["type"] == "json_object"


@respx.mock
def test_identify_returns_iso_code() -> None:
    """Provider calls the API and returns a detected ISO-639-1 language code."""
    respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"language": "ja"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
    )

    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
        system_prompt="You are a translator.",
    )

    result = provider.identify("手紙")
    assert result == "ja"
