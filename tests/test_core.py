"""Tests for core translation orchestration and provider dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dtrans.config import Config, ProviderConfig
from dtrans.core import Translator
from dtrans.models import TranslationResult
from dtrans.providers.gemini import GeminiProvider
from dtrans.providers.openai_compat import OpenAICompatibleProvider


def _make_config(
    provider_type: str = "openai",
    system_prompt: str | None = None,
) -> Config:
    return Config(
        default_provider="test",
        providers={
            "test": ProviderConfig(
                api_key="sk-test",
                model="gpt-4",
                provider_type=provider_type,
                system_prompt=system_prompt,
            )
        },
    )


def _make_result(translation: str = "Bonjour") -> TranslationResult:
    return TranslationResult(
        translation=translation,
        alternatives=[],
        examples=[],
        detected_source_language="en",
    )


class TestProviderDispatch:
    def test_builds_openai_provider_by_default(self) -> None:
        cfg = _make_config(provider_type="openai")
        translator = Translator(cfg)
        assert isinstance(translator.provider, OpenAICompatibleProvider)

    def test_builds_gemini_provider_when_configured(self) -> None:
        cfg = _make_config(provider_type="gemini")
        translator = Translator(cfg)
        assert isinstance(translator.provider, GeminiProvider)

    def test_uses_custom_system_prompt_when_set(self) -> None:
        custom_prompt = "You are a pirate translator."
        cfg = _make_config(provider_type="openai", system_prompt=custom_prompt)
        translator = Translator(cfg)
        assert translator.provider.system_prompt == custom_prompt

    def test_uses_default_system_prompt_when_not_set(self) -> None:
        cfg = _make_config(provider_type="openai")
        translator = Translator(cfg)
        assert "translation" in translator.provider.system_prompt.lower()
        assert "alternatives" in translator.provider.system_prompt.lower()


class TestTranslate:
    def test_translate_delegates_to_provider(self) -> None:
        cfg = _make_config()
        translator = Translator(cfg)
        with patch.object(
            translator.provider,
            "translate",
            return_value=_make_result("Hola"),
        ) as mock_translate:
            result = translator.translate("hello", source_lang="en", target_lang="es")

        assert result.translation == "Hola"
        mock_translate.assert_called_once_with("hello", source_lang="en", target_lang="es")

    def test_identify_delegates_to_provider(self) -> None:
        cfg = _make_config()
        translator = Translator(cfg)
        with patch.object(
            translator.provider,
            "identify",
            return_value="ja",
        ) as mock_identify:
            result = translator.identify("手紙")

        assert result == "ja"
        mock_identify.assert_called_once_with("手紙")
