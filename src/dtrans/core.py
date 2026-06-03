"""Core translation orchestration."""

from __future__ import annotations

from dtrans.config import Config, ProviderConfig
from dtrans.models import TranslationResult
from dtrans.prompts import DEFAULT_SYSTEM_PROMPT
from dtrans.providers.gemini import GeminiProvider
from dtrans.providers.openai_compat import OpenAICompatibleProvider


class Translator:
    """Orchestrates translation requests across configured LLM providers."""

    def __init__(self, config: Config, provider_name: str | None = None) -> None:
        self.config = config
        self.provider_name = provider_name or config.default_provider
        if not self.provider_name:
            raise ValueError("No default_provider set in configuration.")

        provider_cfg = config.providers.get(self.provider_name)
        if provider_cfg is None:
            raise ValueError(f"Provider '{self.provider_name}' not found in configuration.")

        self.provider = self._build_provider(provider_cfg)

    @staticmethod
    def _build_provider(cfg: ProviderConfig) -> OpenAICompatibleProvider | GeminiProvider:
        """Instantiate the appropriate provider for the given config."""
        system_prompt = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT

        if cfg.provider_type == "gemini":
            return GeminiProvider(
                api_key=cfg.api_key,
                model=cfg.model,
                system_prompt=system_prompt,
            )

        # Default: OpenAI-compatible provider (covers OpenAI, DeepSeek, custom endpoints).
        return OpenAICompatibleProvider(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            system_prompt=system_prompt,
        )

    def translate(
        self,
        text: str,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> TranslationResult:
        """Translate TEXT and return a structured result."""
        return self.provider.translate(text, source_lang=source_lang, target_lang=target_lang)

    def identify(self, text: str) -> str:
        """Detect the source language of TEXT and return its ISO code."""
        return self.provider.identify(text)
