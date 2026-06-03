"""OpenAI-compatible provider (covers OpenAI and DeepSeek)."""

from __future__ import annotations

import json

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dtrans.models import TranslationResult
from dtrans.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, DeepSeek, custom endpoints)."""

    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str | None, model: str, system_prompt: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    @retry(
        retry=retry_if_exception_type((APIConnectionError, InternalServerError, RateLimitError)),
        stop=stop_after_attempt(4),  # 1 initial + 3 retries
        wait=wait_exponential(multiplier=1, min=1, max=4),  # 1s -> 2s -> 4s
        reraise=True,
    )
    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request and return the assistant's content."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return content

    @staticmethod
    def _handle_api_error(exc: Exception) -> RuntimeError:
        """Convert low-level API exceptions into friendly, actionable errors."""
        if isinstance(exc, AuthenticationError):
            return RuntimeError(
                "Authentication failed: the API key appears to be invalid or missing. "
                "Please check your API key in ~/.config/dtrans/config.toml."
            )
        if isinstance(exc, RateLimitError):
            return RuntimeError(
                "Rate limit exceeded: the API provider is throttling requests. "
                "Please wait a moment and try again."
            )
        if isinstance(exc, APIConnectionError):
            return RuntimeError(
                "Network error: unable to reach the API. "
                f"Failed after 3 retries ({exc})"
            )
        if isinstance(exc, InternalServerError):
            return RuntimeError(
                f"API server error: {exc}. "
                "Failed after 3 retries. The provider may be temporarily unavailable."
            )
        if isinstance(exc, APIError):
            return RuntimeError(f"API error: {exc}")
        return RuntimeError(f"Translation request failed: {exc}")

    @staticmethod
    def _handle_validation_error(exc: ValidationError) -> RuntimeError:
        """Convert Pydantic validation errors into friendly, actionable errors."""
        return RuntimeError(
            "Invalid JSON response from the model: the output did not match the expected schema. "
            "Please check your model and system prompt configuration."
        )

    def translate(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str | None,
    ) -> TranslationResult:
        """Translate text using the OpenAI-compatible API."""
        source_desc = f"from {source_lang}" if source_lang else "(auto-detect source)"
        target_desc = f"to {target_lang}" if target_lang else "(default target)"

        user_prompt = (
            f"Translate the following text {source_desc} {target_desc}:\n\n{text}"
        )

        schema_prompt = (
            "Return a JSON object strictly matching this schema:\n"
            + json.dumps(TranslationResult.model_json_schema(), indent=2)
        )

        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{schema_prompt}"},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_json = self._chat_completion(messages)
        except Exception as exc:
            raise self._handle_api_error(exc) from exc

        try:
            return TranslationResult.model_validate_json(raw_json)
        except ValidationError as exc:
            raise self._handle_validation_error(exc) from exc

    def identify(self, text: str) -> str:
        """Detect the source language of the text and return an ISO-639-1 code."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language detection assistant. "
                    "Return ONLY a JSON object with a single key 'language' containing "
                    "the ISO-639-1 code of the language (e.g. 'en', 'ja', 'zh')."
                ),
            },
            {
                "role": "user",
                "content": f"Detect the language of this text:\n\n{text}",
            },
        ]

        try:
            raw_json = self._chat_completion(messages)
        except Exception as exc:
            raise self._handle_api_error(exc) from exc

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid JSON response from the model during language identification. "
                "Please check your model and system prompt configuration."
            ) from exc

        lang = parsed.get("language", "").lower()
        if not lang:
            raise RuntimeError("LLM did not return a language code.")
        return lang
