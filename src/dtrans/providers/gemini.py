"""Gemini provider using the google-genai SDK."""

from __future__ import annotations

import json

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from dtrans.models import TranslationResult
from dtrans.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider for Google's Gemini models via the google-genai SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, system_prompt: str) -> None:
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30000,  # 30 seconds (milliseconds)
                client_args={"http2": False},
            ),
        )

    def _build_config(self) -> types.GenerateContentConfig:
        """Build generation config with JSON schema enforcement."""
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "translation": types.Schema(type=types.Type.STRING),
                "alternatives": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "text": types.Schema(type=types.Type.STRING),
                            "note": types.Schema(type=types.Type.STRING),
                            "comparison": types.Schema(type=types.Type.STRING),
                        },
                    ),
                ),
                "examples": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "source": types.Schema(type=types.Type.STRING),
                            "target": types.Schema(type=types.Type.STRING),
                            "context": types.Schema(type=types.Type.STRING),
                        },
                    ),
                ),
                "detected_source_language": types.Schema(type=types.Type.STRING),
                "translation_phonetics": types.Schema(type=types.Type.STRING),
            },
        )
        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),  # 1 initial + 3 retries
        wait=wait_exponential(multiplier=1, min=1, max=4),  # 1s -> 2s -> 4s
        reraise=True,
    )
    def _generate_content(
        self, prompt: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Call the Gemini API with transient-error retries."""
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

    def translate(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str | None,
    ) -> TranslationResult:
        """Translate text using the Gemini API."""
        source_desc = f"from {source_lang}" if source_lang else "(auto-detect source)"
        target_desc = f"to {target_lang}" if target_lang else "(default target)"
        prompt = f"Translate the following text {source_desc} {target_desc}:\n\n{text}"

        try:
            response = self._generate_content(prompt, self._build_config())
        except Exception as exc:
            raise RuntimeError(f"Translation request failed: {exc}") from exc

        if not response.text:
            raise RuntimeError("Gemini returned empty content.")

        return TranslationResult.model_validate_json(response.text)

    def identify(self, text: str) -> str:
        """Detect the source language of the text and return an ISO-639-1 code."""
        prompt = (
            "Detect the language of this text and return ONLY a JSON object with a "
            f"single key 'language' containing the ISO-639-1 code:\n\n{text}"
        )
        config = types.GenerateContentConfig(
            system_instruction="You are a language detection assistant.",
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={"language": types.Schema(type=types.Type.STRING)},
            ),
        )

        try:
            response = self._generate_content(prompt, config)
        except Exception as exc:
            raise RuntimeError(f"Identification request failed: {exc}") from exc

        if not response.text:
            raise RuntimeError("Gemini returned empty content.")

        parsed = json.loads(response.text)
        lang = str(parsed.get("language", "")).lower()
        if not lang:
            raise RuntimeError("Gemini did not return a language code.")
        return lang
