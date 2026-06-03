"""Base provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dtrans.models import TranslationResult


class BaseProvider(ABC):
    """Abstract base class for translation providers."""

    name: str

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str | None,
    ) -> TranslationResult:
        """Translate text and return a structured result."""
        ...

    @abstractmethod
    def identify(self, text: str) -> str:
        """Detect the source language of the text."""
        ...
