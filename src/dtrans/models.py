"""Pydantic models for structured LLM responses and configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Alternative(BaseModel):
    """An alternative translation with explanatory notes."""

    text: str = Field(description="The alternative translation text.")
    note: str = Field(description="Brief note on tone/register (e.g. 'formal', 'casual').")
    comparison: str = Field(description="Comparison to the main translation.")


class Example(BaseModel):
    """An example sentence pair showing usage."""

    source: str = Field(description="Example in the source language.")
    target: str = Field(description="Example in the target language.")
    context: str = Field(description="Context where this phrasing is used.")


class TranslationResult(BaseModel):
    """Structured response from an LLM translator."""

    translation: str = Field(description="The primary translated text.")
    alternatives: list[Alternative] = Field(
        default_factory=list,
        description="Alternative translations with notes and comparisons.",
    )
    examples: list[Example] = Field(
        default_factory=list,
        description="Example sentence pairs with context.",
    )
    detected_source_language: str = Field(
        description="ISO-639-1 code of the detected source language."
    )
    translation_phonetics: str = Field(
        default="",
        description="Phonetic notation or pronunciation guide for the translation.",
    )
