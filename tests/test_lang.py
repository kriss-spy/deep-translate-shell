"""Tests for language syntax parsing and utilities."""

from __future__ import annotations

import locale

import pytest

from dtrans.lang import get_default_target, parse_lang_spec


class TestParseLangSpec:
    def test_auto_detect_to_target(self) -> None:
        assert parse_lang_spec(":fr") == (None, "fr")

    def test_source_to_default_target(self) -> None:
        assert parse_lang_spec("ja:") == ("ja", None)

    def test_source_to_target(self) -> None:
        assert parse_lang_spec("en:fr") == ("en", "fr")

    def test_multiple_targets_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            parse_lang_spec(":zh+ja")
        assert "multiple" in str(exc_info.value).lower()

    def test_plain_text_returns_none(self) -> None:
        """A string without a colon is not a language spec."""
        assert parse_lang_spec("hello") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_lang_spec("") is None

    def test_target_with_hyphen(self) -> None:
        assert parse_lang_spec("zh-CN") is None  # no colon, not a spec
        assert parse_lang_spec(":zh-CN") == (None, "zh-cn")


class TestGetDefaultTarget:
    def test_returns_locale_language_code(self) -> None:
        code = get_default_target()
        assert isinstance(code, str)
        assert len(code) >= 2

    def test_falls_back_to_en_on_locale_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(locale, "getlocale", lambda _=None: (None, None))
        assert get_default_target() == "en"
