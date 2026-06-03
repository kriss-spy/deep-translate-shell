"""Tests for the CLI entry point."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from dtrans.cli import main
from dtrans.config import Config, ConfigError, ProviderConfig
from dtrans.models import Alternative, Example, TranslationResult


def _make_config() -> Config:
    return Config(
        default_provider="test",
        providers={
            "test": ProviderConfig(
                api_key="sk-test",
                model="gpt-4",
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


class TestCliBasic:
    def test_translates_and_prints_text(self) -> None:
        runner = CliRunner()
        with (
            patch("dtrans.cli.load_config", return_value=_make_config()),
            patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("Bonjour"),
            ),
        ):
            result = runner.invoke(main, ["Hello"])

        assert result.exit_code == 0
        assert "Bonjour" in result.output

    def test_brief_mode_prints_only_translation(self) -> None:
        runner = CliRunner()
        with (
            patch("dtrans.cli.load_config", return_value=_make_config()),
            patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("Hola"),
            ),
        ):
            result = runner.invoke(main, ["--brief", "Hello"])

        assert result.exit_code == 0
        assert result.output.strip() == "Hola"
        # No panels, no ANSI, no extra fluff
        assert "┌" not in result.output
        assert "└" not in result.output

    def test_non_tty_stdout_auto_brief(self) -> None:
        """When stdout is not a TTY, output should be plain text only."""
        runner = CliRunner()
        with (
            patch("dtrans.cli.load_config", return_value=_make_config()),
            patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("Ciao"),
            ),
        ):
            # CliRunner already simulates a non-TTY stdout
            result = runner.invoke(main, ["Hello"])

        assert result.exit_code == 0
        assert result.output.strip() == "Ciao"

    def test_from_and_to_flags(self) -> None:
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            mock_translate = patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("Salut"),
            )
            with mock_translate as mt:
                result = runner.invoke(main, ["--from", "en", "--to", "fr", "Hi"])

        assert result.exit_code == 0
        assert "Salut" in result.output
        # Verify the provider was called with the correct languages
        call_kwargs = mt.call_args.kwargs if mt.call_args else {}
        assert call_kwargs.get("source_lang") == "en"
        assert call_kwargs.get("target_lang") == "fr"

    def test_brief_and_to_combined(self) -> None:
        """Acceptance criterion: dtrans --brief --to fr 'hello world' prints translation."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            mock_translate = patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("bonjour le monde"),
            )
            with mock_translate as mt:
                result = runner.invoke(main, ["--brief", "--to", "fr", "hello world"])

        assert result.exit_code == 0
        assert result.output.strip() == "bonjour le monde"
        call_kwargs = mt.call_args.kwargs if mt.call_args else {}
        assert call_kwargs.get("target_lang") == "fr"

    def test_missing_config_friendly_error(self) -> None:
        runner = CliRunner()
        with patch("dtrans.cli.load_config", side_effect=ConfigError("Config not found")):
            result = runner.invoke(main, ["Hello"])

        assert result.exit_code != 0
        assert "config" in result.output.lower() or "error" in result.output.lower()


class TestCliColonSyntax:
    def test_colon_syntax_auto_detect_to_target(self) -> None:
        """dtrans :fr 'hello world' -> auto-detect source, target French."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            mock_translate = patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("bonjour le monde"),
            )
            with mock_translate as mt:
                result = runner.invoke(main, [":fr", "hello world"])

        assert result.exit_code == 0
        assert "bonjour le monde" in result.output
        call_kwargs = mt.call_args.kwargs if mt.call_args else {}
        assert call_kwargs.get("source_lang") is None
        assert call_kwargs.get("target_lang") == "fr"

    def test_colon_syntax_source_and_target(self) -> None:
        """dtrans en:fr 'hello' -> source English, target French."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            mock_translate = patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("bonjour"),
            )
            with mock_translate as mt:
                result = runner.invoke(main, ["en:fr", "hello"])

        assert result.exit_code == 0
        assert "bonjour" in result.output
        call_kwargs = mt.call_args.kwargs if mt.call_args else {}
        assert call_kwargs.get("source_lang") == "en"
        assert call_kwargs.get("target_lang") == "fr"

    def test_colon_syntax_source_only(self) -> None:
        """dtrans ja: 'hello' -> source Japanese, target from locale."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            mock_translate = patch(
                "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
                return_value=_make_result("hello"),
            )
            with mock_translate as mt:
                result = runner.invoke(main, ["ja:", "hello"])

        assert result.exit_code == 0
        call_kwargs = mt.call_args.kwargs if mt.call_args else {}
        assert call_kwargs.get("source_lang") == "ja"
        # target should fall back to default locale
        assert call_kwargs.get("target_lang") is not None

    def test_colon_syntax_rejects_multiple_targets(self) -> None:
        """dtrans :zh+ja 'hello' -> exit code 2 with clear message."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            result = runner.invoke(main, [":zh+ja", "hello"])

        assert result.exit_code == 2
        assert "multiple" in result.output.lower() or "not supported" in result.output.lower()


class TestCliInputValidation:
    def test_input_too_long_exits_with_code_2(self) -> None:
        """Input >2000 characters should be rejected with exit code 2."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()):
            long_text = "x" * 2001
            result = runner.invoke(main, [long_text])

        assert result.exit_code == 2
        assert "2000" in result.output or "too long" in result.output.lower()


class TestCliExitCodes:
    def test_config_error_exits_with_code_4(self) -> None:
        """Missing or invalid config should exit with code 4."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", side_effect=ConfigError("not found")):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 4

    def test_network_error_exits_with_code_3(self) -> None:
        """API/network failure should exit with code 3."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            side_effect=RuntimeError("Connection refused"),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 3

    def test_success_exits_with_code_0(self) -> None:
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=_make_result("hi"),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 0


class TestCliErrorMessages:
    def test_missing_api_key_shows_friendly_error(self) -> None:
        """Authentication error should suggest checking the config file."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            side_effect=RuntimeError(
                "Authentication failed: Incorrect API key provided. "
                "Please check your API key in ~/.config/dtrans/config.toml"
            ),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 3
        assert "api key" in result.output.lower() or "config" in result.output.lower()

    def test_invalid_json_shows_friendly_error(self) -> None:
        """Invalid JSON from LLM should suggest checking model/system prompt."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            side_effect=RuntimeError(
                "Invalid JSON response from the model. "
                "Please check your model and system prompt configuration."
            ),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 3
        assert (
            "json" in result.output.lower()
            or "model" in result.output.lower()
            or "prompt" in result.output.lower()
        )

    def test_rate_limit_shows_friendly_error(self) -> None:
        """Rate limit error should be clearly communicated."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            side_effect=RuntimeError(
                "Rate limit exceeded. Please wait a moment and try again."
            ),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 3
        assert "rate" in result.output.lower() or "limit" in result.output.lower()

    def test_network_failure_shows_retry_count(self) -> None:
        """Network failure after retries should mention retry count."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            side_effect=RuntimeError("Failed after 3 retries"),
        ):
            result = runner.invoke(main, ["hello"])

        assert result.exit_code == 3
        # The error message should mention retries in some form
        assert "retri" in result.output.lower() or "failed" in result.output.lower()

    def test_invalid_provider_name_red_error(self) -> None:
        """dtrans --provider fake 'hello' -> clear error about invalid provider."""
        runner = CliRunner()
        cfg = Config(
            default_provider="real",
            providers={
                "real": ProviderConfig(api_key="sk-test", model="gpt-4")
            },
        )
        with patch("dtrans.cli.load_config", return_value=cfg):
            result = runner.invoke(main, ["--provider", "fake", "hello"])

        assert result.exit_code == 4
        assert "fake" in result.output.lower()
        assert "not found" in result.output.lower() or "provider" in result.output.lower()


class TestCliIdentify:
    def test_identify_prints_iso_code(self) -> None:
        """dtrans --identify '手紙' prints 'ja' and exits 0."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.identify",
            return_value="ja",
        ):
            result = runner.invoke(main, ["--identify", "手紙"])

        assert result.exit_code == 0
        assert result.output.strip() == "ja"

    def test_identify_brief_pipe_aware(self) -> None:
        """Identify output should be plain text only, suitable for piping."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.identify",
            return_value="zh",
        ):
            result = runner.invoke(main, ["--identify", "你好"])

        assert result.exit_code == 0
        assert result.output.strip() == "zh"
        assert "┌" not in result.output
        assert "Translation" not in result.output


class TestCliVerbose:
    def _make_verbose_result(self) -> TranslationResult:
        return TranslationResult(
            translation="courir",
            translation_phonetics="/ku.ʁiʁ/",
            alternatives=[
                Alternative(
                    text="se précipiter",
                    note="formal, urgent haste",
                    comparison="More formal than 'courir'; implies rushing.",
                ),
                Alternative(
                    text="foncer",
                    note="colloquial",
                    comparison="Very casual; implies charging ahead.",
                ),
            ],
            examples=[
                Example(
                    source="I run every morning.",
                    target="Je cours tous les matins.",
                    context="Daily exercise.",
                ),
                Example(
                    source="Run for your life!",
                    target="Cours pour sauver ta vie !",
                    context="Dangerous situations.",
                ),
            ],
            detected_source_language="en",
        )

    def test_verbose_renders_translation_panel(self) -> None:
        """--verbose shows translation and phonetics in a panel."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ):
            result = runner.invoke(main, ["--verbose", "run"])

        assert result.exit_code == 0
        assert "courir" in result.output
        assert "/ku.ʁiʁ/" in result.output

    def test_verbose_renders_alternatives_table(self) -> None:
        """--verbose shows alternatives with text, note, and comparison."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ):
            result = runner.invoke(main, ["--verbose", "run"])

        assert result.exit_code == 0
        assert "se précipiter" in result.output
        assert "formal, urgent haste" in result.output
        assert "More formal than 'courir'" in result.output
        assert "foncer" in result.output
        assert "colloquial" in result.output

    def test_verbose_renders_examples_table(self) -> None:
        """--verbose shows examples with source, target, and context."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ):
            result = runner.invoke(main, ["--verbose", "run"])

        assert result.exit_code == 0
        assert "I run every morning." in result.output
        assert "Je cours tous les matins." in result.output
        assert "Daily exercise." in result.output
        assert "Run for your life!" in result.output
        assert "Cours pour sauver ta vie" in result.output
        assert "Dangerous situations." in result.output

    def test_verbose_renders_footer_language_pair(self) -> None:
        """--verbose footer shows detected source -> target language."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ):
            result = runner.invoke(main, ["--verbose", "--to", "fr", "run"])

        assert result.exit_code == 0
        assert "en" in result.output
        assert "fr" in result.output

    def test_tty_defaults_to_verbose(self) -> None:
        """When stdout is a TTY, default mode is verbose."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ), patch("dtrans.cli._is_tty", return_value=True):
            result = runner.invoke(main, ["run"])

        assert result.exit_code == 0
        assert "courir" in result.output
        # Verbose output should contain the panel/table markers (Rich box chars)
        # or at least more content than just the translation
        assert "se précipiter" in result.output

    def test_pipe_still_yields_brief(self) -> None:
        """Piping stdout still auto-switches to brief (non-TTY)."""
        runner = CliRunner()
        with patch("dtrans.cli.load_config", return_value=_make_config()), patch(
            "dtrans.providers.openai_compat.OpenAICompatibleProvider.translate",
            return_value=self._make_verbose_result(),
        ):
            result = runner.invoke(main, ["run"])

        assert result.exit_code == 0
        assert result.output.strip() == "courir"
        assert "se précipiter" not in result.output
