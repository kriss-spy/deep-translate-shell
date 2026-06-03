"""Click CLI entry point for dtrans."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from dtrans import __version__
from dtrans.config import ConfigError, load_config
from dtrans.core import Translator
from dtrans.lang import get_default_target, parse_lang_spec
from dtrans.render import get_spinner_text, render_verbose

# Exit codes (per issue #3)
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_BAD_USAGE = 2
EXIT_NETWORK_ERROR = 3
EXIT_CONFIG_ERROR = 4

_MAX_INPUT_LENGTH = 2000


def _is_tty() -> bool:
    """Return True if stdout is connected to a terminal."""
    return sys.stdout.isatty()


def _parse_args(text_args: tuple[str, ...]) -> tuple[str | None, str | None, str]:
    """Parse positional args into (source_lang, target_lang, actual_text).

    If the first arg is a colon language spec, extract languages and join
    the remainder as text. Otherwise, join all args as text.
    """
    if not text_args:
        return None, None, ""

    # Try to parse the first arg as a language spec
    parsed = parse_lang_spec(text_args[0])
    if parsed is not None:
        source_lang, target_lang = parsed
        actual_text = " ".join(text_args[1:])
        return source_lang, target_lang, actual_text

    # No colon syntax — join everything as the text to translate
    return None, None, " ".join(text_args)


@click.command()
@click.argument("text", nargs=-1)
@click.option(
    "--from",
    "source_lang",
    help="Source language code (e.g. 'en', 'ja'). Omit to auto-detect.",
)
@click.option(
    "--to",
    "target_lang",
    help="Target language code (e.g. 'fr', 'zh'). Defaults to system locale.",
)
@click.option(
    "--provider",
    help="Override the default provider from the config file.",
)
@click.option(
    "--brief/--verbose",
    "brief",
    default=None,
    help="Brief mode prints only the translation. Verbose is the default in a TTY.",
)
@click.option(
    "--identify",
    "identify",
    is_flag=True,
    help="Identify the source language and exit.",
)
@click.version_option(version=__version__, prog_name="dtrans")
@click.pass_context
def main(
    ctx: click.Context,
    text: tuple[str, ...],
    source_lang: str | None,
    target_lang: str | None,
    provider: str | None,
    brief: bool | None,
    identify: bool,
) -> None:
    """A modern command-line translator powered by LLMs.

    Translate TEXT from one language to another using configurable LLM providers.

    \b
    Examples:
      dtrans "Hello, world!"
      dtrans --to fr "Hello, world!"
      dtrans en:fr "Hello, world!"
      dtrans :ja "Hello, world!"
    """
    # Determine brief mode: explicit flag wins, otherwise auto-detect TTY.
    is_brief = brief if brief is not None else not _is_tty()

    # Read text from stdin if piped and no positional argument given.
    if not text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        text = (stdin_text,) if stdin_text else ()

    if not text:
        click.echo(ctx.get_help())
        sys.exit(EXIT_SUCCESS)

    # Parse colon language syntax from positional args
    try:
        parsed_source, parsed_target, actual_text = _parse_args(text)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(EXIT_BAD_USAGE)

    # CLI flags override colon syntax
    final_source = source_lang or parsed_source
    final_target = target_lang or parsed_target

    # Default target from locale if still not set
    if final_target is None and not identify:
        final_target = get_default_target()

    # Validate input length
    if len(actual_text) > _MAX_INPUT_LENGTH:
        click.echo(
            f"Error: Input is too long ({len(actual_text)} characters). "
            f"Maximum allowed is {_MAX_INPUT_LENGTH} characters.",
            err=True,
        )
        sys.exit(EXIT_BAD_USAGE)

    try:
        cfg = load_config()
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    try:
        translator = Translator(cfg, provider_name=provider)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(EXIT_CONFIG_ERROR)

    if identify:
        try:
            detected = translator.identify(actual_text)
        except Exception as exc:
            click.echo(f"Error: Language identification failed: {exc}", err=True)
            sys.exit(EXIT_NETWORK_ERROR)
        click.echo(detected)
        sys.exit(EXIT_SUCCESS)

    console = Console()
    try:
        if is_brief:
            # Brief mode: no spinner, just print translation
            result = translator.translate(
                actual_text, source_lang=final_source, target_lang=final_target
            )
            click.echo(result.translation)
        else:
            # Verbose mode: show spinner during API call, then rich layout
            with console.status(get_spinner_text(translator.provider_name)):
                result = translator.translate(
                    actual_text, source_lang=final_source, target_lang=final_target
                )
            render_verbose(
                result,
                source_lang=final_source,
                target_lang=final_target,
                console=console,
            )
    except Exception as exc:
        click.echo(f"Error: Translation failed: {exc}", err=True)
        sys.exit(EXIT_NETWORK_ERROR)


if __name__ == "__main__":
    main()
