"""Rich terminal rendering for verbose translation output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dtrans.models import TranslationResult


def render_verbose(
    result: TranslationResult,
    source_lang: str | None,
    target_lang: str | None,
    console: Console | None = None,
) -> None:
    """Render a rich, verbose translation result to the console."""
    console = console or Console()

    # Main translation panel with phonetics
    translation_content = result.translation
    if result.translation_phonetics:
        translation_content += f"\n\n[dim]{result.translation_phonetics}[/dim]"

    console.print(
        Panel(
            translation_content,
            title="Translation",
            border_style="green",
        )
    )

    # Alternatives table
    if result.alternatives:
        alt_table = Table(title="Alternatives", show_header=True, header_style="bold magenta")
        alt_table.add_column("Alternative", style="cyan")
        alt_table.add_column("Note", style="yellow")
        alt_table.add_column("Comparison", style="dim")

        for alt in result.alternatives:
            alt_table.add_row(alt.text, alt.note, alt.comparison)

        console.print(alt_table)

    # Examples table
    if result.examples:
        ex_table = Table(title="Examples", show_header=True, header_style="bold blue")
        ex_table.add_column("Source", style="cyan")
        ex_table.add_column("Target", style="green")
        ex_table.add_column("Context", style="dim")

        for ex in result.examples:
            ex_table.add_row(ex.source, ex.target, ex.context)

        console.print(ex_table)

    # Footer panel with language pair
    src = source_lang or result.detected_source_language or "auto"
    tgt = target_lang or "?"
    console.print(
        Panel(
            f"[bold]{src}[/bold] → [bold]{tgt}[/bold]",
            title="Language Pair",
            border_style="blue",
        )
    )


def get_spinner_text(provider_name: str) -> str:
    """Return the spinner status text for the given provider."""
    return f"Translating with {provider_name}..."
