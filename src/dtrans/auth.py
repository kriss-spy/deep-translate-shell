"""Credential-safe ChatGPT authentication commands backed by Codex CLI."""

from __future__ import annotations

import shutil
import subprocess

import click


def _run_codex_auth(args: list[str]) -> None:
    codex = shutil.which("codex")
    if codex is None:
        raise click.ClickException(
            "Codex CLI is not installed. Install it from https://chatgpt.com/codex/."
        )
    completed = subprocess.run([codex, *args], check=False)
    if completed.returncode != 0:
        raise click.exceptions.Exit(completed.returncode)


@click.group()
def main() -> None:
    """Manage the ChatGPT login used by dtrans."""


@main.command()
@click.option(
    "--device-auth",
    is_flag=True,
    help="Use device-code authentication for remote or headless systems.",
)
def login(device_auth: bool) -> None:
    """Sign in with a ChatGPT subscription through Codex CLI."""
    args = ["login"]
    if device_auth:
        args.append("--device-auth")
    _run_codex_auth(args)


@main.command()
def status() -> None:
    """Show the active Codex authentication method."""
    _run_codex_auth(["login", "status"])


@main.command()
def logout() -> None:
    """Remove the cached Codex authentication."""
    _run_codex_auth(["logout"])


if __name__ == "__main__":
    main()
