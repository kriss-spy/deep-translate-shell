"""Tests for ChatGPT subscription authentication commands."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from click.testing import CliRunner

from dtrans.auth import main


def test_status_delegates_to_codex_authentication() -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0)
    runner = CliRunner()

    with (
        patch("dtrans.auth.shutil.which", return_value="/usr/bin/codex"),
        patch("dtrans.auth.subprocess.run", return_value=completed) as run,
    ):
        result = runner.invoke(main, ["status"])

    assert result.exit_code == 0
    run.assert_called_once_with(["/usr/bin/codex", "login", "status"], check=False)
