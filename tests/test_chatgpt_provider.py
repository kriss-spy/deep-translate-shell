"""Tests for the ChatGPT subscription provider."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from dtrans.models import TranslationResult
from dtrans.providers.chatgpt import ChatGPTProvider


def test_translate_uses_ephemeral_codex_session_with_structured_output() -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({
            "translation": "Minors and cofactors",
            "alternatives": [],
            "examples": [],
            "detected_source_language": "zh",
            "translation_phonetics": "",
        }),
        stderr="",
    )
    provider = ChatGPTProvider(model=None, system_prompt="You are a translator.")

    with (
        patch("dtrans.providers.chatgpt.shutil.which", return_value="/usr/bin/codex"),
        patch("dtrans.providers.chatgpt.subprocess.run", return_value=completed) as run,
    ):
        result = provider.translate("余子式和代数余子式", None, "en")

    assert isinstance(result, TranslationResult)
    assert result.translation == "Minors and cofactors"
    command = run.call_args.args[0]
    assert command[:2] == ["/usr/bin/codex", "exec"]
    assert "--ephemeral" in command
    assert "--output-schema" in command
    assert run.call_args.kwargs["input"].endswith("余子式和代数余子式")
