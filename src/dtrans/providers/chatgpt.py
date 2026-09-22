"""ChatGPT subscription provider backed by the authenticated Codex CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from dtrans.models import TranslationResult
from dtrans.providers.base import BaseProvider


class ChatGPTProvider(BaseProvider):
    """Run structured translation tasks through a ChatGPT-authenticated Codex CLI."""

    name = "chatgpt"

    def __init__(self, model: str | None, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt

    @staticmethod
    def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
        """Return a JSON schema accepted by Codex structured output."""
        schema = model.model_json_schema()

        def make_strict(node: Any) -> None:
            if isinstance(node, dict):
                node.pop("default", None)
                if node.get("type") == "object" and "properties" in node:
                    node["additionalProperties"] = False
                    node["required"] = list(node["properties"])
                for value in node.values():
                    make_strict(value)
            elif isinstance(node, list):
                for value in node:
                    make_strict(value)

        make_strict(schema)
        return schema

    def _run_codex(self, prompt: str, schema: dict[str, Any]) -> str:
        codex = shutil.which("codex")
        if codex is None:
            raise RuntimeError(
                "Codex CLI is required for ChatGPT subscription access. "
                "Install it, then run 'dtrans-auth login'."
            )

        with tempfile.TemporaryDirectory(prefix="dtrans-chatgpt-") as temp_dir:
            schema_path = Path(temp_dir) / "output-schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                codex,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")

            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    cwd=temp_dir,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=180,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("ChatGPT translation timed out after 180 seconds.") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "Codex exited without an error message."
            raise RuntimeError(
                f"ChatGPT subscription request failed: {detail} "
                "Run 'dtrans-auth status' to check authentication."
            )
        return completed.stdout.strip()

    def translate(
        self,
        text: str,
        source_lang: str | None,
        target_lang: str | None,
    ) -> TranslationResult:
        """Translate text through an ephemeral Codex session."""
        source_desc = source_lang or "auto-detect"
        target_desc = target_lang or "the locale-appropriate target language"
        prompt = (
            f"{self.system_prompt}\n\n"
            "Perform only this translation task. Do not inspect files, call tools, or execute "
            "commands. Return only the JSON object required by the output schema.\n"
            f"Source language: {source_desc}\n"
            f"Target language: {target_desc}\n"
            "Text to translate follows after the delimiter:\n---\n"
            f"{text}"
        )
        raw_json = self._run_codex(prompt, self._strict_schema(TranslationResult))
        try:
            return TranslationResult.model_validate_json(raw_json)
        except ValidationError as exc:
            raise RuntimeError("ChatGPT returned an invalid structured translation.") from exc

    def identify(self, text: str) -> str:
        """Detect the source language through an ephemeral Codex session."""
        schema = {
            "type": "object",
            "properties": {"language": {"type": "string"}},
            "required": ["language"],
            "additionalProperties": False,
        }
        prompt = (
            "Detect the language of the text after the delimiter. Do not inspect files, call "
            "tools, or execute commands. Return its ISO-639-1 code in the required JSON object."
            f"\n---\n{text}"
        )
        raw_json = self._run_codex(prompt, schema)
        try:
            language = str(json.loads(raw_json)["language"]).lower()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("ChatGPT returned an invalid language identification.") from exc
        if not language:
            raise RuntimeError("ChatGPT did not return a language code.")
        return language
