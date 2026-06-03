"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import click
from pydantic import BaseModel, Field, ValidationError

DEFAULT_CONFIG_PATHS = [
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "dtrans" / "config.toml",
    Path.home() / ".config" / "dtrans" / "config.toml",
]


class ConfigError(click.ClickException):
    """Raised when configuration is missing, malformed, or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Configuration error: {message}")


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str = Field(description="API key for the provider.")
    model: str = Field(description="Model name to use for translation.")
    base_url: str | None = Field(default=None, description="Custom base URL for the API.")
    provider_type: str = Field(
        default="openai",
        description="Provider SDK type: 'openai' or 'gemini'.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for this provider.",
    )


class Config(BaseModel):
    """Top-level configuration for dtrans."""

    default_provider: str = Field(description="Name of the default provider to use.")
    providers: dict[str, ProviderConfig] = Field(description="Map of provider name to config.")


def find_config_file() -> Path | None:
    """Return the first existing config file path, or None."""
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def load_config(path: Path | None = None) -> Config:
    """Load and validate the TOML configuration.

    Raises ConfigError if the file is missing, malformed, or invalid.
    """
    config_path = path or find_config_file()
    if config_path is None:
        raise ConfigError(
            f"Configuration file not found. Please create one at: {DEFAULT_CONFIG_PATHS[0]}"
        )

    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib

    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except Exception as exc:
        raise ConfigError(f"Failed to parse TOML configuration file: {exc}") from exc

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        # Extract the first error for a friendly message
        first_error = exc.errors()[0]
        loc = ".".join(str(x) for x in first_error["loc"])
        msg = first_error["msg"]
        raise ConfigError(f"Invalid configuration at '{loc}': {msg}") from exc
