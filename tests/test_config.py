"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from dtrans.config import ConfigError, find_config_file, load_config


class TestFindConfigFile:
    def test_returns_none_when_no_config_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "dtrans.config.DEFAULT_CONFIG_PATHS",
            [tmp_path / "nonexistent.toml"],
        )
        assert find_config_file() is None


class TestLoadConfig:
    def test_raises_on_missing_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "dtrans.config.DEFAULT_CONFIG_PATHS",
            [tmp_path / "nonexistent.toml"],
        )
        with pytest.raises(ConfigError):
            load_config()

    def test_raises_friendly_error_when_default_provider_missing(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "default_provider" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()

    def test_raises_friendly_error_when_provider_missing_api_key(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n[providers.deepseek]\nmodel = "deepseek-chat"\n'
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "api_key" in str(exc_info.value)

    def test_raises_friendly_error_when_provider_missing_model(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n[providers.deepseek]\napi_key = "sk-xxx"\n'
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "model" in str(exc_info.value)

    def test_raises_friendly_error_on_malformed_toml(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text('this is not valid toml [[[')

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "parse" in str(exc_info.value).lower() or "TOML" in str(exc_info.value)

    def test_loads_valid_config(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n'
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
            'base_url = "https://api.example.com/v1"\n'
        )

        cfg = load_config(config_path)

        assert cfg.default_provider == "deepseek"
        assert cfg.providers["deepseek"].api_key == "sk-xxx"
        assert cfg.providers["deepseek"].model == "deepseek-chat"
        assert cfg.providers["deepseek"].base_url == "https://api.example.com/v1"

    def test_base_url_is_optional(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n'
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
        )

        cfg = load_config(config_path)

        assert cfg.providers["deepseek"].base_url is None

    def test_provider_type_defaults_to_provider_name(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n'
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
        )

        cfg = load_config(config_path)
        assert cfg.providers["deepseek"].provider_type == "deepseek"

    def test_provider_type_can_be_gemini(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "gemini"\n\n'
            '[providers.gemini]\n'
            'provider_type = "gemini"\n'
            'api_key = "abc123"\n'
            'model = "gemini-1.5-flash"\n'
        )

        cfg = load_config(config_path)
        assert cfg.providers["gemini"].provider_type == "gemini"

    def test_system_prompt_is_optional(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n'
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
        )

        cfg = load_config(config_path)
        assert cfg.providers["deepseek"].system_prompt is None

    def test_system_prompt_can_be_set(self, tmp_path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'default_provider = "deepseek"\n\n'
            '[providers.deepseek]\n'
            'api_key = "sk-xxx"\n'
            'model = "deepseek-chat"\n'
            'system_prompt = "You are a pirate translator."\n'
        )

        cfg = load_config(config_path)
        assert cfg.providers["deepseek"].system_prompt == "You are a pirate translator."
