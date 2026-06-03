# dtrans

A modern command-line translator powered by LLMs.

> **Note:** This is a ground-up rewrite of [translate-shell](https://github.com/soimort/translate-shell) with a focus on LLM-based translation and a significantly improved CLI experience.

## Features

- **LLM-powered translation** via DeepSeek, Gemini, ChatGPT, or any OpenAI-compatible API
- **Verify your translations** with alternatives, comparisons, and contextual examples
- **Rich terminal UI** with beautiful panels, tables, and spinners
- **Modern CLI** with explicit long-form flags and intuitive language syntax
- **Configurable providers** via a simple TOML config file

## Installation

```bash
pipx install dtrans
```

Or with `pip`:

```bash
pip install dtrans
```

## Quick Start

1. Create your configuration file at `~/.config/dtrans/config.toml`:

```toml
default_provider = "deepseek"

[providers.deepseek]
api_key = "sk-..."
model = "deepseek-chat"

[providers.openai]
api_key = "sk-..."
model = "gpt-4o"

[providers.gemini]
api_key = "..."
model = "gemini-1.5-flash"
```

2. Translate:

```bash
# Auto-detect source, translate to French
dtrans --to fr "Hello, world!"

# Specify source and target
dtrans en:fr "Hello, world!"

# Brief mode (translation text only)
dtrans --brief "Hello, world!"

# Identify language
dtrans --identify "手紙"
```

## Usage

```
Usage: dtrans [OPTIONS] TEXT

Options:
  --from TEXT       Source language code (e.g. 'en', 'ja'). Omit to auto-detect.
  --to TEXT         Target language code (e.g. 'fr', 'zh'). Defaults to system locale.
  --provider TEXT   Override the default provider from the config file.
  --brief / --verbose
                    Brief mode prints only the translation. Verbose is the default in a TTY.
  --identify        Identify the source language and exit.
  --version         Show version and exit.
  --help            Show this message and exit.
```

## Language Syntax

| Pattern | Meaning |
|---------|---------|
| `:fr` | Auto-detect source → French |
| `ja:` | Japanese → default target |
| `en:fr` | English → French |

## Custom Providers

Any OpenAI-compatible API can be configured:

```toml
[providers.local]
base_url = "http://localhost:11434/v1"
api_key = "ollama"
model = "llama3"
```

## Development

```bash
# Clone
git clone https://github.com/kriss-spy/deep-translate-shell.git
cd deep-translate-shell

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src tests
mypy src
```

## License

See [LICENSE](LICENSE).
