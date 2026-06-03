"""Language syntax parsing and locale utilities."""

from __future__ import annotations

import locale
import re

# Pattern: optional source, colon, optional target.
# Supports language codes with hyphens, e.g. zh-CN, zh-TW.
_LANG_SPEC_RE = re.compile(r"^([a-zA-Z-]+)?\:([a-zA-Z-]+)?$")


def parse_lang_spec(spec: str) -> tuple[str | None, str | None] | None:
    """Parse a translate-shell-style language specification.

    Returns (source_lang, target_lang) or None if *spec* is not a lang spec.

    Examples:
        ":fr"      -> (None, "fr")
        "ja:"      -> ("ja", None)
        "en:fr"    -> ("en", "fr")
        "hello"    -> None
        ":zh+ja"   -> raises ValueError (multiple targets)
    """
    if "+" in spec:
        raise ValueError("Multiple target languages are not supported.")

    match = _LANG_SPEC_RE.match(spec)
    if not match:
        return None

    source, target = match.group(1), match.group(2)
    return (source.lower() if source else None, target.lower() if target else None)


def get_default_target() -> str:
    """Return the default target language from the system locale.

    Falls back to 'en' if the locale cannot be determined.
    """
    try:
        loc, _ = locale.getlocale()
        if loc and "_" in loc:
            return loc.split("_")[0].lower()
    except Exception:
        pass
    return "en"
