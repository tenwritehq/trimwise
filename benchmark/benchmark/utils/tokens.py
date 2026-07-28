"""Count and slice text with the benchmark's token encoding."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=4)
def get_encoding(name: str = "o200k_base") -> Any:
    """Return a cached tiktoken encoding, using a bundled cache when present.

    Args:
        name: Tiktoken encoding identifier.

    Returns:
        Encoding instance used for all benchmark budget accounting.
    """
    import tiktoken

    bundled_cache = Path(__file__).resolve().parents[2] / "vendor" / "tiktoken-cache"
    if bundled_cache.is_dir():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(bundled_cache))
    return tiktoken.get_encoding(name)


def token_ids(text: str, encoding_name: str = "o200k_base") -> list[int]:
    """Encode text without allowing special tokens.

    Args:
        text: Source text to encode.
        encoding_name: Tiktoken encoding identifier.

    Returns:
        Token IDs representing the input text.
    """
    return get_encoding(encoding_name).encode(text, disallowed_special=())


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count source text with the configured tiktoken encoding.

    Args:
        text: Source text to count.
        encoding_name: Tiktoken encoding identifier.

    Returns:
        Number of encoded tokens.
    """
    return len(token_ids(text, encoding_name))


def token_prefix(text: str, budget: int, encoding_name: str = "o200k_base") -> str:
    """Return the longest exact character prefix that fits the token budget.

    Args:
        text: Source text to truncate.
        budget: Maximum permitted token count.
        encoding_name: Tiktoken encoding identifier.

    Returns:
        Fitting exact character prefix.
    """
    if budget <= 0:
        return ""
    enc = get_encoding(encoding_name)
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= budget:
        return text
    # Decode can normalize malformed byte boundaries. Scan backwards until the
    # exact source prefix also satisfies the external benchmark tokenizer.
    candidate = enc.decode(ids[:budget])
    end = min(len(text), len(candidate) + 8)
    while end > 0 and count_tokens(text[:end], encoding_name) > budget:
        end -= 1
    return text[:end]


def token_suffix(text: str, budget: int, encoding_name: str = "o200k_base") -> str:
    """Return the longest exact character suffix that fits the token budget.

    Args:
        text: Source text to truncate.
        budget: Maximum permitted token count.
        encoding_name: Tiktoken encoding identifier.

    Returns:
        Fitting exact character suffix.
    """
    if budget <= 0:
        return ""
    enc = get_encoding(encoding_name)
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= budget:
        return text
    candidate = enc.decode(ids[-budget:])
    start = max(0, len(text) - len(candidate) - 8)
    while start < len(text) and count_tokens(text[start:], encoding_name) > budget:
        start += 1
    return text[start:]
