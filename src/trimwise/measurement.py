"""Measure text consistently and derive source-preserving fitting prefixes."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import tiktoken

from trimwise.models import BudgetUnit

if TYPE_CHECKING:
    from tiktoken import Encoding

TokenCounter = Callable[[str], int]


class Measurer:
    """Measure text in one configured unit and expose lexical token IDs."""

    def __init__(
        self,
        unit: BudgetUnit,
        encoding_name: str,
        token_counter: TokenCounter | None,
    ) -> None:
        """Store measurement settings without eagerly loading unused encodings.

        Args:
            unit: Unit used for public budget counts.
            encoding_name: Tiktoken encoding used for tokens and ranking.
            token_counter: Optional caller-supplied token counter.
        """
        self.unit = unit
        self._encoding_name = encoding_name
        self._token_counter = token_counter
        self._encoding: Encoding | None = None

    def count(self, text: str) -> int:
        """Measure text and validate custom counter output.

        Args:
            text: Text to measure.

        Returns:
            Nonnegative size in the configured unit.

        Raises:
            ValueError: If a custom counter returns an invalid value.
        """
        if self.unit is BudgetUnit.CHARACTERS:
            return len(text)
        if self.unit is BudgetUnit.WORDS:
            return len(text.split())
        if self._token_counter is None:
            return len(self._get_encoding().encode(text, disallowed_special=()))

        count = self._token_counter(text)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("token_counter must return a nonnegative integer")
        return count

    def token_ids(self, text: str) -> list[int]:
        """Tokenize normalized ranking text with the configured encoding.

        Args:
            text: Ranking-only text.

        Returns:
            Tiktoken subword identifiers.
        """
        return self._get_encoding().encode(text, disallowed_special=())

    def fitting_prefix(self, text: str, limit: int) -> str:
        """Return the longest source prefix found within a measurement limit.

        Args:
            text: Source text whose prefix may be retained.
            limit: Maximum measured size.

        Returns:
            A source-preserving prefix, possibly empty.
        """
        if limit < 0:
            return ""
        if self.count(text) <= limit:
            return text
        if self.unit is BudgetUnit.CHARACTERS:
            return text[:limit]
        if self.unit is BudgetUnit.WORDS:
            matches = list(re.finditer(r"\S+(?:\s+|$)", text))
            return text[: matches[limit - 1].end()] if limit else ""
        if self._token_counter is None:
            return self._fitting_encoded_prefix(text, limit)
        return self._fitting_scanned_prefix(text, limit)

    def _fitting_encoded_prefix(self, text: str, limit: int) -> str:
        """Fit a source prefix around the configured encoding's token boundary.

        Args:
            text: Oversized source text.
            limit: Maximum encoded token count.

        Returns:
            Longest fitting source prefix around the first excluded token.
        """
        encoding = self._get_encoding()
        tokens = encoding.encode(text, disallowed_special=())
        decoded, offsets = encoding.decode_with_offsets(tokens)
        if decoded != text:
            return self._fitting_scanned_prefix(text, limit)

        boundary = offsets[limit]
        upper = next(
            (offset for offset in offsets[limit + 1 :] if offset > boundary),
            len(text),
        )
        fitting_end = 0
        for end in range(max(0, boundary - 1), upper + 1):
            if self.count(text[:end]) <= limit:
                fitting_end = end
        return text[:fitting_end]

    def _fitting_scanned_prefix(self, text: str, limit: int) -> str:
        """Find an exact prefix for a counter with no token-offset API.

        Args:
            text: Oversized source text.
            limit: Maximum custom token count.

        Returns:
            Longest fitting source prefix, possibly empty.
        """
        # ponytail: custom counters may be non-monotonic; keep their rare fallback exact until
        # callers can supply a source-offset API.
        for end in range(len(text) - 1, -1, -1):
            if self.count(text[:end]) <= limit:
                return text[:end]
        return ""

    def _get_encoding(self) -> Encoding:
        """Load and cache the tiktoken encoding on first lexical use.

        Returns:
            Configured tiktoken encoding.
        """
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding
