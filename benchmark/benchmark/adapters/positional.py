"""Simple positional controls for the compression benchmark."""

from __future__ import annotations

from benchmark.utils.tokens import count_tokens, token_prefix, token_suffix

from .base import CompressionResult, TimedAdapter


class PrefixAdapter(TimedAdapter):
    """Keep the first N benchmark tokens."""

    method_id = "naive_first_n"

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        return CompressionResult(self.method_id, token_prefix(context, budget))


class HeadTailAdapter(TimedAdapter):
    """Keep fitting head and tail token slices around an omission marker."""

    method_id = "head_tail"

    def __init__(self, omission_marker: str = "\n[…omitted…]\n") -> None:
        """Configure the marker placed between retained head and tail text.

        Args:
            omission_marker: Text inserted when both source regions fit.
        """
        self.omission_marker = omission_marker

    def _compress(self, context: str, query: str, budget: int, seed: int) -> CompressionResult:
        marker_cost = count_tokens(self.omission_marker)
        if budget <= marker_cost:
            return CompressionResult(self.method_id, token_prefix(context, budget))
        usable = budget - marker_cost
        head = token_prefix(context, (usable + 1) // 2)
        tail = token_suffix(context[len(head) :], usable // 2)
        output = head + (self.omission_marker if head and tail else "") + tail
        while output and count_tokens(output) > budget:
            tail = tail[1:]
            output = head + (self.omission_marker if head and tail else "") + tail
        return CompressionResult(self.method_id, output)
