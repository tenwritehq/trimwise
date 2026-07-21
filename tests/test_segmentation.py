"""Verify Markdown segmentation and source-preserving fallback behavior."""

from __future__ import annotations

import pytest

from trimwise import SourceSpan, TrimConfig, Trimmer
from trimwise.measurement import Measurer
from trimwise.models import BudgetUnit
from trimwise.segmentation import (
    _deduplicate_spans,
    _line_offsets,
    _with_uncovered_source,
    segment_text,
)


def test_markdown_blocks_keep_exact_source_slices() -> None:
    """Represent headings, lists, quotes, tables, HTML, and code without rendering."""
    source = (
        "# Heading\n\nParagraph.\n\n- first\n- second\n\n> quote\n\n"
        "| a | b |\n| - | - |\n| 1 | 2 |\n\n<div>raw</div>\n\n```py\nx = 1\n```\n"
    )
    segments = segment_text(source)
    kinds = {segment.kind for segment in segments}
    assert {"heading", "paragraph", "list", "blockquote", "table", "html", "fence"} <= kinds
    assert all(segment.text == source[segment.start : segment.end] for segment in segments)


def test_setext_heading_starts_a_section() -> None:
    """Recognize setext headings through Markdown token maps."""
    segments = segment_text("Title\n=====\n\nBody\n")
    assert segments[0].kind == "heading"
    assert segments[1].heading_index == 0


def test_nested_lists_keep_list_kind_and_exact_source() -> None:
    """Keep nested list items distinguishable without re-rendering their source."""
    source = "- parent\n  - child\n    continuation\n"
    segments = segment_text(source)
    assert [(segment.kind, segment.text) for segment in segments] == [
        ("list", "- parent\n"),
        ("list", "  - child\n    continuation\n"),
    ]


def test_reference_definition_is_not_silently_lost() -> None:
    """Cover source ranges consumed without a normal visible block."""
    source = "Paragraph with [link][id].\n\n[id]: https://example.com\n"
    segments = segment_text(source)
    covered = "".join(segment.text for segment in segments)
    assert "[id]: https://example.com" in covered


def test_front_matter_is_one_raw_source_unit() -> None:
    """Prevent CommonMark rules from reinterpreting leading front matter."""
    source = "---\ntitle: Demo\nauthor: Aakash\n---\n\nBody.\n"
    segments = segment_text(source)
    assert (segments[0].kind, segments[0].text) == (
        "raw",
        "---\ntitle: Demo\nauthor: Aakash\n---\n",
    )


def test_whitespace_only_source_is_one_raw_unit() -> None:
    """Preserve a blank source for character-budget handling."""
    segments = segment_text(" \n\t")
    assert len(segments) == 1
    assert segments[0].kind == "raw"


def test_empty_source_has_no_segments() -> None:
    """Avoid manufacturing candidates for an empty string."""
    assert segment_text("") == []


def test_line_offsets_support_missing_final_newline() -> None:
    """Map every source line and the exact terminal offset."""
    assert _line_offsets("a\nb") == [0, 2, 3]


def test_nested_and_duplicate_spans_are_removed() -> None:
    """Prefer the richer exact span and remove contained parser ranges."""
    spans = _deduplicate_spans([(0, 5, "paragraph"), (0, 5, "heading"), (1, 4, "raw")])
    assert spans == [(0, 5, "heading")]


def test_partially_overlapping_spans_are_clipped() -> None:
    """Keep nonoverlapping tails from unusual parser token maps."""
    spans = _deduplicate_spans([(0, 5, "paragraph"), (3, 8, "raw")])
    assert spans == [(0, 5, "paragraph"), (5, 8, "raw")]


def test_uncovered_nonblank_ranges_become_raw() -> None:
    """Add source that falls between parser-backed spans."""
    spans = _with_uncovered_source("raw\nbody\ntail", [(4, 9, "paragraph")])
    assert spans == [(0, 4, "raw"), (4, 9, "paragraph"), (9, 13, "raw")]


def test_closed_fence_fallback_remains_balanced() -> None:
    """Retain original opening and closing fence lines when the body is oversized."""
    source = "```python\n" + "print('large')\n" * 20 + "```\n"
    result = Trimmer().trim(source, 35, unit="characters")
    assert result.text.startswith("```python\n")
    assert "\n```" in result.text[len("```python\n") :]
    assert result.output_count <= 35
    closing = "```\n"
    retained_prefix = result.text[: -len(closing)]
    assert result.spans == (
        SourceSpan(0, len(retained_prefix)),
        SourceSpan(len(source) - len(closing), len(source)),
    )


def test_closed_fence_shell_has_two_source_spans() -> None:
    """Map a retained opening and closing fence around an omitted body."""
    opening = "```python\n"
    closing = "```\n"
    source = opening + "one oversized body line" * 10 + "\n" + closing
    config = TrimConfig(omission_marker="an omission marker that cannot fit")
    result = Trimmer(config).trim(source, len(opening + closing), unit="characters")
    assert result.text == opening + closing
    assert result.spans == (
        SourceSpan(0, len(opening)),
        SourceSpan(len(source) - len(closing), len(source)),
    )


def test_unclosed_fence_is_not_artificially_closed() -> None:
    """Preserve an unclosed source fence rather than inventing source syntax."""
    source = "```python\n" + "x = 1\n" * 10
    result = Trimmer().trim(source, 20, unit="characters")
    assert result.text.startswith("```python\n")
    assert result.text.count("```") == 1


def test_fence_like_body_line_is_not_treated_as_a_closer() -> None:
    """Require a closing fence line to contain only its marker and whitespace."""
    source = "```python\nvalue = 1\n```not-a-close"
    result = Trimmer().trim(source, 24, unit="characters")
    assert all(line.strip() != "```" for line in result.text.splitlines()[1:])


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5])
def test_progressive_fallback_always_respects_tiny_limits(limit: int) -> None:
    """Return a measurable source prefix at every positive character boundary.

    Args:
        limit: Tiny character limit under test.
    """
    result = Trimmer().trim("A sentence that cannot fit whole.", limit, unit="characters")
    assert 0 < result.output_count <= limit


def test_measurer_prefix_handles_negative_and_fitting_limits() -> None:
    """Cover direct prefix boundaries used by progressive fallback."""
    measurer = Measurer(BudgetUnit.CHARACTERS, "o200k_base", None)
    assert measurer.fitting_prefix("abc", -1) == ""
    assert measurer.fitting_prefix("abc", 3) == "abc"


def test_word_prefix_preserves_whitespace_after_the_last_fitting_word() -> None:
    """Stop at the requested word while retaining its exact trailing whitespace."""
    measurer = Measurer(BudgetUnit.WORDS, "o200k_base", None)
    assert measurer.fitting_prefix("one  two\nthree four", 2) == "one  two\n"


def test_token_prefix_handles_nonmonotonic_token_counts() -> None:
    """Find the longest prefix even when appending text reduces its token count."""
    measurer = Measurer(BudgetUnit.TOKENS, "o200k_base", None)
    assert measurer.fitting_prefix("Aqux", 1) == "Aqu"


def test_custom_counter_prefix_handles_nonmonotonic_counts() -> None:
    """Find the longest prefix for a caller-defined non-monotonic counter."""

    def counter(text: str) -> int:
        """Make the three-character prefix cheaper than its predecessor."""
        return 1 if len(text) == 3 else len(text)

    measurer = Measurer(BudgetUnit.TOKENS, "o200k_base", counter)
    assert measurer.fitting_prefix("abcd", 1) == "abc"


def test_token_prefix_preserves_unpaired_surrogate_source() -> None:
    """Fall back safely when tokenizer normalization cannot round-trip the source."""
    text = "\ud800Aqux"
    measurer = Measurer(BudgetUnit.TOKENS, "o200k_base", None)
    expected = max(
        (text[:end] for end in range(len(text) + 1) if measurer.count(text[:end]) <= 1),
        key=len,
    )
    assert measurer.fitting_prefix(text, 1) == expected


def test_fallback_prefers_a_complete_sentence() -> None:
    """Keep a complete sentence instead of filling spare budget with a partial one."""
    source = "Keep this sentence. " + "unfinished" * 20
    config = TrimConfig(omission_marker="an omission marker that cannot fit")
    result = Trimmer(config).trim(source, 30, unit="characters")
    assert result.text == "Keep this sentence. "
    assert result.spans == (SourceSpan(0, len(result.text)),)


def test_fallback_prefers_a_complete_source_line() -> None:
    """Keep a complete source line when an oversized block has no sentence boundary."""
    source = "complete line\n" + "unfinished" * 20
    config = TrimConfig(omission_marker="an omission marker that cannot fit")
    result = Trimmer(config).trim(source, 20, unit="characters")
    assert result.text == "complete line\n"


@pytest.mark.parametrize(
    ("source", "opening", "closing"),
    [
        (
            "Opening anchor. Common signal alpha beta. Common signal alpha gamma. "
            "Rare middle decision Zephyr. Common signal beta gamma. Closing anchor.",
            "Opening anchor.",
            "Closing anchor.",
        ),
        (
            "Opening anchor\nCommon signal alpha beta\nCommon signal alpha gamma\n"
            "Rare middle decision Zephyr\nCommon signal beta gamma\nClosing anchor",
            "Opening anchor\n",
            "Closing anchor",
        ),
        (
            "Opening anchor。Common signal alpha beta。Common signal alpha gamma。"
            "Rare middle decision Zephyr。Common signal beta gamma。Closing anchor。",
            "Opening anchor。",
            "Closing anchor。",
        ),
    ],
)
def test_structural_plaintext_covers_an_oversized_paragraph(
    source: str,
    opening: str,
    closing: str,
) -> None:
    """Rank complete units across one oversized plain-text paragraph.

    Args:
        source: Sentence- or line-delimited plain text.
        opening: Expected retained beginning.
        closing: Expected retained ending.
    """
    result = Trimmer(TrimConfig(omission_marker="<cut>")).trim(
        source,
        90,
        unit="characters",
    )
    assert result.text.startswith(opening)
    assert "Common signal" in result.text
    assert result.text.endswith(closing)
    assert result.output_count <= 90


def test_closed_fence_fallback_prefers_complete_body_lines() -> None:
    """Remove whole code lines before considering a partial source prefix."""
    source = "```\nshort\n" + "unfinished" * 20 + "\n```\n"
    config = TrimConfig(omission_marker="an omission marker that cannot fit")
    result = Trimmer(config).trim(source, 18, unit="characters")
    assert result.text == "```\nshort\n```\n"
    assert result.spans == (
        SourceSpan(0, len("```\nshort\n")),
        SourceSpan(len(source) - len("```\n"), len(source)),
    )
