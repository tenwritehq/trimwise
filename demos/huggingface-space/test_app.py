"""Check demo budgets, provenance, measurement, baselines, and CPU boundaries."""

from __future__ import annotations

import asyncio
import inspect
import math
import unittest
from collections.abc import Sequence
from html import escape
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tiktoken
from app import (
    CONCURRENCY,
    CONFIG,
    MAX_BUDGET,
    MAX_INPUT_LINES,
    MAX_INPUT_TOKENS,
    MAX_QUERY_CHARS,
    MAX_SOURCE_CHARS,
    MAX_TOTAL_CHARS,
    MEASURER,
    MULTIPLE_PRESET,
    QUEUE_SIZE,
    SINGLE_PRESET,
    assemble_sources,
    build_app,
    display_result,
    method_notice,
    render_sources,
    render_statistics,
    run_demo,
    validate_inputs,
)

from trimwise import ContextSource, Strategy, Trimmer


class DemoTests(unittest.IsolatedAsyncioTestCase):
    """Exercise both modes through their shared async implementation."""

    async def test_single_preset_retains_late_evidence(self) -> None:
        """Keep the synthetic late decision that the short prefix does not reach."""
        result = await run_demo(*SINGLE_PRESET)
        self.assertIn("expired signing certificate", result.context)
        self.assertIn("Mira", result.context)
        self.assertNotIn("expired signing certificate", result.baseline)
        self.assertLessEqual(result.retained_tokens, SINGLE_PRESET[2])

    async def test_multiple_preset_retains_evidence_across_sources(self) -> None:
        """Keep the synthetic cause and recovery in separate input-aligned rows."""
        result = await run_demo(*MULTIPLE_PRESET)
        self.assertIn("ignoring backoff settings", result.sources[0].text)
        self.assertIn("14:32 UTC", result.sources[2].text)
        self.assertNotIn("14:32 UTC", result.baseline)
        self.assertEqual([row.source_index for row in result.sources], [0, 1, 2])

    async def test_zero_tiny_and_large_shared_budgets(self) -> None:
        """Enforce the complete rendered ceiling, including wrappers, in both modes."""
        for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
            for budget in (0, 1, 2, 7, 32, 96, MAX_BUDGET):
                with self.subTest(sources=len(preset[0]), budget=budget):
                    result = await run_demo(preset[0], preset[1], budget)
                    self.assertEqual(result.retained_tokens, MEASURER.count(result.context))
                    self.assertLessEqual(result.retained_tokens, budget)
                    self.assertLessEqual(result.baseline_tokens, budget)
                    if budget == 0:
                        self.assertEqual(result.context, "")
                        self.assertEqual(result.baseline, "")
                    if len(preset[0]) == 3 and budget < MEASURER.count("[Source 0]\nx"):
                        self.assertEqual(result.context, "")

    async def test_spans_refer_to_the_correct_original(self) -> None:
        """Map each retained range to its exact source with ordered character bounds."""
        for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
            result = await run_demo(*preset)
            for row in result.sources:
                original = preset[0][row.source_index]
                previous_end = 0
                for span in row.spans:
                    self.assertGreaterEqual(span.start, previous_end)
                    self.assertLess(span.start, span.end)
                    self.assertLessEqual(span.end, len(original))
                    self.assertIn(original[span.start : span.end], row.text)
                    previous_end = span.end
                self.assertEqual(row.input_count, MEASURER.count(original))
                self.assertEqual(row.output_count, MEASURER.count(row.text))
                self.assertEqual(bool(row.spans), bool(row.text))

    async def test_public_single_api_matches_demo(self) -> None:
        """Keep single-source text, counts, and spans identical to public atrim."""
        texts, query, budget = SINGLE_PRESET
        expected = await Trimmer(CONFIG).atrim(texts[0], budget, strategy="lexical", query=query)
        actual = await run_demo(*SINGLE_PRESET)
        self.assertEqual(actual.context, expected.text)
        self.assertEqual(actual.retained_tokens, expected.output_count)
        self.assertEqual(actual.sources[0].spans, expected.spans)

    async def test_public_shared_api_matches_demo(self) -> None:
        """Use the library's global competition and complete rendering unchanged."""
        texts, query, budget = MULTIPLE_PRESET
        expected = await Trimmer(CONFIG).atrim_context(
            [ContextSource(text, f"[Source {index}]\n") for index, text in enumerate(texts)],
            budget,
            strategy="lexical",
            query=query,
            separator="\n\n",
        )
        actual = await run_demo(*MULTIPLE_PRESET)
        self.assertEqual(actual.context, expected.text)
        self.assertEqual(actual.sources, expected.sources)
        self.assertEqual(actual.retained_tokens, expected.output_count)

    async def test_statistics_use_complete_assemblies(self) -> None:
        """Count wrappers independently and define reduction against the same input."""
        encoding = tiktoken.get_encoding(CONFIG.token_encoding)
        for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
            result = await run_demo(*preset)
            original_tokens = len(encoding.encode(result.original_context, disallowed_special=()))
            retained_tokens = len(encoding.encode(result.context, disallowed_special=()))
            self.assertEqual(result.original_tokens, original_tokens)
            self.assertEqual(result.retained_tokens, retained_tokens)
            self.assertEqual(result.evidence_tokens, sum(row.input_count for row in result.sources))
            self.assertAlmostEqual(
                result.reduction_percent, 100 * (1 - retained_tokens / original_tokens)
            )
            self.assertTrue(math.isfinite(result.trim_ms))
            self.assertGreaterEqual(result.trim_ms, 0)
            self.assertIn(f"{result.retained_tokens:,}", display_result(result)[0])

    async def test_baseline_uses_the_same_assembly_counter_and_budget(self) -> None:
        """Fit a real prefix with the library helper, preserving Unicode boundaries."""
        sources = ("hello\n\nworld", "", "late evidence: \U0001f680 \u4f60\u597d e\u0301")
        expected_input = (
            "[Source 0]\nhello\n\nworld\n\n[Source 2]\n"
            "late evidence: \U0001f680 \u4f60\u597d e\u0301"
        )
        for texts in ((sources[2],), sources):
            for budget in (0, 1, 8, 14, 20):
                result = await run_demo(texts, "evidence", budget)
                self.assertEqual(
                    result.original_context, texts[0] if len(texts) == 1 else expected_input
                )
                self.assertTrue(result.original_context.startswith(result.baseline))
                self.assertEqual(
                    result.baseline, MEASURER.fitting_prefix(result.original_context, budget)
                )
                self.assertEqual(result.baseline_tokens, MEASURER.count(result.baseline))
                self.assertLessEqual(result.baseline_tokens, budget)
                self.assertNotIn("\ufffd", result.baseline)

    async def test_empty_sources_keep_indices_and_finite_statistics(self) -> None:
        """Keep empty rows without labels and report zero reduction for empty input."""
        for texts in (("",), ("", "", "")):
            result = await run_demo(texts, "", 64)
            self.assertEqual(result.original_tokens, 0)
            self.assertEqual(result.retained_tokens, 0)
            self.assertEqual(result.reduction_percent, 0)
            self.assertEqual(result.context, "")
            self.assertEqual(len(result.sources), len(texts))
        result = await run_demo(("", "", "late evidence"), "evidence", 64)
        self.assertEqual(result.context, "[Source 2]\nlate evidence")
        self.assertEqual(result.sources[2].source_index, 2)
        self.assertIn("Source 0", render_sources(result))
        self.assertIn("No retained evidence", render_sources(result))

    async def test_blank_query_uses_model_free_structural_selection(self) -> None:
        """Accept whitespace-only queries and expose the strategy actually used."""
        for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
            result = await run_demo(preset[0], " \n\t", 64)
            self.assertEqual(result.strategy, Strategy.STRUCTURAL)
            self.assertLessEqual(result.retained_tokens, 64)

    async def test_unicode_and_repeated_sources_keep_identity(self) -> None:
        """Preserve duplicate rows and source-local code-point offsets without rewriting."""
        text = "\u4f60\u597d \U0001f680 e\u0301\n\nlate evidence \u6771\u4eac"
        result = await run_demo((text, "", text), "late evidence", 128)
        self.assertEqual([row.text for row in result.sources], [text, "", text])
        self.assertLess(result.context.index("[Source 0]"), result.context.index("[Source 2]"))
        for row in (result.sources[0], result.sources[2]):
            self.assertEqual(row.spans[0].end, len(text))
            self.assertEqual(text[row.spans[0].start : row.spans[0].end], row.text)

    async def test_user_html_is_escaped_in_excerpt_and_span_views(self) -> None:
        """Render user markup as text rather than executable tags or attributes."""
        text = '<img src=x onerror="alert(1)"><script>alert(2)</script> & <b>evidence</b>'
        result = await run_demo((text,), "evidence", 128)
        html = render_sources(result)
        self.assertNotIn("<img", html)
        self.assertNotIn("<script", html)
        self.assertIn(escape(text), html)
        self.assertEqual(result.context, text)

    async def test_concurrent_requests_are_isolated(self) -> None:
        """Avoid sharing user strings or results across simultaneous handlers."""
        first, second = await asyncio.gather(
            run_demo(("private alpha evidence",), "alpha", 64),
            run_demo(("", "private beta evidence", ""), "beta", 64),
        )
        self.assertNotIn("beta", first.context)
        self.assertNotIn("alpha", second.context)
        self.assertEqual(second.sources[1].text, "private beta evidence")

    async def test_no_semantic_backend_is_invoked(self) -> None:
        """Exercise queried and queryless requests with embedding calls forbidden."""
        with patch("trimwise.semantic.SemanticEmbedder.embed", side_effect=AssertionError):
            for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
                await run_demo(*preset)
                await run_demo(preset[0], "", preset[2])

    async def test_selected_methods_use_public_strategy_resolution(self) -> None:
        """Expose the actual strategy for explicit structural and automatic choices."""
        for texts in (SINGLE_PRESET[0], MULTIPLE_PRESET[0]):
            for method, query, expected in (
                ("structural", "evidence", Strategy.STRUCTURAL),
                ("auto", "evidence", Strategy.LEXICAL),
                ("auto", "", Strategy.STRUCTURAL),
            ):
                result = await run_demo(texts, query, 64, method)
                self.assertEqual(result.strategy, expected)
                self.assertLessEqual(result.retained_tokens, 64)

    async def test_semantic_and_hybrid_dispatch_preserve_shared_budgets(self) -> None:
        """Exercise real ranking with an offline callback, avoiding unit-test downloads."""

        def embed(query: str, passages: Sequence[str]) -> tuple[object, list[object]]:
            """Return deterministic vectors for the two synthetic evidence signals.

            Args:
                query: Query required by the selected strategy.
                passages: Library-prepared scoring text.

            Returns:
                A query vector and one same-dimensional finite vector per passage.
            """
            return [1.0, 0.0], [
                [1.0, 0.0] if "certificate" in text or "retry" in text else [0.0, 1.0]
                for text in passages
            ]

        trimmer = Trimmer(CONFIG, embedding_callback=embed)
        with patch("app.TRIMMER", trimmer):
            for preset in (SINGLE_PRESET, MULTIPLE_PRESET):
                for method in ("semantic", "hybrid"):
                    result = await run_demo(*preset, method)
                    self.assertEqual(result.strategy.value, method)
                    self.assertEqual(result.retained_tokens, MEASURER.count(result.context))
                    self.assertLessEqual(result.retained_tokens, preset[2])
                    self.assertEqual(len(result.sources), len(preset[0]))

    async def test_invalid_methods_and_blank_model_queries_are_rejected(self) -> None:
        """Reject unsupported methods and missing semantic intent before model loading."""
        with patch("trimwise.semantic.SemanticEmbedder.embed", side_effect=AssertionError):
            for method in ("unknown", None, 3):
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Choose"):
                    await run_demo(("text",), "query", 64, method)
            for method in ("semantic", "hybrid"):
                with self.assertRaisesRegex(ValueError, "nonblank query"):
                    await run_demo(("text",), " \n", 0, method)

    def test_model_disclosure_and_initial_statistics(self) -> None:
        """Name the configured CPU model and avoid fabricating counts before trimming."""
        for method in ("semantic", "hybrid"):
            notice = method_notice(method)
            self.assertIn(CONFIG.embedding_model, notice)
            self.assertIn("ONNX CPU", notice)
            self.assertIn("Downloads", notice)
        self.assertNotIn(CONFIG.embedding_model, method_notice("lexical"))
        self.assertIn("Awaiting trim", render_statistics())
        self.assertEqual(render_statistics().count("&ndash;"), 4)


class InputLimitTests(unittest.IsolatedAsyncioTestCase):
    """Reject excessive or invalid requests before expensive compression work."""

    def test_invalid_budget_types_and_values(self) -> None:
        """Reject fractional, nonfinite, missing, negative, and over-limit budgets."""
        for budget in (
            None,
            "64",
            True,
            -1,
            1.5,
            float("nan"),
            float("inf"),
            MAX_BUDGET + 1,
            10**1000,
        ):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "whole number"):
                validate_inputs(("text",), "", budget)

    def test_integer_float_and_boundary_budgets(self) -> None:
        """Accept Gradio's integer-valued numeric inputs at both allowed endpoints."""
        for budget in (0, 64.0, MAX_BUDGET):
            self.assertEqual(validate_inputs(("text",), "", budget)[2], int(budget))

    def test_source_count_and_types(self) -> None:
        """Reject inputs that cannot map to the one- or three-source interfaces."""
        for sources in ((), ("a", "b"), ("a", 3, "b")):
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, "text sources"):
                validate_inputs(sources, "", 64)

    def test_per_source_character_limit(self) -> None:
        """Reject oversized strings before tokenization, preserving the input unchanged."""
        with self.assertRaisesRegex(ValueError, "Each source"):
            validate_inputs(("a" * (MAX_SOURCE_CHARS + 1),), "", 64)

    def test_total_character_limit(self) -> None:
        """Reject individually fitting sources whose combined characters exceed the cap."""
        texts = ("a" * (MAX_TOTAL_CHARS // 3 + 1),) * 3
        with self.assertRaisesRegex(ValueError, "characters"):
            validate_inputs(texts, "", 64)

    def test_total_token_limit(self) -> None:
        """Cap Unicode-heavy source evidence even when the character limits allow it."""
        text = "\U0001f600" * (MAX_INPUT_TOKENS + 1)
        self.assertLess(len(text), MAX_SOURCE_CHARS)
        self.assertGreater(MEASURER.count(text), MAX_INPUT_TOKENS)
        with self.assertRaisesRegex(ValueError, "tokens in total"):
            validate_inputs((text,), "", 64)

    def test_total_line_limit(self) -> None:
        """Bound high-candidate Markdown inputs across the full request."""
        with self.assertRaisesRegex(ValueError, "lines"):
            validate_inputs(("x\n" * (MAX_INPUT_LINES + 1),), "", 64)

    def test_query_limit_and_types(self) -> None:
        """Reject nontext or oversized queries without echoing their contents."""
        for query in (None, 1, "x" * (MAX_QUERY_CHARS + 1)):
            with self.subTest(query_type=type(query)), self.assertRaisesRegex(ValueError, "query"):
                validate_inputs(("evidence",), query, 64)

    def test_character_and_line_boundaries_are_inclusive(self) -> None:
        """Accept exact hosting boundaries when the token cap also fits."""
        text = "a" * MAX_SOURCE_CHARS
        self.assertEqual(validate_inputs((text,), "x" * MAX_QUERY_CHARS, 0)[0], (text,))
        self.assertEqual(
            validate_inputs(("x\n" * MAX_INPUT_LINES,), "", 0)[0], ("x\n" * MAX_INPUT_LINES,)
        )

    def test_full_assembly_preserves_blank_positions_and_text(self) -> None:
        """Emit only nonempty sources without trimming whitespace or renumbering them."""
        self.assertEqual(assemble_sources(("", "  ", "tail")), "[Source 1]\n  \n\n[Source 2]\ntail")

    async def test_queue_and_async_event_configuration(self) -> None:
        """Share CPU slots across both tabs and disable app analytics."""
        demo = build_app()
        self.assertFalse(demo.analytics_enabled)
        self.assertEqual(demo._queue.max_size, QUEUE_SIZE)
        functions = [
            fn for fn in demo.fns.values() if fn.api_name in ("trim_single", "trim_multiple")
        ]
        self.assertEqual(len(functions), 2)
        for fn in functions:
            self.assertTrue(inspect.iscoroutinefunction(fn.fn))
            self.assertEqual(fn.concurrency_id, "trimwise_cpu")
            self.assertEqual(fn.concurrency_limit, CONCURRENCY)
        demo.close()

    async def test_zero_gpu_registers_only_a_private_unused_handler(self) -> None:
        """Satisfy ZeroGPU startup without decorating either CPU trim handler."""

        def decorate(function: object) -> object:
            """Preserve the handler while recording platform decorator registration."""
            return function

        gpu = Mock(return_value=decorate)
        with (
            patch.dict("os.environ", {"SPACES_ZERO_GPU": "1"}),
            patch.dict("sys.modules", {"spaces": SimpleNamespace(GPU=gpu)}),
        ):
            demo = build_app()
        gpu.assert_called_once_with(duration=1)
        registration = [fn for fn in demo.fns.values() if fn.api_name == "zero_gpu_registration"]
        self.assertEqual(len(registration), 1)
        self.assertEqual(registration[0].api_visibility, "private")
        trimming = [
            fn for fn in demo.fns.values() if fn.api_name in ("trim_single", "trim_multiple")
        ]
        self.assertEqual(len(trimming), 2)
        self.assertTrue(all(inspect.iscoroutinefunction(fn.fn) for fn in trimming))
        demo.close()


if __name__ == "__main__":
    unittest.main()
