"""Run resumable OpenAI QA evaluation without GPU inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from benchmark.datasets.loader import selected_cases
from benchmark.metrics.qa import score_answer

DEFAULT_PROMPT_TEMPLATE = "{protected_prefix}\n\nCONTEXT:\n{context}\n\n{protected_suffix}"
FULL_CONTEXT_METHOD = "full_context"


def _key(row: dict[str, Any]) -> tuple[str, str, int | None, bool]:
    """Return the stable identity for one saved answer.

    Args:
        row: Saved evaluation result row.

    Returns:
        Case, context method, optional compression budget, and compressor query access.
    """
    budget = row.get("budget")
    return (
        str(row["case_id"]),
        str(row["method_id"]),
        None if budget is None else int(budget),
        bool(row.get("query_aware", True)),
    )


def _completed_keys(path: Path) -> set[tuple[str, str, int | None, bool]]:
    """Return successful evaluation identities already persisted.

    Args:
        path: JSONL output file to inspect.

    Returns:
        Identities that do not need another billable API call.
    """
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as source:
        return {
            _key(row)
            for line in source
            if line.strip()
            for row in [json.loads(line)]
            if row.get("qa_status") == "success"
        }


@contextmanager
def _exclusive_run(path: Path) -> Iterator[None]:
    """Prevent concurrent billable runs from sharing one evaluation output.

    Args:
        path: JSONL result path whose companion lock file is held.

    Yields:
        Control while this process exclusively owns the output path.

    Raises:
        RuntimeError: If another evaluation process already holds the lock.
    """
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"OpenAI QA evaluation is already running for {path}") from error
        yield


def _successful_contexts(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index successful saved compression outputs for the requested evaluation.

    Args:
        path: Compression JSONL produced by the benchmark runner.

    Returns:
        Successful compression rows keyed by case, method, and budget.
    """
    with path.open("r", encoding="utf-8") as source:
        rows = (json.loads(line) for line in source if line.strip())
        contexts: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            if row.get("status") != "success":
                continue
            key = str(row["case_id"]), str(row["method_id"]), int(row["budget"])
            if key in contexts and bool(contexts[key].get("query_aware", True)) != bool(
                row.get("query_aware", True)
            ):
                raise ValueError(f"ambiguous query access for compression result: {key}")
            contexts[key] = row
        return contexts


def _prompt(case: dict[str, Any], context: str) -> str:
    """Build a source-grounded prompt that requests only the short answer.

    Args:
        case: Benchmark case containing source-grounding instructions.
        context: Full or compressed context supplied to the evaluator.

    Returns:
        Prompt sent to the Responses API.
    """
    prefix = case.get("protected_prefix") or "Answer only from the supplied context."
    suffix = case.get("protected_suffix") or f"Question: {case.get('query', '')}"
    return (
        DEFAULT_PROMPT_TEMPLATE.format(
            protected_prefix=prefix,
            context=context,
            protected_suffix=suffix,
        )
        + "\n\nReturn only the concise answer. Do not add a label, explanation, or citation."
    )


def _candidates(settings: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the full-context control and selected compressed QA candidates.

    Args:
        settings: Evaluation configuration loaded from YAML.

    Yields:
        Rows containing one benchmark case and the exact context to evaluate.

    Raises:
        ValueError: If a requested compression result is unavailable.
    """
    cases = list(selected_cases(settings["dataset"], settings.get("case_ids")))
    contexts = _successful_contexts(Path(settings["results"]))
    methods = tuple(str(method) for method in settings["methods"])
    budgets = tuple(int(budget) for budget in settings["budgets"])
    for case in cases:
        yield {
            "case": case,
            "method_id": FULL_CONTEXT_METHOD,
            "budget": None,
            "query_aware": True,
            "context": str(case["context"]),
        }
        for method_id in methods:
            for budget in budgets:
                row = contexts.get((str(case["case_id"]), method_id, budget))
                if row is None:
                    raise ValueError(
                        f"missing successful compression result for {case['case_id']}, "
                        f"{method_id}, {budget}"
                    )
                yield {
                    "case": case,
                    "method_id": method_id,
                    "budget": budget,
                    "query_aware": bool(row.get("query_aware", True)),
                    "context": str(row["output"]),
                }


def _usage(response: Any) -> dict[str, int | None]:
    """Extract API token usage without depending on SDK response internals.

    Args:
        response: Completed Responses API result.

    Returns:
        Input, output, and total token counts when the API supplies them.
    """
    usage = getattr(response, "usage", None)
    return {
        "qa_input_tokens": getattr(usage, "input_tokens", None),
        "qa_output_tokens": getattr(usage, "output_tokens", None),
        "qa_total_tokens": getattr(usage, "total_tokens", None),
    }


def _write_result(
    sink: Any, candidate: dict[str, Any], settings: dict[str, Any], result: dict[str, Any]
) -> None:
    """Persist one API result so interrupted evaluations resume safely.

    Args:
        sink: Open JSONL output handle.
        candidate: Case and context identity that was evaluated.
        settings: Model and generation configuration.
        result: Completion, failure, timing, and usage fields.
    """
    case = candidate["case"]
    sink.write(
        json.dumps(
            {
                "case_id": case["case_id"],
                "track": case["track"],
                "method_id": candidate["method_id"],
                "query_aware": candidate["query_aware"],
                "budget": candidate["budget"],
                "qa_context_kind": "full" if candidate["budget"] is None else "compressed",
                "qa_status": result["qa_status"],
                "qa_model_id": settings["model_id"],
                "qa_model": settings["model"],
                "qa_max_output_tokens": settings["max_output_tokens"],
                "qa_reasoning_effort": settings["reasoning_effort"],
                "qa_verbosity": settings["verbosity"],
                **result,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sink.flush()


def _append_result(
    destination: Path,
    candidate: dict[str, Any],
    settings: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Append one completed API result without sharing a file handle across tasks.

    Args:
        destination: JSONL path exclusively held by this evaluator.
        candidate: Case and context identity that was evaluated.
        settings: Model and generation configuration.
        result: Completion, failure, timing, and usage fields.
    """
    with destination.open("a", encoding="utf-8") as sink:
        _write_result(sink, candidate, settings, result)


def _model_settings(settings: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one complete configuration for each requested OpenAI evaluator.

    Args:
        settings: Shared evaluation settings, optionally with a ``models`` list.

    Yields:
        A single-model settings dictionary suitable for one resumable output.
    """
    models = settings.get("models")
    if not models:
        yield settings
        return
    shared = {key: value for key, value in settings.items() if key != "models"}
    for model in models:
        model_settings = shared | dict(model)
        model_settings.setdefault(
            "model_id", str(model_settings.get("id", model_settings["model"]))
        )
        yield model_settings


async def _request(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Request and score one answer while respecting evaluator concurrency.

    Args:
        candidate: Case and exact context sent to the evaluator.
        settings: Model and generation configuration.
        client: Async Responses API client.
        semaphore: Shared limit for in-flight API calls.

    Returns:
        The candidate and its completed result row fields.
    """
    from openai import APIError

    async with semaphore:
        call_started = time.perf_counter_ns()
        try:
            response = await client.responses.create(
                model=settings["model"],
                input=_prompt(candidate["case"], candidate["context"]),
                max_output_tokens=int(settings["max_output_tokens"]),
                reasoning={"effort": settings["reasoning_effort"]},
                text={"verbosity": settings["verbosity"]},
                store=False,
            )
            answer = response.output_text.strip()
            result = {
                "qa_status": "success",
                "qa_output": answer,
                "qa_latency_ms": (time.perf_counter_ns() - call_started) / 1_000_000,
                **_usage(response),
                **score_answer(candidate["case"], answer),
            }
        except APIError as error:
            result = {
                "qa_status": "failed",
                "qa_error": f"{type(error).__name__}: {error}"[:500],
                "qa_latency_ms": (time.perf_counter_ns() - call_started) / 1_000_000,
            }
    return candidate, result


async def _run_model(
    settings: dict[str, Any],
    dry_run: bool,
    semaphore: asyncio.Semaphore,
    max_concurrency: int,
) -> None:
    """Execute a resumable Responses-API QA evaluation.

    Args:
        settings: Validated single-model evaluation configuration.
        dry_run: Print the intended call count without calling the API.
        semaphore: Shared limit for in-flight API calls.
        max_concurrency: Maximum scheduled requests per evaluator.

    Raises:
        RuntimeError: If the API key is absent when an API call is required.
    """
    all_candidates = list(_candidates(settings))
    destination = Path(settings["output"])
    if dry_run:
        completed = _completed_keys(destination)
        print(
            f"[openai-qa] total={len(all_candidates)} "
            f"pending={len(all_candidates) - len(completed)} "
            f"resumed={len(completed)}",
            flush=True,
        )
        return
    api_key_env = str(settings.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for the OpenAI QA evaluation")
    from openai import AsyncOpenAI

    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_run(destination):
        completed = _completed_keys(destination)
        pending = [
            candidate
            for candidate in all_candidates
            if _key(candidate | {"case_id": candidate["case"]["case_id"]}) not in completed
        ]
        print(
            f"[openai-qa] total={len(all_candidates)} pending={len(pending)} "
            f"resumed={len(completed)} concurrency={settings['max_concurrency']}",
            flush=True,
        )
        if not pending:
            return
        started = time.perf_counter()
        async with AsyncOpenAI(
            api_key=api_key, timeout=float(settings["timeout_seconds"])
        ) as client:
            candidate_stream = iter(pending)
            tasks = {
                asyncio.create_task(_request(next(candidate_stream), settings, client, semaphore))
                for _ in range(min(len(pending), max_concurrency))
            }
            index = 0
            try:
                while tasks:
                    completed, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in completed:
                        candidate, result = task.result()
                        await asyncio.to_thread(
                            _append_result, destination, candidate, settings, result
                        )
                        index += 1
                        if index == 1 or index % 10 == 0 or index == len(pending):
                            print(f"[openai-qa] progress={index}/{len(pending)}", flush=True)
                        try:
                            next_candidate = next(candidate_stream)
                        except StopIteration:
                            continue
                        tasks.add(
                            asyncio.create_task(
                                _request(next_candidate, settings, client, semaphore)
                            )
                        )
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        print(f"[openai-qa] done elapsed_s={time.perf_counter() - started:.1f}", flush=True)


async def _run_models(
    model_settings: tuple[dict[str, Any], ...], dry_run: bool, max_concurrency: int
) -> None:
    """Run each independently resumable evaluator with bounded parallel requests.

    Args:
        model_settings: Complete settings for the requested evaluator models.
        dry_run: Print intended call counts without calling the API.
        max_concurrency: Shared cap for all in-flight Responses API calls.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    await asyncio.gather(
        *(_run_model(settings, dry_run, semaphore, max_concurrency) for settings in model_settings)
    )


def run(settings: dict[str, Any], dry_run: bool = False) -> None:
    """Run one or more independently resumable OpenAI QA evaluators.

    Args:
        settings: Shared evaluation settings with one model or a ``models`` list.
        dry_run: Print intended call counts without calling the API.
    """
    max_concurrency = int(settings.get("max_concurrency", 16))
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    model_settings = tuple(
        settings | {"max_concurrency": max_concurrency} for settings in _model_settings(settings)
    )
    asyncio.run(_run_models(model_settings, dry_run, max_concurrency))


def _openai_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Extract a nested OpenAI QA section and inherit its benchmark identity.

    Args:
        settings: Parsed configuration, optionally with an ``openai_qa`` section.

    Returns:
        Settings accepted by :func:`run`.
    """
    openai_qa = settings.get("openai_qa")
    if not openai_qa:
        return settings
    return dict(openai_qa) | {
        "dataset": settings["dataset"],
        "results": settings["output"],
        "methods": [method["name"] for method in settings["methods"]],
        "budgets": settings["budgets"],
    }


def main() -> None:
    """Parse configuration and launch the configured OpenAI QA evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/position_controlled_160.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        settings = _openai_settings(yaml.safe_load(source))
    run(settings, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
