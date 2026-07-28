"""Lightweight answer-quality metrics for optional downstream QA runs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .evidence import overlap

_ASSISTANT_MARKER = "<|im_start|>assistant\n"


def assistant_completion(text: str) -> str:
    """Return only the final assistant turn from a chat-generation result.

    Args:
        text: Generated text that may include the chat prompt.

    Returns:
        The assistant continuation, or the supplied text when it already is one.
    """
    return text.rsplit(_ASSISTANT_MARKER, 1)[-1].strip()


def _answer_match(reference: str, answer: str) -> bool:
    """Return whether an answer contains every normalized key-fact token.

    Args:
        reference: Gold answer or accepted answer alias.
        answer: Assistant continuation to evaluate.

    Returns:
        Whether every reference token, including repeated tokens, occurs in the answer.
    """
    reference_tokens = Counter(re.findall(r"[\w]+(?::[\w]+)*", reference.casefold()))
    answer_tokens = Counter(re.findall(r"[\w]+(?::[\w]+)*", answer.casefold()))
    return bool(reference_tokens) and not (reference_tokens - answer_tokens)


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Score only an assistant continuation against one benchmark case.

    Args:
        case: Benchmark case containing gold-answer and safety fields.
        answer: Generated text, with or without its chat prompt.

    Returns:
        Strict exact-match, answer-match, token-overlap, and prohibited-phrase metrics.
    """
    answer = assistant_completion(answer)
    gold = str(case.get("gold_answer", ""))
    candidates = [gold, *(str(alias) for alias in case.get("answer_aliases", []))]
    normalized_answer = " ".join(answer.casefold().split())
    exact_match = any(
        " ".join(candidate.casefold().split()) in normalized_answer
        for candidate in candidates
        if candidate
    )
    answer_match = any(_answer_match(candidate, answer) for candidate in candidates if candidate)
    recall, precision, f1 = overlap(gold, answer)
    prohibited = [str(phrase) for phrase in case.get("prohibited_phrases", [])]
    prohibited_hits = [
        phrase for phrase in prohibited if " ".join(phrase.casefold().split()) in normalized_answer
    ]
    return {
        "answer_exact_match": exact_match,
        "answer_match": answer_match,
        "answer_token_recall": recall,
        "answer_token_precision": precision,
        "answer_token_f1": f1,
        "answer_prohibited_phrase": bool(prohibited_hits),
        "answer_prohibited_phrase_hits": prohibited_hits,
    }
