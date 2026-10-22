"""Exercise the local HTTP boundary without downloading a tokenizer or model."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from rag_playground import backends


class TokenizerDouble:
    """Make native prompt accounting transparent to the HTTP contract tests."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Count characters as synthetic tokens, independently of Ollama usage.

        Args:
            text: Exact supplied text.
            kwargs: Native tokenizer options, unused by this double.

        Returns:
            One synthetic token per character.
        """
        return list(range(len(text)))

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Expose the complete system/user content with explicit framing.

        Args:
            messages: System and user messages.
            kwargs: Native template options, unused by this double.

        Returns:
            Deterministic synthetic native prompt.
        """
        return "<chat>" + "|".join(item["content"] for item in messages) + "</chat>"


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace only HTTP transport and tokenizer while keeping backend logic real.

    Args:
        monkeypatch: Scoped dependency replacement.

    Returns:
        Mutable runtime faults and observed requests.
    """
    state: dict[str, Any] = {"requests": [], "short_prompt": False, "missing_usage": False}
    client_type = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        """Return explicit local runtime metadata and inspect generation payloads.

        Args:
            request: Actual HTTP request constructed by the backend.

        Returns:
            Controlled Ollama-shaped response.
        """
        import json

        body = json.loads(request.content) if request.content else {}
        state["requests"].append((request.url.path, body))
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/tags":
            data = {
                "models": [
                    {
                        "name": backends.LOCAL_MODEL,
                        "digest": state.get("digest", backends.MODEL_DIGEST[7:]),
                    }
                ]
            }
        elif request.url.path == "/api/show":
            data = {
                "model_info": {"qwen3.context_length": 262144},
                "details": {"quantization_level": "Q4_K_M"},
            }
        elif request.url.path == "/api/version":
            data = {"version": "test-runtime"}
        elif request.url.path == "/api/ps":
            data = {
                "models": [
                    {
                        "name": backends.LOCAL_MODEL,
                        "size_vram": 0 if state.get("existing_cpu") else 1234,
                    }
                ]
            }
        elif "prompt" not in body:
            if body.get("keep_alive") == 0:
                state["existing_cpu"] = False
            data = {"done": True}
        else:
            data = {
                "done": True,
                "model": backends.LOCAL_MODEL,
                "response": "Answer [S1].",
                "prompt_eval_count": 1 if state["short_prompt"] else len(body["prompt"]),
                "eval_count": 5,
                "done_reason": "stop",
            }
            if state["missing_usage"]:
                del data["eval_count"]
        return httpx.Response(200, json=data)

    def local_client(**kwargs: Any) -> httpx.Client:
        """Use real HTTP serialization with an in-process controlled transport.

        Args:
            kwargs: Backend client options, including loopback and proxy policy.

        Returns:
            A client which cannot perform network I/O.
        """
        assert kwargs["trust_env"] is False
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(backends.httpx, "Client", local_client)
    monkeypatch.setattr(backends, "native_tokenizer", TokenizerDouble)
    return state


def test_runtime_uses_raw_native_prompt_and_cpu_only(runtime: dict[str, Any]) -> None:
    """Keep CPU-only inference mandatory across preload and actual generation.

    Args:
        runtime: Controlled local runtime transport.
    """
    backend = backends.OllamaBackend()
    try:
        result = backend.generate("Question?", "Evidence [S1]", 64)
        bodies = [body for path, body in runtime["requests"] if path == "/api/generate"]
        assert all(body["options"]["num_gpu"] == 0 for body in bodies)
        body = bodies[-1]
        assert body["raw"] is True and body["stream"] is False
        assert backends.INSTRUCTIONS in body["prompt"]
        assert body["options"]["num_predict"] == 64
        assert result.prompt_tokens == backend.prompt_count("Question?", "Evidence [S1]")
        assert backend.info["device_choice"] == "CPU only"
    finally:
        backend.close()


@pytest.mark.parametrize("fault", ["short_prompt", "missing_usage"])
def test_runtime_refuses_unverifiable_prompt_usage(runtime: dict[str, Any], fault: str) -> None:
    """Reject a silently shortened prompt or omitted usage instead of reporting a fair pair.

    Args:
        runtime: Controlled local runtime transport.
        fault: Runtime accounting failure.
    """
    runtime[fault] = True
    backend = backends.OllamaBackend()
    try:
        with pytest.raises(ValueError):
            backend.generate("Question?", "Full evidence", 64)
    finally:
        backend.close()


def test_runtime_rejects_changed_model_digest(runtime: dict[str, Any]) -> None:
    """Refuse changed weights before preloading or generation.

    Args:
        runtime: Controlled local runtime transport.
    """
    runtime["digest"] = "a" * 64
    with pytest.raises(ValueError, match="digest differs"):
        backends.OllamaBackend()
    assert not any(path == "/api/generate" for path, body in runtime["requests"])


def test_runtime_refuses_oversized_prompt_before_generation(runtime: dict[str, Any]) -> None:
    """Preserve the complete baseline by refusing rather than truncating it.

    Args:
        runtime: Controlled local runtime transport.
    """
    backend = backends.OllamaBackend()
    try:
        with pytest.raises(ValueError, match="exceeds"):
            backend.generate("Question?", "x" * 4096, 64)
        assert not any("prompt" in body for path, body in runtime["requests"])
    finally:
        backend.close()
