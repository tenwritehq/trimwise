"""Check PDF validation, fair comparison, provenance, and session cleanup without models."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from conftest import TestBackend, TestEmbeddings
from pypdf import PdfReader, PdfWriter
from rag_playground.comparison import compare
from rag_playground.documents import ChunkSettings, _page_chunks, load_pdfs
from rag_playground.evaluation import evaluate
from rag_playground.exports import export_data
from rag_playground.models import RunSettings
from rag_playground.presentation import answers, evidence
from rag_playground.retrieval import Corpus, CorpusBuilder
from rag_playground.session import FIXTURE, Session


def test_fixture_pages_and_labels_are_exact() -> None:
    """Keep reviewed QA support labels tied to the actual extracted PDF text."""
    reader = PdfReader(FIXTURE)
    cases = json.loads(FIXTURE.with_name("questions.json").read_text())
    assert len(reader.pages) == 3
    assert len(cases) == 5
    assert any(case["support"] is None for case in cases)
    for case in cases:
        if support := case["support"]:
            page = reader.pages[support["page"] - 1].extract_text()
            assert page[support["start"] : support["end"]] == support["quote"]


def test_pdf_duplicate_content_and_chunk_spans() -> None:
    """Index repeated uploads once and preserve exact page-local chunk coordinates."""
    loaded = load_pdfs([FIXTURE, FIXTURE], ChunkSettings())
    assert len(loaded.files) == 1
    pages = [page.extract_text() for page in PdfReader(FIXTURE).pages]
    for document in loaded.chunks:
        metadata = document.metadata
        assert (
            pages[metadata["page"] - 1][metadata["page_start"] : metadata["page_end"]]
            == document.page_content
        )
    again = load_pdfs([FIXTURE], ChunkSettings())
    assert [document.id for document in loaded.chunks] == [document.id for document in again.chunks]


@pytest.mark.parametrize("text", ["x" * 1400, "句子。" * 500, "abc\n" * 350])
def test_chunk_windows_cover_source(text: str) -> None:
    """Bound chunk sizes and retain every source character even without whitespace.

    Args:
        text: Long unpunctuated, CJK, or line-oriented input.
    """
    spans = _page_chunks(text, ChunkSettings(300, 40))
    covered = set()
    for start, end in spans:
        assert 0 < end - start <= 300
        covered.update(range(start, end))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize("kind", ["empty", "encrypted", "image-only", "malformed", "oversized"])
def test_unsupported_pdfs_are_explicit(kind: str, tmp_path: Path) -> None:
    """Reject unsupported files before indexing or generation.

    Args:
        kind: Invalid PDF partition.
        tmp_path: Isolated test directory.
    """
    path = tmp_path / "input.pdf"
    writer = PdfWriter()
    if kind != "empty":
        writer.add_blank_page(width=100, height=100)
    if kind == "encrypted":
        writer.encrypt("test-only-password")
    writer.write(path)
    if kind == "malformed":
        path.write_bytes(b"not a PDF")
    if kind == "oversized":
        with path.open("wb") as stream:
            stream.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        load_pdfs([path], ChunkSettings())


def test_frozen_retrieval_and_shared_budget(corpus: Corpus, monkeypatch: Any) -> None:
    """Retrieve once and preserve exactly the original ordered baseline evidence.

    Args:
        corpus: Native vector index with test embeddings.
        monkeypatch: Scoped call recorder.
    """
    calls = []
    retrieve = corpus.retrieve

    def recorded(question: str, top_k: int) -> Any:
        """Record the single retrieval boundary.

        Args:
            question: Retrieval query.
            top_k: Requested chunk count.

        Returns:
            Original native retriever result.
        """
        calls.append((question, top_k))
        return retrieve(question, top_k)

    monkeypatch.setattr(corpus, "retrieve", recorded)
    before = [asdict(document) for document in corpus.loaded.chunks]
    backend = TestBackend()
    run = compare(corpus, backend, ("retry deadline", RunSettings(top_k=2, budget=80)))
    assert calls == [("retry deadline", 2)]
    assert [call[0] for call in backend.calls] == ["retry deadline", "retry deadline"]
    assert backend.calls[0][1] == run.contexts["baseline"]
    assert all(document.page_content in backend.calls[0][1] for document in run.original)
    assert backend.calls[1][1] == run.contexts["trimwise"]
    assert sum(backend.count(document.page_content) for document in run.compressed) <= 80
    assert [asdict(document) for document in corpus.loaded.chunks] == before
    for document in run.compressed:
        source = run.original[document.metadata["trimwise"]["source_index"]]
        for span in document.metadata["trimwise"]["spans"]:
            assert source.page_content[span["start"] : span["end"]] in document.page_content


def test_baseline_overflow_never_generates_or_truncates(corpus: Corpus) -> None:
    """Refuse the pair when the complete baseline does not fit the answer window.

    Args:
        corpus: Native vector index with controlled evidence.
    """
    backend = TestBackend(context_limit=200)
    run = compare(corpus, backend, ("retry", RunSettings(top_k=2, budget=32)))
    assert backend.calls == []
    assert set(run.errors) == {"baseline", "trimwise"}
    assert "not truncated" in run.errors["baseline"]
    assert all(document.page_content in run.contexts["baseline"] for document in run.original)


def test_native_adapter_is_called_and_preserves_scores(corpus: Corpus, monkeypatch: Any) -> None:
    """Prove each example invokes its real adapter and keeps native retrieval scores.

    Args:
        corpus: Real native index with synthetic embeddings.
        monkeypatch: Scoped recorder around the actual integration API.
    """
    module = import_module(type(corpus).__module__)
    calls = []
    if corpus.framework == "LangChain":
        factory = module.create_trimwise_runnable

        def runnable(limit: int, **kwargs: Any) -> Any:
            """Record the native runnable factory while preserving real compression."""
            calls.append(limit)
            return factory(limit, **kwargs)

        monkeypatch.setattr(module, "create_trimwise_runnable", runnable)
    else:
        method = module.TrimwiseNodePostprocessor.postprocess_nodes

        def postprocess(processor: Any, nodes: list[Any], **kwargs: Any) -> Any:
            """Require real TextNode inputs and invoke the actual native postprocessor."""
            assert all(type(item.node) is module.TextNode for item in nodes)
            calls.append(processor.limit)
            return method(processor, nodes, **kwargs)

        monkeypatch.setattr(module.TrimwiseNodePostprocessor, "postprocess_nodes", postprocess)
    run = compare(corpus, TestBackend(), ("retry deadline", RunSettings(top_k=2, budget=80)))
    assert calls == [80]
    assert export_data(run)["framework"] == corpus.framework
    original = {item.id: item for item in run.original}
    assert all(item.score == original[item.id].score for item in run.compressed)


def test_changed_retrieval_cannot_replace_native_evidence(corpus: Corpus) -> None:
    """Reject stale or modified display records before invoking the native adapter."""
    documents = corpus.retrieve("retry", 2)
    documents[0].page_content = "replacement evidence"
    with pytest.raises(ValueError, match="frozen retrieval"):
        corpus.compress(documents, ("retry", RunSettings()), len)


def test_llamaindex_native_fields_reach_postprocessor(corpus: Any, monkeypatch: Any) -> None:
    """Preserve native-only node fields and keep adapter mutation off the frozen result."""
    if corpus.framework != "LlamaIndex":
        pytest.skip("LlamaIndex-specific native node contract")
    module = import_module(type(corpus).__module__)
    documents = corpus.retrieve("retry", 2)
    corpus._retrieved[0].node.start_char_idx = 37
    original = module.TrimwiseNodePostprocessor.postprocess_nodes
    calls = []

    def postprocess(processor: Any, nodes: list[Any], **kwargs: Any) -> Any:
        """Observe a native-only field before calling the real postprocessor."""
        calls.append(nodes[0].node.start_char_idx)
        nodes[0].node.start_char_idx = 99
        return original(processor, nodes, **kwargs)

    monkeypatch.setattr(module.TrimwiseNodePostprocessor, "postprocess_nodes", postprocess)
    corpus.compress(documents, ("retry", RunSettings(budget=80)), len)
    assert calls == [37]
    assert corpus._retrieved[0].node.start_char_idx == 37


def test_empty_results_do_not_generate(corpus: Corpus, monkeypatch: Any) -> None:
    """Report a retrieval failure separately from any compression or model failure.

    Args:
        corpus: Corpus whose retriever is replaced with an empty test result.
        monkeypatch: Scoped fault injection.
    """
    monkeypatch.setattr(corpus, "retrieve", lambda *args: ())
    backend = TestBackend()
    run = compare(corpus, backend, ("retry", RunSettings()))
    assert not backend.calls
    assert "No evidence was retrieved" in run.errors["baseline"]


def test_failures_and_content_export_policy(corpus: Corpus) -> None:
    """Retain both failed conditions without leaking exception bodies or private content.

    Args:
        corpus: Native vector index test corpus.
    """
    backend = TestBackend()
    backend.fail = True
    run = compare(corpus, backend, ("private retry question", RunSettings(top_k=2)))
    assert len(backend.calls) == 2
    assert len(run.errors) == 2
    exported = export_data(run)
    serialized = json.dumps(exported)
    assert "secret-key-sentinel" not in serialized
    assert "private retry question" not in serialized
    assert "Gardens have flowers" not in serialized
    assert "source-0.pdf" not in serialized
    complete = export_data(run, include_content=True)
    assert complete["question"] == "private retry question"
    assert complete["contexts"] == run.contexts


def test_safe_answer_and_provenance_html(corpus: Corpus) -> None:
    """Escape uploaded/model content and keep highlights tied to exact source spans.

    Args:
        corpus: Test evidence containing a deliberately hostile filename.
    """
    backend = TestBackend()
    run = compare(corpus, backend, ("retry", RunSettings(top_k=2, budget=80)))
    from dataclasses import replace

    run.original[0].metadata["filename"] = '<img src=x onerror="alert(1)">'
    run.generations["baseline"] = replace(
        run.generations["baseline"], text="<script>alert(1)</script> Answer [S1]."
    )
    rendered = answers(run) + evidence(run)
    assert "&lt;img" in rendered and "&lt;script&gt;" in rendered
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert '<a class="citation" href="#excerpt-S1">' in rendered
    assert "<mark>" in rendered and "<del>" in rendered


def test_reference_checks_separate_retrieval_from_retention(corpus: Corpus) -> None:
    """Label a missing quote explicitly rather than inventing an accuracy score.

    Args:
        corpus: Native vector index test corpus.
    """
    run = compare(corpus, TestBackend(), ("retry", RunSettings(top_k=2, budget=80)))
    checked = evaluate(run, corpus, ("A mocked answer [S1].", "not present anywhere"))
    assert checked["baseline"]["normalized_exact_match"] is True
    assert checked["supporting_quote"]["found_in_indexed_chunk"] is False
    assert "not semantic" in checked["limitations"]


def test_index_replacement_reset_and_session_isolation(
    runtime_root: Path, builder: CorpusBuilder
) -> None:
    """Keep repeated uploads idempotent, failed indexing atomic, and resets isolated.

    Args:
        runtime_root: Example-owned test workspace.
        builder: Native framework index factory.
    """
    first, second = Session(runtime_root, builder), Session(runtime_root, builder)
    try:
        one = first.index([FIXTURE], ChunkSettings(), TestEmbeddings())
        assert first.index([FIXTURE], ChunkSettings(), TestEmbeddings()) is one
        two = second.index([FIXTURE], ChunkSettings(), TestEmbeddings())
        invalid = runtime_root / "bad.pdf"
        invalid.write_text("not a PDF")
        with pytest.raises(ValueError):
            first.index([invalid], ChunkSettings(), TestEmbeddings())
        assert first.corpus is one
        first.reset()
        assert first.corpus is None
        assert two.retrieve("retry", 2)
        replacement = first.index([FIXTURE], ChunkSettings(500, 40), TestEmbeddings())
        assert replacement.loaded.fingerprint != one.loaded.fingerprint
        assert len(replacement.loaded.chunks) > len(one.loaded.chunks)
    finally:
        first.close()
        second.close()


def test_session_rejects_arbitrary_paths(
    runtime_root: Path, tmp_path_factory: Any, builder: CorpusBuilder
) -> None:
    """Refuse direct callback paths outside uploads and the bundled fixture.

    Args:
        runtime_root: Allowed application directory.
        builder: Native framework index factory.
        tmp_path_factory: Fixture for an unrelated sibling directory.
    """
    outside = tmp_path_factory.mktemp("outside") / "private.pdf"
    shutil.copyfile(FIXTURE, outside)
    session = Session(runtime_root, builder)
    try:
        with pytest.raises(ValueError, match="upload control"):
            session.index([outside], ChunkSettings(), TestEmbeddings())
        with pytest.raises(ValueError, match="Run a comparison"):
            session.export(False)
    finally:
        session.close()


def test_pending_chunk_changes_require_reindexing(
    runtime_root: Path, builder: CorpusBuilder
) -> None:
    """Refuse stale chunk controls before acquiring a real model.

    Args:
        runtime_root: Application-owned test directory.
        builder: Native framework index factory.
    """
    session = Session(runtime_root, builder)
    try:
        session.index([FIXTURE], ChunkSettings(), TestEmbeddings())
        with pytest.raises(ValueError, match="changed"):
            session.compare_selected(
                ("retry", RunSettings()),
                ([FIXTURE], ChunkSettings(500, 40)),
            )
        assert session.run is None
    finally:
        session.close()


def test_reset_removes_only_current_session_exports(
    runtime_root: Path, builder: CorpusBuilder
) -> None:
    """Delete owned exports and invalidate the run without touching a sibling's files.

    Args:
        runtime_root: Application-owned test directory.
        builder: Native framework index factory.
    """
    first, second = Session(runtime_root, builder), Session(runtime_root, builder)
    try:
        for session in (first, second):
            session.index([FIXTURE], ChunkSettings(), TestEmbeddings())
            session.run_comparison("retry", RunSettings(top_k=1, budget=80), TestBackend())
        own, sibling = first.export(False), second.export(False)
        assert own.is_file() and sibling.is_file()
        first.reset()
        assert not own.exists() and sibling.is_file()
        with pytest.raises(ValueError, match="Run a comparison"):
            first.export(False)
    finally:
        first.close()
        second.close()
