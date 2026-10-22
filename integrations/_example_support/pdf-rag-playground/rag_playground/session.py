"""Own one browser session's corpus, latest comparison, and explicit exports."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from pathlib import Path
from time import perf_counter

from rag_playground.backends import AnswerBackend, OllamaBackend
from rag_playground.comparison import compare
from rag_playground.documents import MAX_FILE_BYTES, MAX_FILES, ChunkSettings, load_pdfs
from rag_playground.exports import export_data
from rag_playground.models import Comparison, RunSettings
from rag_playground.retrieval import Corpus, CorpusBuilder, Embeddings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "field-guide.pdf"


class Session:
    """Keep every collection, result, and export bound to one browser session."""

    def __init__(self, runtime_root: Path, builder: CorpusBuilder) -> None:
        """Create an example-owned export directory without changing user files.

        Args:
            runtime_root: Temporary application directory, also containing Gradio uploads.
            builder: Native framework index factory supplied by the example.
        """
        self.builder = builder
        self.runtime_root = runtime_root.resolve()
        self.directory = tempfile.TemporaryDirectory(prefix="session-", dir=self.runtime_root)
        self.lock = threading.RLock()
        self.corpus: Corpus | None = None
        self.run: Comparison | None = None
        self.selection: tuple[tuple[str, str], ...] = ()
        self.chunk_settings: ChunkSettings | None = None

    def _signature(self, paths: list[Path]) -> tuple[tuple[str, str], ...]:
        """Check upload ownership and identify changed files without reading arbitrary paths.

        Args:
            paths: Gradio temporary uploads or the bundled public fixture.

        Returns:
            Ordered filename/content-hash identities, with exact duplicates removed.
        """
        if not 1 <= len(paths) <= MAX_FILES:
            raise ValueError("Choose between one and five PDFs.")
        result = []
        for path in paths:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.runtime_root) and resolved != FIXTURE:
                raise ValueError("Select PDFs through the upload control.")
            if not resolved.is_file() or resolved.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Each uploaded PDF must exist and be at most 10 MiB.")
            item = (path.name, hashlib.sha256(resolved.read_bytes()).hexdigest())
            if item not in result:
                result.append(item)
        return tuple(result)

    def index(
        self,
        paths: list[Path],
        settings: ChunkSettings,
        embeddings: Embeddings | None = None,
    ) -> Corpus:
        """Atomically replace the complete selected corpus, preserving the old one on failure.

        Args:
            paths: Complete selected set, not an incremental append operation.
            settings: Requested page-local chunking settings.
            embeddings: Explicit deterministic test embeddings, or the real local model.

        Returns:
            Active fully built corpus.
        """
        with self.lock:
            self.run = None
            signature = self._signature(paths)
            if (
                self.corpus is not None
                and signature == self.selection
                and settings == self.chunk_settings
            ):
                return self.corpus
            loaded = load_pdfs(paths, settings)
            replacement = self.builder(loaded, embeddings)
            previous = self.corpus
            self.corpus = replacement
            self.selection = signature
            self.chunk_settings = settings
            if previous is not None:
                previous.close()
            return replacement

    def run_comparison(
        self,
        question: str,
        settings: RunSettings,
        backend: AnswerBackend,
    ) -> Comparison:
        """Serialize comparison state so repeated UI actions cannot mix runs or corpora.

        Args:
            question: Question for both answer conditions.
            settings: Identical retrieval/output settings for the pair.
            backend: Real generation backend or explicitly supplied test double.

        Returns:
            Latest comparison record, including any condition failures.
        """
        with self.lock:
            self.run = None
            if self.corpus is None:
                raise ValueError("Index at least one PDF before comparing answers.")
            self.run = compare(self.corpus, backend, (question, settings))
            return self.run

    def compare_selected(
        self,
        request: tuple[str, RunSettings],
        selection: tuple[list[Path], ChunkSettings],
    ) -> Comparison:
        """Check pending index changes and acquire exactly one backend for both conditions.

        Args:
            request: Question and comparison settings; generation is always CPU-only.
            selection: Current upload/chunk controls, checked against the active corpus.

        Returns:
            Latest result, with model acquisition recorded separately.
        """
        with self.lock:
            self.run = None
            paths, chunk_settings = selection
            if self.corpus is None:
                raise ValueError("Index your PDFs, or load the example, before comparing.")
            if self._signature(paths) != self.selection or chunk_settings != self.chunk_settings:
                raise ValueError(
                    "The selected files or chunk settings changed. Index PDFs again first."
                )
            question, settings = request
            started = perf_counter()
            backend = OllamaBackend()
            acquired = perf_counter() - started
            try:
                run = self.run_comparison(question, settings, backend)
                run.model_load_seconds = acquired
                return run
            finally:
                backend.close()

    def export(self, include_content: bool) -> Path:
        """Write only an explicitly requested export inside this session's directory.

        Args:
            include_content: Permission to include private question, answers, names, and evidence.

        Returns:
            Session-owned JSON file for the download component.
        """
        with self.lock:
            if self.run is None:
                raise ValueError("Run a comparison before exporting.")
            path = Path(self.directory.name) / f"comparison-{self.run.run_id}.json"
            path.write_text(
                json.dumps(export_data(self.run, include_content=include_content), indent=2),
                encoding="utf-8",
            )
            return path

    def reset(self) -> None:
        """Delete this session's native index, results, and generated export files."""
        with self.lock:
            if self.corpus is not None:
                self.corpus.close()
            self.corpus = None
            self.run = None
            self.selection = ()
            self.chunk_settings = None
            for path in Path(self.directory.name).glob("comparison-*.json"):
                path.unlink()

    def close(self) -> None:
        """Release only session-owned resources when Gradio expires the session."""
        with self.lock:
            self.reset()
            self.directory.cleanup()
