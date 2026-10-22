"""Load uploaded PDFs into stable, exact page-backed framework-neutral evidence chunks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rag_playground.models import Evidence

MAX_FILES = 5
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CORPUS_CHARS = 500_000
MAX_CHUNKS = 1000


@dataclass(frozen=True)
class ChunkSettings:
    """Define reproducible page-local character windows.

    Attributes:
        size: Maximum characters per chunk, before embedding tokenization.
        overlap: Characters carried into the next chunk on the same page.
    """

    size: int = 1000
    overlap: int = 120

    def __post_init__(self) -> None:
        """Reject invalid windows before any indexing work starts."""
        if not 200 <= self.size <= 2000 or not 0 <= self.overlap < self.size // 2:
            raise ValueError("Chunk size must be 200-2000 characters; overlap must be under half.")


@dataclass(frozen=True)
class LoadedPDFs:
    """Keep exact source text and indexing identity alongside caller-owned metadata.

    Attributes:
        chunks: Exact page-backed evidence documents.
        files: Content hashes, names, page counts, and extraction warnings.
        fingerprint: Identity of this complete corpus and its chunking settings.
    """

    chunks: tuple[Evidence, ...]
    files: tuple[dict[str, object], ...]
    fingerprint: str


def _extract(path: Path) -> dict[str, object]:
    """Run parsing in a timed subprocess, keeping parser output out of application logs.

    Args:
        path: Size-checked local upload.

    Returns:
        Parsed page records from the worker.

    Raises:
        ValueError: If parsing fails, exceeds its timeout, or produces invalid output.
    """
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("pdf_worker.py")), str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=25,
            check=False,
        )
        result = json.loads(completed.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ValueError("PDF parsing failed or exceeded the 25-second limit.") from error
    if not isinstance(result, dict):
        raise ValueError("PDF parsing returned an invalid result.")
    if completed.returncode or "error" in result:
        raise ValueError(str(result.get("error", "PDF parsing failed.")))
    return result


def _page_chunks(text: str, settings: ChunkSettings) -> list[tuple[int, int]]:
    """Choose exact windows, preferring a sentence or paragraph boundary near the end.

    Args:
        text: One extracted page's exact Python string.
        settings: Character window and overlap.

    Returns:
        Ordered, potentially overlapping start/end offsets within that page.
    """
    spans = []
    start = 0
    while start < len(text):
        end = min(start + settings.size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(mark, start + settings.size // 2, end) for mark in ("\n", ". ", "。")
            )
            if boundary >= 0:
                end = boundary + 1
        if text[start:end].strip():
            spans.append((start, end))
        if end == len(text):
            break
        start = end - settings.overlap
    return spans


def load_pdfs(paths: list[Path], settings: ChunkSettings) -> LoadedPDFs:
    """Validate the complete upload set and create stable chunks without mutating files.

    Args:
        paths: One to five user-selected PDFs; duplicate content is indexed once.
        settings: Page-local chunking configuration.

    Returns:
        A complete replacement corpus, with original page-local source coordinates.

    Raises:
        ValueError: If any PDF or the aggregate corpus violates a limit.
    """
    if not 1 <= len(paths) <= MAX_FILES:
        raise ValueError(f"Choose between one and {MAX_FILES} PDFs.")
    chunks = []
    files = []
    seen: set[str] = set()
    total_chars = 0
    for path in paths:
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError("Choose existing PDF files.")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{path.name}: each PDF must be at most 10 MiB.")
        data = path.read_bytes()
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"{path.name}: the file does not have a PDF header.")
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        parsed = _extract(path)
        pages = parsed["pages"]
        assert isinstance(pages, list)
        files.append(
            {
                "name": path.name,
                "sha256": digest,
                "text_pages": len(pages),
                "skipped_pages": parsed["skipped_pages"],
                "bytes": len(data),
            }
        )
        for page in pages:
            text = page["text"]
            total_chars += len(text)
            if total_chars > MAX_CORPUS_CHARS:
                raise ValueError("Combined extracted text may contain at most 500,000 characters.")
            for start, end in _page_chunks(text, settings):
                identity = f"{digest}:{page['page']}:{start}:{end}"
                chunks.append(
                    Evidence(
                        id=hashlib.sha256(identity.encode()).hexdigest(),
                        page_content=text[start:end],
                        metadata={
                            "filename": path.name,
                            "document_sha256": digest,
                            "page": page["page"],
                            "page_start": start,
                            "page_end": end,
                        },
                    )
                )
                if len(chunks) > MAX_CHUNKS:
                    raise ValueError(
                        "This upload produces over 1,000 chunks. Use fewer or smaller PDFs."
                    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "files": files,
                "chunk_size": settings.size,
                "chunk_overlap": settings.overlap,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return LoadedPDFs(tuple(chunks), tuple(files), fingerprint)
