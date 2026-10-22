"""Extract bounded PDF page text in a disposable process, without running PDF code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader

MAX_PAGES = 100
MAX_PAGE_CHARS = 100_000
MAX_DOCUMENT_CHARS = 500_000


def extract_pages(path: Path) -> dict[str, object]:
    """Read text pages while rejecting encrypted, empty, and oversized documents.

    Args:
        path: PDF file already checked for upload size by the parent.

    Returns:
        Page-numbered text and a list of pages without usable text.

    Raises:
        ValueError: If the document is unsupported or exceeds extraction limits.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported. Supply an unencrypted copy.")
    if not reader.pages:
        raise ValueError("This PDF has no pages.")
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDFs may contain at most {MAX_PAGES} pages.")
    pages = []
    skipped = []
    total = 0
    for number, page in enumerate(reader.pages, 1):
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 4_000_000:
            raise ValueError("A PDF page has an oversized decoded content stream.")
        text = page.extract_text() or ""
        total += len(text)
        if len(text) > MAX_PAGE_CHARS or total > MAX_DOCUMENT_CHARS:
            raise ValueError("Extracted PDF text exceeds this playground's safety limit.")
        if text.strip():
            pages.append({"page": number, "text": text})
        else:
            skipped.append(number)
    if not pages:
        raise ValueError(
            "No selectable text was found. Scanned/image-only PDFs need OCR, outside this demo."
        )
    return {"pages": pages, "skipped_pages": skipped}


def main() -> None:
    """Apply a process memory ceiling where supported and emit a bounded JSON result."""
    if sys.platform != "win32":
        import resource

        ceiling = 768 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
    try:
        result = extract_pages(Path(sys.argv[1]))
        print(json.dumps(result, ensure_ascii=False))
    except ValueError as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
    except Exception:
        print(json.dumps({"error": "This PDF could not be parsed within the extraction limits."}))
        sys.exit(1)


if __name__ == "__main__":
    main()
