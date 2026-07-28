"""Build a frozen public-source extension for the benchmark dataset."""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.utils.tokens import count_tokens

REPOSITORIES: tuple[dict[str, Any], ...] = (
    {
        "repo": "python/cpython",
        "ref": "v3.13.5",
        "license": "PSF-2.0",
        "kind": "real_code",
        "paths": (
            "Lib/enum.py",
            "Lib/asyncio/tasks.py",
            "Lib/dataclasses.py",
            "Lib/functools.py",
            "Lib/http/client.py",
            "Lib/json/decoder.py",
            "Lib/abc.py",
            "Lib/re/__init__.py",
            "Lib/contextlib.py",
            "Lib/urllib/parse.py",
        ),
    },
    {
        "repo": "pallets/flask",
        "ref": "3.1.1",
        "license": "BSD-3-Clause",
        "kind": "real_api_docs",
        "paths": (
            "docs/api.rst",
            "docs/appcontext.rst",
            "docs/async-await.rst",
            "docs/blueprints.rst",
            "docs/cli.rst",
            "docs/config.rst",
            "docs/errorhandling.rst",
            "docs/quickstart.rst",
            "docs/templating.rst",
            "docs/views.rst",
        ),
    },
    {
        "repo": "fastapi/fastapi",
        "ref": "0.115.13",
        "license": "MIT",
        "kind": "real_api_docs",
        "paths": (
            "fastapi/applications.py",
            "fastapi/encoders.py",
            "fastapi/exceptions.py",
            "fastapi/params.py",
            "fastapi/routing.py",
            "fastapi/security/http.py",
            "fastapi/security/oauth2.py",
            "fastapi/middleware/cors.py",
            "fastapi/dependencies/utils.py",
            "fastapi/openapi/utils.py",
            "fastapi/utils.py",
            "fastapi/security/api_key.py",
            "fastapi/concurrency.py",
            "fastapi/middleware/gzip.py",
        ),
    },
    {
        "repo": "encode/httpx",
        "ref": "0.28.1",
        "license": "BSD-3-Clause",
        "kind": "real_api_docs",
        "paths": (
            "docs/advanced/clients.md",
            "docs/advanced/timeouts.md",
            "docs/api.md",
            "docs/async.md",
            "docs/quickstart.md",
            "httpx/_api.py",
            "httpx/_client.py",
            "httpx/_config.py",
            "httpx/_exceptions.py",
            "httpx/_models.py",
        ),
    },
    {
        "repo": "django/django",
        "ref": "5.2.3",
        "license": "BSD-3-Clause",
        "kind": "real_web_docs",
        "paths": (
            "docs/howto/deployment/index.txt",
            "docs/ref/models/fields.txt",
            "docs/ref/request-response.txt",
            "docs/topics/async.txt",
            "docs/topics/auth/default.txt",
            "docs/topics/db/models.txt",
            "docs/topics/forms/index.txt",
            "docs/topics/http/urls.txt",
            "docs/topics/http/views.txt",
            "django/http/request.py",
        ),
    },
    {
        "repo": "scikit-learn/scikit-learn",
        "ref": "1.7.0",
        "license": "BSD-3-Clause",
        "kind": "real_research_docs",
        "paths": (
            "doc/modules/cross_validation.rst",
            "doc/modules/ensemble.rst",
            "doc/modules/linear_model.rst",
            "doc/modules/model_evaluation.rst",
            "doc/modules/neural_networks_supervised.rst",
            "doc/modules/feature_selection.rst",
            "doc/modules/impute.rst",
            "doc/modules/learning_curve.rst",
            "doc/modules/mixture.rst",
            "doc/modules/naive_bayes.rst",
            "doc/modules/outlier_detection.rst",
        ),
    },
    {
        "repo": "pandas-dev/pandas",
        "ref": "v2.3.0",
        "license": "BSD-3-Clause",
        "kind": "real_data_docs",
        "paths": (
            "doc/source/user_guide/10min.rst",
            "doc/source/user_guide/categorical.rst",
            "doc/source/user_guide/groupby.rst",
            "doc/source/user_guide/io.rst",
            "doc/source/user_guide/merging.rst",
            "doc/source/user_guide/missing_data.rst",
            "doc/source/user_guide/text.rst",
            "doc/source/user_guide/timeseries.rst",
            "doc/source/user_guide/window.rst",
            "doc/source/user_guide/duplicates.rst",
            "doc/source/user_guide/reshaping.rst",
            "doc/source/user_guide/scale.rst",
        ),
    },
    {
        "repo": "OWASP/CheatSheetSeries",
        "ref": "f5c2040a",
        "license": "CC-BY-SA-4.0",
        "kind": "real_security_policy",
        "paths": (
            "cheatsheets/Authentication_Cheat_Sheet.md",
            "cheatsheets/Authorization_Cheat_Sheet.md",
            "cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.md",
            "cheatsheets/Input_Validation_Cheat_Sheet.md",
            "cheatsheets/Logging_Cheat_Sheet.md",
            "cheatsheets/Secrets_Management_Cheat_Sheet.md",
            "cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.md",
            "cheatsheets/Session_Management_Cheat_Sheet.md",
            "cheatsheets/Transport_Layer_Security_Cheat_Sheet.md",
            "cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.md",
        ),
    },
    {
        "repo": "open-telemetry/opentelemetry-specification",
        "ref": "v1.43.0",
        "license": "Apache-2.0",
        "kind": "real_specification",
        "paths": (
            "specification/context/README.md",
            "specification/glossary.md",
            "specification/logs/api.md",
            "specification/logs/sdk.md",
            "specification/metrics/api.md",
            "specification/metrics/sdk.md",
            "specification/overview.md",
            "specification/metrics/README.md",
            "specification/trace/api.md",
            "specification/trace/sdk.md",
        ),
    },
    {
        "repo": "psf/requests",
        "ref": "v2.32.3",
        "license": "Apache-2.0",
        "kind": "real_code",
        "paths": (
            "src/requests/adapters.py",
            "src/requests/api.py",
            "src/requests/auth.py",
            "src/requests/structures.py",
            "src/requests/cookies.py",
            "src/requests/exceptions.py",
            "src/requests/utils.py",
            "src/requests/models.py",
            "src/requests/sessions.py",
            "src/requests/status_codes.py",
        ),
    },
)


def request_text(url: str) -> str:
    """Fetch and decode one public source file."""
    request = urllib.request.Request(url, headers={"User-Agent": "trimwise-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def source_excerpt(text: str) -> str:
    """Extract a source-backed evidence span for retention scoring."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    excerpt = "\n".join(lines[:12]).strip()
    return excerpt[:1200]


def source_title(path: str, text: str) -> str:
    """Find a stable human-readable title or fall back to the source name."""
    heading = re.search(r"^(?:#|=+\s*$)\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if heading:
        return heading.group(1).strip().strip("#")
    symbol = re.search(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.MULTILINE)
    if symbol:
        return symbol.group(1)
    return Path(path).stem.replace("_", " ").replace("-", " ")


def build_case(
    index: int, repository: dict[str, Any], revision: str, path: str, text: str
) -> dict[str, Any]:
    """Create one source-backed benchmark case without inventing verified QA."""
    repo = repository["repo"]
    source_url = f"https://github.com/{repo}/blob/{revision}/{path}"
    excerpt = source_excerpt(text)
    title = source_title(path, text)
    case_id = f"real-public-{index:03d}"
    return {
        "case_id": case_id,
        "document_id": case_id,
        "context": text,
        "query": f"What is the primary topic or responsibility of {Path(path).name}?",
        "gold_answer": title,
        "answer_aliases": [],
        "qa_verified": False,
        "gold_evidence": [
            {
                "id": "source-intro",
                "required": True,
                "start": 0,
                "end": len(excerpt),
                "text": excerpt,
            }
        ],
        "track": "real_source",
        "ordered_steps": [],
        "requirements": [],
        "prohibited_phrases": [],
        "protected_prefix": (
            "Answer only from the supplied public source. Use INSUFFICIENT_CONTEXT "
            "when the source does not support an answer."
        ),
        "protected_suffix": (
            f"Question: What is the primary topic or responsibility of {Path(path).name}?"
        ),
        "metadata": {
            "source_type": repository["kind"],
            "source_repo": repo,
            "source_path": path,
            "source_revision": revision,
            "source_tokens": count_tokens(text),
        },
        "provenance": {
            "kind": "public_permissive_snapshot",
            "license": repository["license"],
            "source_url": source_url,
            "revision": revision,
        },
    }


def build_manifest_row(case: dict[str, Any]) -> dict[str, Any]:
    """Create one auditable source manifest row from a generated case."""
    return {
        "document_id": case["document_id"],
        "source_type": case["metadata"]["source_type"],
        "kind": "public_permissive_snapshot",
        "license": case["provenance"]["license"],
        "source_url": case["provenance"]["source_url"],
        "source_revision": case["provenance"]["revision"],
        "source_tokens": case["metadata"]["source_tokens"],
    }


def main() -> None:
    """Download 100 public source files and merge them into the frozen dataset."""
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "data" / "benchmark_cases.jsonl"
    source_manifest_path = root / "data" / "manifests" / "source_manifest.csv"
    existing_cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_cases = [
        case for case in existing_cases if not case["case_id"].startswith("real-public-")
    ]
    existing_sources = list(
        csv.DictReader(source_manifest_path.read_text(encoding="utf-8").splitlines())
    )
    existing_sources = [
        row for row in existing_sources if not row["document_id"].startswith("real-public-")
    ]
    generated_cases: list[dict[str, Any]] = []
    generated_sources: list[dict[str, Any]] = []

    for repository in REPOSITORIES:
        revision = repository["ref"]
        selected = 0
        for path in repository["paths"]:
            if selected >= 10:
                break
            source_url = f"https://raw.githubusercontent.com/{repository['repo']}/{revision}/{path}"
            try:
                text = request_text(source_url).replace("\x00", "").strip()
            except urllib.error.HTTPError:
                continue
            tokens = count_tokens(text)
            if tokens < 250 or tokens > 25000:
                continue
            case = build_case(len(generated_cases) + 1, repository, revision, path, text)
            generated_cases.append(case)
            generated_sources.append(build_manifest_row(case))
            selected += 1
        if selected != 10:
            raise RuntimeError(
                f"Expected 10 usable files from {repository['repo']}, found {selected}"
            )

    all_cases = existing_cases + generated_cases
    dataset_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in all_cases) + "\n",
        encoding="utf-8",
    )
    fields = [
        "document_id",
        "source_type",
        "kind",
        "license",
        "source_url",
        "source_revision",
        "source_tokens",
    ]
    with source_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_sources + generated_sources)
    print(f"wrote {len(generated_cases)} new real cases; total cases={len(all_cases)}")


if __name__ == "__main__":
    main()
