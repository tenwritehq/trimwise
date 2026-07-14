# Getting Started

This guide takes you from installation to a prompt-ready excerpt. You will trim a document without
a query, focus an excerpt on a task, process several sources, choose the right budget, and use the
asynchronous API.

Trimwise accepts a complete Python `str` and returns a `TrimResult`. Its `text` field contains the
excerpt to place in your prompt. Retained fragments keep their original wording and source order,
and the measured result never exceeds the limit you supplied.

## 1. Install Trimwise

Trimwise supports Python 3.10 through 3.14. Use an isolated environment so its dependencies do not
affect other projects.

### With `pip`

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then choose the installation that matches your application:

```bash
# Structural, lexical, or your own embedding callback
python -m pip install trimwise

# Trimwise-managed semantic model on CPU
python -m pip install "trimwise[semantic]"

# Trimwise-managed semantic model on an NVIDIA GPU
python -m pip install "trimwise[semantic-gpu]"
```

### With `uv`

Run one command from your project directory. `uv` records the dependency and manages the project
environment for you:

```bash
# Structural, lexical, or your own embedding callback
uv add trimwise

# Trimwise-managed semantic model on CPU
uv add "trimwise[semantic]"

# Trimwise-managed semantic model on an NVIDIA GPU
uv add "trimwise[semantic-gpu]"
```

Do not install the CPU and GPU semantic extras together because their FastEmbed runtime packages
conflict. The GPU option also requires compatible CUDA and cuDNN libraries.

The normal `trimwise` installation is enough for most users. It includes structural and lexical
selection and can also run semantic or hybrid selection through your own embedding callback. The
semantic extras are needed only when Trimwise should provide the embedding model.

### Verify the installation

```bash
python -c "from trimwise import Trimmer; r = Trimmer().trim('alpha beta gamma', 2, unit='words'); assert r.output_count <= 2; print(r.text)"
```

This check uses only the lightweight core and confirms that the public API returns an excerpt
within the requested limit.

## 2. Trim a document without a query

Start with `auto`, the default strategy. Without a query, it resolves to `structural` and builds a
balanced overview from across the document.

```python
from trimwise import Trimmer

document = """
# Project Zephyr

Zephyr began as an internal reporting experiment in January 2025.

## Adoption

Twenty-three teams now use it for weekly operational reviews.
The finance group uses it to compare forecast changes.

## Current decision

The steering group approved a public beta for September 2026.
Documentation and access controls must be completed before launch.
"""

result = Trimmer().trim(
    document,
    limit=45,
    unit="words",
)

print(result.text)
assert result.output_count <= 45
assert result.strategy.value == "structural"
```

Structural mode does not take a fixed portion from the beginning, middle, and end. It protects
useful opening and closing units when they fit, gives each Markdown section an initial share, and
then spends remaining room on central, nonredundant material. This is why it can retain a later
decision instead of returning only the introduction.

If the full input already fits, Trimwise returns it unchanged. It skips Markdown parsing, ranking,
and embedding work on that path.

## 3. Focus the excerpt on a task

Pass the same question or task that the eventual LLM prompt needs to answer:

```python
from trimwise import Trimmer

document = """
# Project Zephyr

Zephyr began as an internal reporting experiment in January 2025.

## Adoption

Twenty-three teams now use it for weekly operational reviews.

## Current decision

The steering group approved a public beta for September 2026.
Documentation and access controls must be completed before launch.
"""

result = Trimmer().trim(
    document,
    limit=35,
    unit="words",
    query="What was approved, and what must happen before launch?",
)

print(result.text)
assert result.output_count <= 35
assert result.strategy.value == "lexical"
```

With a query, `auto` resolves to `lexical`. BM25 favors source units containing the query's exact
terms, while diversity-aware selection reduces repeated-looking evidence. A relevant passage's
section heading is included when the complete result still fits.

Use an explicit strategy when the task needs different matching behavior:

| Strategy | Start here when |
| --- | --- |
| `structural` | You need a queryless overview of the whole document |
| `lexical` | Names, IDs, URLs, error codes, and exact wording matter |
| `semantic` | The document may express the answer with different words or in another supported language |
| `hybrid` | Exact identifiers and paraphrased explanations both matter |

`lexical`, `semantic`, and `hybrid` require a nonblank query. `auto` never enables embeddings by
itself, even when a semantic extra or callback is available. This keeps the common path fast and
prevents an unexpected model download.

## 4. Trim several sources before prompt assembly

A common agentic workflow collects several potentially useful sources and gives each one a small
share of the prompt. Trim each source independently so one long document cannot consume all the
available evidence space.

```python
from trimwise import Trimmer

task = "Compare the reported causes of the outage."
sources = [
    {
        "label": "Incident report",
        "text": "The API failed after workers exhausted the database connection pool. " * 30,
    },
    {
        "label": "Engineering notes",
        "text": "A retry loop ignored backoff settings and created excess connections. " * 30,
    },
    {
        "label": "Support summary",
        "text": "Customers first reported timeouts in the billing workflow. " * 30,
    },
]

trimmer = Trimmer()
evidence = []
for source in sources:
    excerpt = trimmer.trim(
        source["text"],
        limit=120,
        query=task,
    ).text
    evidence.append(f"## {source['label']}\n\n{excerpt}")

joined_evidence = "\n\n".join(evidence)
prompt = f"""Answer the task using only the evidence below.

Task: {task}

{joined_evidence}
"""
```

Here, three excerpts can contribute up to 360 tokens. The final prompt also contains the task,
labels, separators, and instructions, and the model needs room to answer. Plan those parts when
choosing the per-source limit. Trimwise guarantees each excerpt's ceiling; your application decides
how the excerpt ceilings fit inside the complete prompt or context window.

Keep instructions outside the source text passed to Trimwise. The library is designed to reduce
evidence, not to shorten system prompts, tool rules, output schemas, or other instructions that the
model must follow exactly.

## 5. Choose the budget unit

The `limit` is always an upper bound. Query-aware selection may return a shorter excerpt when the
remaining candidates are weak, rather than adding unrelated text just to fill the budget.

| Unit | How Trimwise measures it | Good for |
| --- | --- | --- |
| `tokens` | Tiktoken with `o200k_base` by default | Planning LLM context usage |
| `words` | The number of whitespace-separated parts from `text.split()` | Human-readable editorial limits |
| `characters` | Python code points from `len(text)` | Exact application or storage limits |

Tokens are the default:

```python
from trimwise import Trimmer

result = Trimmer().trim(source_text, limit=500)
```

Use another unit explicitly:

```python
from trimwise import Trimmer

by_words = Trimmer().trim(source_text, limit=300, unit="words")
by_characters = Trimmer().trim(source_text, limit=2_000, unit="characters")
```

If your target model uses a different tokenizer, supply a synchronous token counter. Trimwise uses
it whenever it measures a proposed result, including the final check:

```python
import tiktoken

from trimwise import Trimmer

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count text with the tokenizer used by the target model."""
    return len(encoding.encode(text, disallowed_special=()))


result = Trimmer().trim(
    source_text,
    limit=500,
    token_counter=count_tokens,
)
```

A custom counter is valid only with a token budget and must return a nonnegative integer. Tiktoken
may need network access the first time it loads an encoding into its local cache.

## 6. Read the result

`trim()` and `atrim()` return an immutable `TrimResult`:

| Field | Meaning |
| --- | --- |
| `text` | The prompt-ready excerpt |
| `input_count` | Measured size of the original source |
| `output_count` | Measured size of the returned excerpt |
| `limit` | Maximum output size requested by the caller |
| `unit` | The unit used for the three counts |
| `strategy` | Concrete strategy used after resolving `auto` |
| `trimmed` | Whether the returned text differs from the input |

```python
from trimwise import Trimmer

result = Trimmer().trim("First fact. Middle detail. Final decision.", 5, unit="words")

print(result.text)
print(f"{result.output_count}/{result.limit} {result.unit.value}")
print(f"strategy={result.strategy.value}, trimmed={result.trimmed}")
```

Use `result.text` in the prompt. The other fields are useful for logging, budgeting, and checking
which automatic strategy was selected.

## 7. Use Trimwise without blocking an async application

`atrim()` has the same arguments and result as `trim()`. It moves parsing, token counting, ranking,
selection, synchronous callbacks, and Trimwise-managed model work away from the event loop.

```python
import asyncio

from trimwise import Trimmer


async def main() -> None:
    """Create one excerpt without blocking the event loop."""
    result = await Trimmer().atrim(
        "First fact. Middle explanation. Final decision.",
        limit=6,
        unit="words",
    )
    print(result.text)


asyncio.run(main())
```

Use `trim()` in ordinary synchronous programs and `atrim()` in web servers, async agents, and
concurrent pipelines. Cancelling an `atrim()` call stops waiting for its result, but Python cannot
terminate a worker thread that has already started.

## 8. Try semantic matching when wording differs

Semantic selection is useful when the query and source express the same idea with different words.
After installing the CPU semantic extra, request it explicitly:

```python
from trimwise import Trimmer

result = Trimmer().trim(
    source_text,
    limit=500,
    strategy="semantic",
    query="Why did customers lose access?",
)
```

Without an embedding callback, the first semantic or hybrid call that actually needs trimming
loads the configured FastEmbed model and may download its weights. The default multilingual model
is approximately 220 MB; later calls on the same `Trimmer` reuse it. Backend failures raise
`SemanticBackendError` rather than silently falling back to lexical matching.

If your application already has an embedding model, API, or cache, keep the core installation and
provide an embedding callback instead. Trimwise then uses your vectors and does not import or
download FastEmbed.

## Why these defaults?

The beginner path is lightweight, but it is not a positional heuristic:

- Queryless structural selection uses TF-IDF document-centroid salience, section coverage, and
  [Maximal Marginal Relevance](https://aclanthology.org/X98-1025/) to balance representative content
  against repeated-looking fragments.
- Query-focused automatic selection uses
  [BM25](https://doi.org/10.1561/1500000019), then keeps candidates around the strongest relevance
  drop with a small recall buffer before diversity-aware selection. This adapts the central idea of
  [Adaptive-k context selection](https://aclanthology.org/2025.emnlp-main.1017/) to Trimwise's
  source-backed candidates.
- Semantic matching is opt-in because embeddings add model or service cost. When requested,
  sentence-vector cosine similarity follows the general approach established by
  [Sentence-BERT](https://aclanthology.org/D19-1410/).

Trimwise adapts these research ideas to exact caller-defined budgets and verbatim source fragments;
it does not reproduce any paper's complete system. The scores estimate usefulness and similarity,
not factual truth.

## Common first-use mistakes

- **Expecting `auto` to use embeddings:** it intentionally stays structural or lexical. Select
  `semantic` or `hybrid` explicitly when meaning-based matching is required.
- **Omitting the query:** lexical, semantic, and hybrid strategies require a nonblank query.
- **Treating the limit as a target:** it is a ceiling. A query-aware result may stop early instead
  of adding weak evidence.
- **Budgeting only the excerpts:** leave space for labels, separators, instructions, examples, and
  the model's answer.
- **Passing instructions as evidence:** trim source material, then assemble it around instructions
  that remain unchanged.
- **Expecting a summary:** Trimwise selects and joins original fragments; it does not paraphrase or
  synthesize new sentences.

## Continue exploring

- Return to the [Trimwise overview](index.md).
- Review the current [strategy guide](https://github.com/tenwritehq/trimwise#which-strategy-should-i-use).
- Learn about [embedding callbacks and FastEmbed](https://github.com/tenwritehq/trimwise#semantic-models).
- See the [public package on PyPI](https://pypi.org/project/trimwise/).
