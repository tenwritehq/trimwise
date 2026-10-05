---
title: Trimwise Context Workbench
emoji: "\u2702"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
license: mit
short_description: Try query-aware text trimming with one shared context budget.
tags:
  - text
  - context
  - lexical
---

# Trimwise Context Workbench

A CPU-only demo of [Trimwise](https://github.com/tenwritehq/trimwise), running the library
locally in the Space process. No hosted Trimwise API, LLM, GPU, or API key is used.
Lexical is the default and requires no embedding-model download. Semantic and Hybrid are
optional selections that load a CPU embedding model only when vectors are needed.
The repository folder `demos/huggingface-space/` is the source of truth.
The core package and its dependency configuration are unchanged.

The **Single source** tab awaits public `Trimmer.atrim()`. **Multiple sources** awaits
`Trimmer.atrim_context()` with three input-aligned `ContextSource` objects and one shared
limit. Evidence from all sources competes for that allowance. The synthetic presets place
relevant details late in a document or in separate sources. They illustrate behavior;
they are not a quality evaluation.

Both tabs default to **Lexical**. With this selection, an empty or whitespace-only query
uses **Structural**, because lexical mode requires a query. The selector also offers
Structural, Semantic, Hybrid, and Auto. Auto resolves to lexical with a query or structural
without one. Semantic and Hybrid require a nonblank query and never silently fall back to
another method if the backend fails. The result strip names the strategy actually used.
All inputs remain editable. Reset clears the active tab's inputs and results; Load example
restores its synthetic preset. Both restore Lexical and cancel pending results for that tab.

Custom CSS and native Gradio theme tokens reuse the public Trimwise site's bracket mark,
teal accent, and restrained editor layout. System fonts avoid external font requests.
The visible **Free & private API** link opens [API setup and privacy details](https://trimwise.aatbit.com)
for developers who prefer HTTP. It is only a link; this demo's trim handlers always run
the installed Python library. The API's separate privacy statement says it processes text
in memory without saving prompts, queries, or results. That statement is not a privacy
guarantee about Hugging Face's hosting infrastructure.

## Local run from this repository

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv demos/huggingface-space/.venv
source demos/huggingface-space/.venv/bin/activate
python -m pip install -r demos/huggingface-space/requirements.txt
python -m pip install --no-deps -e .
python demos/huggingface-space/app.py
```

Open `http://127.0.0.1:7860`. If that port is busy, launch with
`GRADIO_SERVER_PORT=7861 python demos/huggingface-space/app.py` and open that port instead.
The editable install ensures this run uses the checkout's library. The demo requirements
pin the published `trimwise==0.7.0`, compatible with this checkout, for a standalone Space.
UI packages are confined to this virtual environment; they are not core dependencies.

On Windows, use `py -3.11 -m venv demos/huggingface-space/.venv` and activate with
`demos\huggingface-space\.venv\Scripts\activate` before the `python -m pip` commands.

## Copy into a Hugging Face Space

Create a separate **Gradio** Space with **CPU Basic** hardware. In a local clone of that
Space repository, copy only these five files into its root:

```bash
cp /path/to/trimwise/demos/huggingface-space/{app.py,styles.css,requirements.txt,README.md,test_app.py} /path/to/space/
```

For a standalone smoke run in the Space clone:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s . -p test_app.py -v
python app.py
```

Commit only those five demo files to the Space repository and push them using your own
Hugging Face account. Do not copy the Trimwise repository, `.venv`, caches, benchmark
data, credentials, or private files. Spaces reads the YAML metadata above and installs
`requirements.txt`. On Spaces, `SPACE_ID` enables the required `0.0.0.0` binding; local
runs bind to loopback. There is no automatic deployment workflow or credential setup.

Metadata fields follow the [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).

## Counts, timing, and provenance

The counter is Trimwise's existing `Measurer`, configured from `TrimConfig` with tiktoken
**`o200k_base`**, the same built-in encoding and special-token handling as the trim calls.
The demo also reuses its Unicode-preserving `fitting_prefix()` for the illustrative baseline;
it does not implement another compression algorithm. `trimwise.measurement` is an internal
module, so this small dependency is deliberately tied to the pinned release and covered
by tests. tiktoken may download its tokenizer vocabulary on first startup, then cache it;
this is not an embedding-model download. Model-free methods need no network once the
tokenizer vocabulary is cached; Semantic and Hybrid may also download model files.

- **Original context tokens:** measure the complete untrimmed assembly. For multiple
  sources this includes `[Source i]\n` labels and `\n\n` separators between nonempty sources.
- **Retained context tokens:** use the public result's `output_count`. For multiple
  sources, `result.text` includes contributing source labels and separators within the limit.
- **Original evidence tokens:** the public `input_count`, which measures source text only;
  multi-source counts are summed independently. It need not equal assembled input tokens.
- **Reduction:** `100 * (1 - retained_context_tokens / original_context_tokens)`.
  Empty input is defined as 0%. This is size reduction, not quality or accuracy.
- **Trim API wall time:** `perf_counter()` around the awaited public trim call, including
  its worker scheduling. It excludes validation, baseline, rendering, and queue wait.
  It is a measurement of this request, not a portable performance claim.
- **Per-source counts:** evidence and any library omission markers only. They exclude
  source labels/separators and may not sum to the final context count because tokenization
  at concatenation boundaries is not additive.

Source indices are zero-based input positions, including empty or noncontributing rows.
Sources and their excerpts stay in input order. Spans are half-open `[start, end)` Python
string offsets into each original source, measured in Unicode code points rather than
UTF-8 bytes or JavaScript UTF-16 units. Each span view displays the exact original slice.
Labels, separators, and generated omission markers have no source spans.

The prefix comparison receives the **same original assembly, tokenizer, and token ceiling**
as Trimwise. It is a literal prefix, so it may cut a label or passage in the middle. It
does not impose Trimwise's complete-wrapper policy and it is not a claim of universal
quality improvement. Neither result is graded for downstream answer accuracy.

The limit covers evidence/context, **not the whole eventual LLM prompt**. Reserve and
measure instructions, the question, tool definitions, other messages, and answer space
separately using the eventual model's tokenizer.

## Hosting limits and privacy

Semantic and Hybrid disclose the configured model:
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
through **FastEmbed 0.8.1 / ONNX Runtime on CPU**. Trimwise itself handles lazy downloading,
model reuse, vector validation, ranking, and serialized inference. Switching the selector
alone does not download weights; a fitting input or zero budget also bypasses inference.
The first trim that needs embeddings can be slow. Model files are cached on disk and can
be downloaded again after a Space rebuild or cache eviction; user text is not part of that
cache. No model revision is frozen by this demo, so upstream weight updates can affect
results. CPU-only inference uses two threads and passage batches of 16. Both tabs share
one reusable Trimmer for its managed model cache and lock, with separate request-local
inputs, selections, vectors, and results. The query and evidence are processed locally by
the model, not sent to an inference service. Backend failures show a bounded message and
allow the user to choose Lexical explicitly.

The server validates at most 12,000 characters per source, 18,000 source characters in
total, 4,096 evidence tokens, and 300 source lines. The query cap is 800 characters and
the integer budget range is 0 through 2,048 tokens. UI character caps also apply, but
server validation remains necessary for API requests. A rejected request is not silently
shortened. Both tabs share **two concurrent trimming slots** and a queue of **eight**;
Gradio's direct queue bypass is disabled. These are small-demo limits, not a document
processing service or an infrastructure-level denial-of-service defense.

Please do not submit sensitive data. There are no file outputs, input/result logs,
flagging, cached examples, application analytics, or configured telemetry exporters.
Gradio analytics and run history are disabled. Text/results exist transiently in process,
queue, transport, and browser memory, with no application-level persistence; Reset does
not promise secure erasure. User text is escaped in HTML views and displayed as plain
text in output boxes. The app does not control Hugging Face's infrastructure logging or
retention. Other users' requests have separate result objects; configuration, tokenizer,
and Trimwise's managed model are reused without caching user inputs or vectors.

Very small budgets can return nothing, especially when no complete source label and
evidence fit. Repeated passages retain their source identities; lexical mode does not
globally deduplicate them. Evidence may be omitted, conflicting sources are not resolved,
and lexical matching can miss relevant paraphrases. Cancelling an async handler cannot
forcibly stop CPU work that the library has already offloaded to a thread. Hosting limits
bound that work. Semantic matching is heuristic too; it does not verify truth or source
authority. Model-dependent output is not evidence of downstream answer quality.

## Tests

With the demo environment active, from the Trimwise repository root:

```bash
python -m unittest discover -s demos/huggingface-space -p test_app.py -v
```

This independently runnable suite uses the standard library, checks both presets, public
API parity, shared budgets with labels, empty/tiny/Unicode/repeated inputs, span provenance,
statistics, baseline fairness, input limits, HTML escaping, request isolation, model-free
paths, selected-method dispatch, CPU model disclosure, and async/queue configuration.
Semantic/Hybrid unit checks use a deterministic callback, so this suite never downloads
weights. The same test file works in the separate Space clone.

Runtime pins: Python 3.11, Gradio 6.29.1, Trimwise 0.7.0, tiktoken 0.14.0, FastEmbed 0.8.1. The top-level
dependencies are pinned; this is not a full transitive lockfile. Run the tests after any
dependency update and before copying updates to your Space.
