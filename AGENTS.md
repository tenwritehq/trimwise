# Repository instructions

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them—don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't improve adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it—don't delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria require clarification.

## 5. Leave Behind No Debt

After every change, fix, feature, or implementation, review the work and ask:

- What bugs did it introduce?
- What bugs exist in the changes?
- What edge cases and errors were missed?
- Was it overcomplicated or overengineered?
- Did it stay DRY and reuse or extend instead of rebuilding?
- Are all of the above checked and covered?

Fix bugs, missed edge cases, errors, and rule violations before hand-off.

## 6. Mandatory Change Workflow

Before every repository change, no matter how small, read and use these skills:

1. Consequences: `C:\Users\Aakash\.agents\skills\consequences\SKILL.md`
2. Ponytail: `C:\Users\Aakash\.codex\plugins\cache\ponytail\ponytail\4.8.4\skills\ponytail\SKILL.md`
3. Clean Features: `C:\Users\Aakash\.codex\skills\clean-features\SKILL.md`

This applies to source, tests, documentation, configuration, and workflow files.

## 7. Python Documentation

- Every Python module, class, method, function, async function, and nested function must have a
  meaningful Google-style docstring, including tests and test helpers.
- Include `Args`, `Returns`, `Yields`, `Raises`, and `Attributes` only when applicable.
- Docstrings explain purpose and contract rather than repeating a symbol's name.
- Inline comments explain why. Do not add redundant comments, metadata, or commented-out code.

## 8. Repository Structure

Keep each responsibility in its existing file. Generated folders such as `dist/`, `build/`,
`*.egg-info/`, `.consequences/`, and tool caches are not source files and are not part of this map.

### Project files

- `.gitignore`: excludes generated packages, caches, virtual environments, and local task data.
- `AGENTS.md`: defines the repository-wide rules that agents must follow.
- `LICENSE`: contains the MIT license for Trimwise.
- `README.md`: explains installation, public usage, strategies, configuration, and releases.
- `ROADMAP.md`: records deliberately deferred ideas that need evidence before implementation.
- `plan.md`: preserves the detailed Trimwise v1 implementation and acceptance plan.
- `pyproject.toml`: defines Flit packaging, dependencies, Python support, and tool configuration.

### GitHub automation

- `.github/workflows/ci.yml`: runs the supported-Python tests, quality checks, semantic integration
  test, and package build smoke test.
- `.github/workflows/release.yml`: validates a version tag, builds distributions, and publishes
  them to PyPI through trusted publishing.

### Package source

- `src/trimwise/__init__.py`: exposes the six supported public API names and nothing else.
- `src/trimwise/measurement.py`: measures token, word, and character budgets and finds fitting
  prefixes.
- `src/trimwise/models.py`: defines public enums, configuration, result values, and semantic
  backend errors.
- `src/trimwise/ranking.py`: builds scoring-only section and neighbor context, then implements
  structural, BM25, semantic, hybrid, signal, cosine, and MMR ranking calculations.
- `src/trimwise/segmentation.py`: splits Markdown into exact source-backed blocks and assigns
  section context.
- `src/trimwise/semantic.py`: invokes caller-provided sync or async embedding callbacks, validates
  and normalizes all semantic vectors, lazily loads FastEmbed, serializes managed model use, and
  converts backend failures into stable package errors.
- `src/trimwise/trimmer.py`: validates public calls, expands a sole oversized paragraph into
  complete structural candidates, and orchestrates measurement, callback or FastEmbed ranking,
  selection, omission markers, fallback splitting, and async execution boundaries.
- `src/trimwise/py.typed`: tells type checkers that the installed package includes inline types.

### Tests

- `tests/test_api.py`: checks the public API, configuration, validation, budgets, fast paths,
  omission behavior, and source ordering.
- `tests/test_async_semantic.py`: checks FastEmbed and caller callback precedence, vector validation,
  staged failures, model reuse, concurrency, async equivalence, and cancellation behavior.
- `tests/test_docstrings.py`: enforces Python docstrings and verifies that `py.typed` is packaged.
- `tests/test_ranking.py`: checks BM25, centrality, semantic and hybrid fusion, signal scoring,
  similarity, normalization, and MMR diversity.
- `tests/test_segmentation.py`: checks Markdown source spans, headings, raw gaps, fences, tiny
  budget fallbacks, and oversized plain-text sentence, line, and CJK coverage.
- `tests/integration/test_fastembed.py`: exercises real multilingual retrieval with the default
  FastEmbed model; it is kept separate because it downloads or loads a large optional model.

## 9. Structural and Performance Invariants

- Automatic mode without a nonblank query resolves to structural mode.
- The core installation supports semantic and hybrid ranking through a caller-provided embedding
  callback. FastEmbed extras are required only when Trimwise must provide the embedding model.
- Configure at most one of `embedding_callback` and `async_embedding_callback`. A callback returns
  one query vector and exactly one same-dimension, finite, nonempty vector per supplied passage;
  validation and NumPy `float32` normalization remain inside `semantic.py`.
- A caller-provided callback overrides FastEmbed for explicit semantic and hybrid ranking, even
  when a FastEmbed extra is installed. Do not import, initialize, or download FastEmbed on that
  path. Automatic mode stays structural or lexical, and fitting inputs never invoke a backend.
- Synchronous embedding callbacks run inside `atrim()`'s worker thread. Asynchronous callbacks run
  on the calling event loop while parsing, normalization, ranking, and selection remain offloaded;
  `trim()` rejects an async callback only when semantic vectors are actually required.
- Caller callbacks may run concurrently and own model or client reuse, caching, rate limiting, and
  thread or task safety. Only Trimwise-managed FastEmbed is cached and serialized per `Trimmer`.
- When an oversized structural input produces one paragraph, rank its complete sentence and
  source-line slices before selection. Preserve exact source spans and recognize CJK sentence
  punctuation that is not followed by whitespace.
- Structural selection protects fitting first and last units, then uses centrality and MMR. Do
  not replace this with a fixed positional ratio such as 50/25/25.
- Query-aware selection treats the limit as a ceiling. Bound non-heading evidence at the largest
  primary-score drop, ignore bottom-decile gap outliers, keep a five-unit recall buffer, and then
  run MMR inside that pool. Do not add a public cutoff setting without cross-model quality evidence.
- Context-rich ranking text may repeat source evidence, but selection and composition must use
  only the exact `Segment.text` source slices.
- A single unpunctuated source line has no smaller complete unit and therefore keeps the exact
  fitting-prefix fallback.
- Performance reports must distinguish character and token budgets, cold and warm calls, core
  and semantic strategies, candidate counts, and input/output limits. Never turn one machine's
  timings into a test threshold.
- The exact greedy MMR pass is deliberately `O(selected x candidates)`. Profile representative
  token workloads before changing that tradeoff or adding approximate-neighbor machinery.
