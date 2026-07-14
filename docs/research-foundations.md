# Research Foundations

Trimwise uses established ranking and extractive-summarization methods to answer a practical
question: **which parts of this source are most useful to keep when the whole source will not fit?**

The answer depends on what you know at trimming time. Without a query, Trimwise looks for content
that represents the document and covers its structure. With a query, it can favor exact matches,
semantic matches, or both. It then reduces obvious repetition and assembles the chosen fragments
inside the requested budget.

These methods are research-based, but Trimwise is its own engineering system. The cited papers
support the scoring and selection ideas; they do not guarantee that every Trimwise excerpt is the
best possible excerpt for every document or downstream model.

## The methods at a glance

| Method | Why it helps | How Trimwise uses it |
| --- | --- | --- |
| [BM25](https://doi.org/10.1561/1500000019) | Finds literal evidence such as names, identifiers, error codes, and exact phrases | Scores each source fragment against your query using Tiktoken subword IDs |
| [Centroid-based extraction](https://aclanthology.org/W00-0403/) | Finds passages that represent the document when no query is available | Compares each fragment with the document's TF-IDF lexical center |
| [Sentence-BERT](https://aclanthology.org/D19-1410/) | Matches meaning even when the query and source use different wording | Compares query and passage embeddings with cosine similarity |
| [Multilingual sentence embeddings](https://aclanthology.org/2020.emnlp-main.365/) | Makes cross-language and multilingual similarity practical | Provides the research basis for using a multilingual default embedding model |
| [Convex score fusion](https://arxiv.org/abs/2210.11934) | Combines exact-match and meaning-based evidence without throwing away score strength | Blends normalized BM25 and semantic scores equally in `hybrid` mode |
| [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114) | Combines ranked lists when their raw score scales are not useful | Provides a fallback when a hybrid score list cannot be meaningfully normalized |
| [Maximal Marginal Relevance](https://aclanthology.org/X98-1025/) | Balances relevance with reduced repetition | Penalizes candidates that look too similar to material already selected |
| [Adaptive-k](https://aclanthology.org/2025.emnlp-main.1017/) | Avoids assuming that every query needs the same amount of evidence | Finds a natural score boundary before diversity selection instead of always filling the limit |

Trimwise applies all of these methods **inside the single string you provide**. It does not search
the web, retrieve documents, query a vector database, or maintain an index.

## BM25 for exact query evidence

BM25 is a widely used lexical relevance method. It rewards query terms found in a candidate,
limits the benefit of repeating the same term many times, and adjusts for candidate length. That
makes it especially useful when exact wording matters:

- Incident IDs and error codes
- Product names and version numbers
- URLs, symbols, and API names
- Quoted phrases or terminology from the source

Trimwise uses BM25 for the `lexical` strategy with the conventional values `k1=1.5` and `b=0.75`.
Instead of relying on English words separated by spaces, it treats Tiktoken subword IDs as terms.
This keeps lexical scoring useful for identifiers and for writing systems that do not consistently
separate words with spaces.

For ranking only, a short fragment can also see its nearest heading and neighboring fragments in
the same section. This helps a sentence such as “It failed after 43 seconds” inherit the local
context that explains what “it” means. The extra context improves the score but is not silently
inserted into the result; Trimwise still returns exact source fragments.

BM25 matches shared text, not general meaning. If your query says “service interruption” while the
source only says “the API stopped responding,” use `semantic` or `hybrid` instead.

## Centroid salience when there is no query

Sometimes there is no question to rank against. You may be preparing representative excerpts from
hundreds of articles before clustering them, sampling reports for an LLM, or reducing notes before
you know what will be asked later.

Centroid-based extraction provides a queryless signal. It represents each candidate with TF-IDF,
averages those vectors into a document centroid, and scores candidates by their similarity to that
center. A high score means that a fragment reflects vocabulary and topics that occur across the
document.

Trimwise uses this signal in `structural` mode, but it does not simply keep the highest-scoring
sentences. It also:

- Preserves fitting opening and ending units for document-wide context.
- Gives each Markdown section an initial share of the available space.
- Redistributes unused space toward stronger remaining fragments.
- Expands one oversized plain-text paragraph into sentences or source lines before ranking when
  complete boundaries are available.

This is why structural trimming is not a fixed 50/25/25 split. The amount retained from each part
depends on the document's units, sections, salience, repetition, and exact budget.

Centroid similarity measures representativeness, not human importance. A rare warning or final
decision may be important precisely because it differs from the rest of the document. Structural
coverage and boundary protection help with this, but no queryless method can infer every reader's
future intent.

## Sentence embeddings for meaning-based relevance

Sentence-BERT showed that independently produced sentence vectors can make semantic comparison
fast enough for practical use. Texts with similar meanings are placed near one another in the
embedding space and compared with cosine similarity.

Trimwise uses this pattern in `semantic` mode:

1. Produce one vector for the query.
2. Produce one same-dimension vector for each candidate passage.
3. Rank candidates by their cosine similarity to the query.

You can supply those vectors through a synchronous or asynchronous embedding callback with the
core Trimwise installation. Alternatively, the optional FastEmbed integration manages the model
for you. The default is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a 384-dimensional multilingual
model listed by FastEmbed at approximately 0.22 GB. See the
[FastEmbed supported-model list](https://qdrant.github.io/fastembed/examples/Supported_Models/).

The model is loaded only when semantic scoring is actually required. Structural and lexical calls
do not load it, and input that already fits returns before embedding work begins.

Multilingual embedding research supports the idea of mapping related sentences from different
languages into a shared vector space. Actual results still depend on the model, language, domain,
and wording. Trimwise supplies the comparison pipeline; it does not make every embedding model
equally accurate.

## Hybrid scoring for exact terms and paraphrases

Lexical and semantic scoring solve different failure cases. BM25 is strong when literal evidence
matters, while embeddings can connect different wording with similar meaning. `hybrid` uses both.

Trimwise independently normalizes the BM25 and semantic score lists, then combines them with an
equal convex blend:

```text
hybrid score = 0.5 x normalized BM25 + 0.5 x normalized semantic score
```

This preserves useful score differences. A passage that is overwhelmingly stronger than its
neighbors can contribute more than one that wins by only a tiny margin. Research comparing hybrid
fusion methods found normalized convex combinations effective and relatively sample-efficient to
tune, although those experiments used retrieval datasets rather than Trimwise excerpts. See
[An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934).

If a lexical or semantic score list is completely flat or otherwise cannot be safely normalized,
Trimwise falls back to Reciprocal Rank Fusion with `k=60`:

```text
RRF score = 1 / (60 + BM25 rank) + 1 / (60 + semantic rank)
```

RRF combines ordering rather than raw score magnitude, so it remains useful when score scales do
not carry enough information. In normal hybrid calls, the 50/50 normalized blend is used; RRF is
the fallback, not the primary formula.

## MMR for less repetitive excerpts

Selecting only the highest relevance scores can keep several passages that say nearly the same
thing. Maximal Marginal Relevance, or MMR, addresses that by reconsidering each remaining candidate
after every selection.

Trimwise scores the next candidate using:

```text
MMR = lambda x relevance - (1 - lambda) x greatest similarity to selected text
```

The default `mmr_lambda=0.7` gives relevance 70% of the decision and the similarity penalty 30%.
Structural and lexical strategies compare TF-IDF vectors; semantic and hybrid strategies compare
embedding vectors. The final fragments are still restored to their original source order.

MMR improves variety in the selected representation, but it does not understand atomic facts. Two
different phrasings of the same fact may look different, while two different facts about the same
subject may look similar. Therefore, Trimwise promises reduced representational repetition—not
guaranteed factual diversity.

MMR is also a greedy selector rather than a global token knapsack solver. A large useful passage
can occasionally consume space that several smaller complementary passages might have used better.
The exact budget is guaranteed; the globally optimal information mix is not.

## Adaptive evidence selection

A token limit is a ceiling, not a request to add increasingly weak material until every token is
used. Query-aware strategies can therefore stop below the limit when the score distribution shows
a clear boundary between strong and weak evidence.

Trimwise adapts the largest-score-gap rule from Adaptive-k:

1. Sort candidate evidence by its primary query score.
2. Look for the largest adjacent score drop within the first 90% of possible boundaries.
3. Keep candidates through that boundary.
4. Retain five additional candidates as a recall buffer.
5. Run MMR and exact-budget composition within that evidence pool.

The primary score is BM25 for `lexical`, embedding similarity for `semantic`, and the fused score
for `hybrid`. A selected passage can bring its nearest section heading when both fit.

Adaptive-k was studied for selecting retrieved documents in long-context question answering.
Trimwise applies the same boundary idea to fragments inside one supplied source. This helps avoid
irrelevant padding, but a sharp score drop is still a heuristic: a necessary detail can sometimes
receive a weak score.

## What Trimwise adds around the research

The published methods provide relevance, fusion, and diversity signals. They do not by themselves
produce a safe, source-faithful excerpt. Trimwise adds the surrounding behavior required for LLM
prompt assembly.

### Source-aware segmentation

Trimwise recognizes CommonMark headings, paragraphs, lists, blockquotes, tables, HTML blocks,
fenced code, front matter, reference definitions, and otherwise uncovered source ranges. It uses
Markdown line maps to slice the original input instead of rendering Markdown back into new text.

This matters when prompts contain evidence that must remain quotable: numbers, identifiers,
formatting, code fences, qualifications, and attribution stay tied to the supplied source.

### Lightweight signal preservation

After normalizing the main score, Trimwise gives a small additional weight to four details that
often carry useful technical or operational evidence:

- A nearby heading
- A URL
- Number- or date-like content
- A code-style identifier

These four checks contribute 10% of final relevance; the primary strategy contributes 90%. They
are lightweight signals, not entity extraction or factual verification.

### Exact-budget composition

Every candidate is tested as part of the fully composed excerpt. Trimwise accounts for source
order, separators, attached headings, and affordable omission markers, then measures the complete
result again before returning it.

That provides the library's hard operational guarantee:

```text
output_count <= limit
```

When no complete candidate fits, Trimwise progressively tries smaller complete units and finally
an exact fitting source prefix. This guarantees a usable result under tiny limits, although the
last fallback cannot preserve document-wide meaning when the source offers no smaller boundaries.

## Why Trimwise remains extractive

Language-model compressors and abstractive summarizers can sometimes express more information in
fewer tokens, especially at extreme compression ratios. They can also rewrite a number, remove a
qualifier, change attribution, or combine distant claims into wording that never appeared in the
source.

Trimwise makes the opposite tradeoff. It selects and arranges original fragments without
paraphrasing them. This is useful when you need prompt evidence that is predictable, auditable,
and safe to quote, or when running another model solely for compression would add unwanted latency,
cost, or privacy exposure.

The tradeoff is real: Trimwise cannot synthesize two distant facts into one shorter sentence, and
at very aggressive limits a trained compressor may preserve more answer-relevant information per
token. Choose based on whether source fidelity or maximum compression density matters more for your
application.

## What “research-based” means for Trimwise

The research supports the building blocks:

- BM25 for exact lexical relevance
- TF-IDF centroid similarity for queryless representativeness
- Sentence embeddings for semantic similarity
- Normalized convex fusion for combining lexical and semantic evidence
- RRF when score normalization is not informative
- MMR for balancing relevance and representational repetition
- Adaptive-k for varying the amount of retained evidence by query

It does not prove that Trimwise always finds every important fact, produces factually independent
fragments, beats every other compressor, or improves every downstream LLM task. Those outcomes also
depend on the source, query, budget, embedding model, and task.

## Choose a strategy

| Your situation | Start with | Why |
| --- | --- | --- |
| You have no query and want broad coverage | `structural` | Uses document centrality, section coverage, and boundary anchors without loading a model |
| Exact names, IDs, errors, numbers, or phrases matter most | `lexical` | BM25 directly rewards matching query subwords |
| The answer may be paraphrased or expressed in another supported language | `semantic` | Embeddings compare meaning beyond exact vocabulary |
| Both literal evidence and semantic meaning matter | `hybrid` | Combines normalized BM25 and embedding scores before repetition reduction |
| You want a safe default | `auto` | Uses `structural` without a query and `lexical` with one, so it never loads embeddings automatically |

For usage guidance, continue to [Strategies](strategies.md). For the complete processing pipeline,
see [How Trimwise Works](how-it-works.md). For model callbacks, FastEmbed, and async behavior, see
[Semantic Models and Async Use](semantic-and-async.md). Hard guarantees and important boundaries
are collected in [Guarantees and Limitations](guarantees-and-limitations.md).
