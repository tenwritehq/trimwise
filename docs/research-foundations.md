---
title: Research Behind Trimwise Text Selection
description: Explore the research ideas behind Trimwise, including BM25, embeddings, MMR, adaptive evidence selection, and extractive prompt compression.
---

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

## How Trimwise compares with model-based compression

There is no universally best context compressor. The useful choice depends on four questions:

1. Must the output remain readable to people?
2. Must every retained fragment be traceable to exact source text?
3. Can you run another language model or trained encoder during compression?
4. Is maximum task performance at an aggressive compression ratio more important than preserving
   complete source units?

| Method | Selection granularity | Compression model | Output relationship to the source | Main strength |
| --- | --- | --- | --- | --- |
| Prefix slicing | One continuous prefix | None | Exact prefix, but ignores everything later | Minimal overhead |
| Trimwise | Markdown blocks, paragraphs, sentences, or lines | None for structural and lexical use; optional embeddings for semantic ranking | Exact retained fragments in source order, separated by affordable omission markers | Readable evidence with an exact measured ceiling |
| [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/) | Tokens, selected through a coarse-to-fine process | Small causal language model | Retained tokens come from the prompt, but complete sentences and blocks need not survive | Very high compression on the paper's evaluated tasks |
| [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) | Query-aware document and token selection for long prompts | Small causal language model | Can reorder and compress prompt content around the question | Long-context relevance, position handling, and target-model efficiency |
| [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) | Token classification | Trained bidirectional encoder | Extractive at token level, without a complete-unit guarantee | Faster, task-agnostic learned compression |
| [Selective Context](https://arxiv.org/abs/2310.06201) | Tokens, phrases, or sentences | Causal language model for self-information | Retains higher-information lexical units; readability depends on the chosen granularity | Removes language that the scoring model considers predictable |
| [RECOMP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html) | Selected sentences or generated multi-document summaries | Trained extractive or abstractive compressor | Extractive mode keeps sentences; abstractive mode synthesizes new wording | Task-trained compression of retrieved evidence |

### The LLMLingua family

The original LLMLingua uses a budget controller, coarse-to-fine scoring, and iterative token-level
compression. Its paper reports up to 20x compression with little performance loss across its four
evaluated datasets. Because it can remove tokens inside sentences, it can pack more surviving
information into a tight prompt than a complete-fragment selector. The result may also be difficult
for a person to read, even when it remains useful to the target LLM.

LongLLMLingua adapts that approach to question-aware long-context prompts. It considers relevance
and the position of key information, addressing cases where useful evidence is buried in a long
context. The paper reports up to a 21.4% improvement on NaturalQuestions with about four times fewer
tokens in GPT-3.5-Turbo, plus 1.4x-2.6x end-to-end acceleration for roughly 10,000-token prompts
compressed by 2x-6x. These are paper results on specific models and benchmarks, not expected results
for every prompt.

LLMLingua-2 replaces causal-language-model surprisal with a learned token-classification objective.
It distills training data from a larger model and uses a bidirectional encoder to decide which
tokens survive. The paper reports 3x-6x faster compression than the compared prompt-compression
methods and 1.6x-2.9x end-to-end latency acceleration at 2x-5x compression ratios.

Compared with this family, Trimwise uses coarser units and cannot usually reach the same density at
extreme compression ratios. Its benefit is that selected passages remain complete, readable source
fragments; structural and lexical use require no compression model; and the final composed result
is measured against the caller's exact budget. Trimwise has not been benchmarked head-to-head
against the LLMLingua family, so it does not claim better downstream answer quality.

### Selective Context

Selective Context scores lexical units by self-information from a causal language model and removes
the more predictable content. It can operate on tokens, phrases, or sentences; the paper found
phrase-level selection strongest in its experiments. At 50% context reduction, the paper reports
36% lower inference memory use and 32% lower inference time, with small average drops on its
reported quality measures.

This method is attractive when natural-language redundancy is the primary target. Trimwise instead
uses document structure, query relevance, and representation-level diversity. It does not estimate
language-model surprisal. In return, its core paths avoid a language model and preserve complete
source units rather than relying on token- or phrase-level readability.

### RECOMP

RECOMP is designed for evidence produced by a retrieval system. It trains two alternatives: an
extractive compressor that selects useful sentences and an abstractive compressor that synthesizes
information across retrieved documents. Either compressor can return an empty string when the
retrieved material does not help. The paper reports compression to as little as 6% of the retrieved
text with minimal performance loss on its language-modeling and open-domain question-answering
evaluations.

RECOMP's extractive mode is the closest comparison to Trimwise because both can select complete
sentences. The difference is purpose and training: RECOMP learns to optimize a downstream task over
retrieved documents, while Trimwise applies deterministic general-purpose ranking to the string you
already have. RECOMP's abstractive mode can combine distant facts much more densely, but its output
is generated text rather than an exact set of source fragments.

### Which tradeoff should you choose?

Choose Trimwise when:

- Evidence should remain readable, auditable, and easy to trace to its source.
- Markdown sections, lists, tables, code fences, identifiers, and qualifications should stay intact.
- You need an exact token, word, or character ceiling on the finished excerpt.
- You want structural or query-aware lexical selection without another model.
- You want to trim each source separately before assembling instructions and evidence into a prompt.

Choose token- or model-based compression when:

- Extreme compression matters more than complete sentences and blocks.
- You can run and deploy the required compression model.
- You can evaluate answer quality, faithfulness, and latency on your own tasks.
- You want a learned compressor optimized for a particular target model or retrieval workflow.

The approaches can be chained. Trimwise can first select readable evidence from across a very long
source, after which LLMLingua-style compression can prune the smaller result. This may improve total
compression, but the final output no longer has Trimwise's complete-fragment or source-layout
guarantees.

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
