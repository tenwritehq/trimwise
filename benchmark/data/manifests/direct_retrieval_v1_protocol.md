# Direct retrieval baseline protocol

This is a new query-aware comparison over the frozen 160-case suite. It does not
modify the historical v1.1 or v1.2 compression JSONL files.

## Question

Do three ordinary fixed-window retrieval policies preserve normalized contiguous
required source spans as well as the two Trimwise query-aware configurations?

## Fixed inputs and measurement

- Dataset: `data/position_controlled_160.jsonl` (160 cases).
- Budgets: 128, 256, 512, and 1,024 `o200k_base` tokens.
- Primary score: normalized contiguous required-span containment, with prohibited
  content absence and budget compliance required for case pass.
- Sensitivities: local ordered retention at 80% and 90%.
- Output file: `results/position_controlled_160_direct_retrieval_results.jsonl`.
- No downstream QA calls are made for this comparison.

## Direct baselines

All three split each source into 128-token windows with 32-token overlap, choose
windows against the same query, reconstruct selected windows in source order, and
use the same omission marker and token ceiling.

1. **BM25 fixed-window retrieval** ranks windows by BM25.
2. **Embedding top-k fixed-window retrieval** ranks windows by cosine similarity
   to the query using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
3. **Embedding-MMR fixed-window retrieval** uses the same encoder and windows,
   with MMR relevance weight `0.7`.

The encoder is the published default used by Trimwise Hybrid. CUDA runs use the
existing thermal gate: refuse foreign CUDA compute, pause at 70 C, resume at 60 C,
and record thermal metadata. BM25 uses the same runner but no model backend.

## Analysis policy

The new rows are aggregated together with the immutable 6,400 historical
compression rows without duplicate result identities. The report includes all
three direct baselines and both Trimwise strategies at every budget, with
natural-only, controlled-relocation, self-source-excluded, position, task,
feasibility, and paired-bootstrap summaries.

## Released input and output identities

This is a post-review follow-up compression run, not a rescore of the frozen
rows. The released files identify its inputs and result snapshot:

| File | SHA-256 |
| --- | --- |
| `data/position_controlled_160.jsonl` | `9d9667127dc4461b000f78cd4b46e3e845c9d4030af09c353ee69a4bd81bfdab` |
| `results/position_controlled_160_results.jsonl` | `38248f76276c24db0ce2f9a8b3c74eeaaa7668e78862c0f2e1b4436c8a24d38d` |
| `configs/direct_retrieval_160.yaml` | `5ec6a214a6245a0eb9b359ef091b22861c0f411615c1910669032a560ff23fe2` |
| `results/position_controlled_160_direct_retrieval_results.jsonl` | `e0cef466478891832575cd8dd88dd1bed3d86c8b319c76b829f11a3d4b0769af` |

The direct result file has 1,920 successful rows: 160 cases × four budgets ×
three methods. It adds no QA rows and does not modify the historical source
outputs.
