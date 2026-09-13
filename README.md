# Trimwise

[![PyPI version](https://img.shields.io/pypi/v/trimwise.svg)](https://pypi.org/project/trimwise/)

Need to fit a long document into a small AI prompt budget? Trimwise picks excerpts from
across it that fit the space you set aside for source text, instead of just cutting off
the end. Add a question to steer the choice; the selected passages keep their original
wording and order.

[Documentation](https://trimwise.readthedocs.io/en/latest/) · [PyPI](https://pypi.org/project/trimwise/)

## Try it

Install Trimwise with Python 3.10–3.14:

```bash
python -m pip install trimwise
```

Here's the input and the question:

```python
from trimwise import Trimmer

report = """# Incident report

The site went down at 09:14. Initial checks focused on the network.

## Root cause

An expired credential blocked the service.

## Prevention

Credentials will now rotate every 30 days.
"""

result = Trimmer().trim(
    report,
    limit=24,
    query="What caused the outage and how will it be prevented?",
)
print(result.text)
print(f"{result.output_count}/{result.limit} tokens")
```

Output from running that code:

```text
## Root cause

An expired credential blocked the service.

## Prevention

Credentials will now rotate every 30 days.

23/24 tokens
```

The limit is a ceiling, not a target to fill. It can be measured in tokens (the default),
words, or characters. `result.spans` also tells you where each kept piece came from in
the original Python string.

## Have more than one source?

Use `trim_context()` when several sources must share one limit. This input has an initial
report and a follow-up:

```python
from trimwise import ContextSource, Trimmer

sources = [
    ContextSource(
        text="Initial checks focused on the network. The network was healthy.",
        prefix="Incident: ",
    ),
    ContextSource(
        text="A retry loop ignored backoff settings. Service recovered at 14:32 UTC.",
        prefix="Follow-up: ",
    ),
]

result = Trimmer().trim_context(
    sources,
    limit=20,
    query="Which retry loop caused the outage, and when did service recover?",
)
print(result.text)
print(f"{result.output_count}/{result.limit} tokens")
```

Output from running that code:

```text
Follow-up: A retry loop ignored backoff settings. Service recovered at 14:32 UTC.
20/20 tokens
```

Only the follow-up contributes to this result. You still get one result row per input source,
including an empty row for the incident report. Source labels count toward this limit; your
surrounding prompt instructions and answer space do not. See
[Many Sources, One Shared Limit](https://trimwise.readthedocs.io/en/latest/multi-source-context/)
for the full behavior.

## Which strategy should you use?

Start with the default, `auto`. With a question, it matches relevant words (`lexical`).
Without one, it makes a broader document excerpt (`structural`). Neither downloads a model.

If the answer may use different wording from your question, try `semantic`. `hybrid` combines
word matching and semantic matching. Those two need your own embedding callback or the
[optional semantic model](https://trimwise.readthedocs.io/en/latest/semantic-and-async/).
The [strategy guide](https://trimwise.readthedocs.io/en/latest/strategies/) has examples and
tradeoffs.

## What Trimwise does—and doesn't

Trimwise works on text you already have. It does not search for documents or rewrite them.
Unlike token-pruning compressors, it keeps readable source pieces; see the
[research comparison](https://trimwise.readthedocs.io/en/latest/research-foundations/#how-trimwise-compares-with-model-based-compression)
for the tradeoff.

A tight limit can leave out evidence needed to answer a question. The limit applies to
Trimwise's returned text, not your entire prompt, so leave room for instructions and the
model's answer. Read the [guarantees and limitations](https://trimwise.readthedocs.io/en/latest/guarantees-and-limitations/)
before relying on an excerpt for a high-stakes task.

## How it did in one test

In a frozen 160-case test using the published 0.2.0 release, Trimwise Hybrid kept every
annotated source passage in 66.9% of cases at a 512-token limit. This measures passage
survival on that test—not answer quality or performance on every document. The
[benchmark report](https://trimwise.readthedocs.io/en/latest/benchmark/) explains the
setup, comparisons, and limits.

![Complete annotated source-passage survival by output-token limit on 160 position-controlled cases, comparing Trimwise with evaluated adapters.](./assets/readme/query-aware-benchmark.svg)

In a separate 93-case short-answer check at 256 tokens, GPT-5.4 Nano matched 43 reference
answers with Hybrid's output, versus 41 with the full source (46.2% vs. 44.1%). GPT-5.4 Mini
scored 46.2% vs. 45.2%; GPT-5.6 Luna tied at 45.2%. Each context received one model response.
This automatic text match is a useful diagnostic, not a human correctness judgment or proof
that trimming generally improves answers.

At 512 tokens, Hybrid's median warm trim time was 42.8 ms on the benchmark machine
(Lexical: 6.4 ms). Those timings exclude cold model loading and are not portable speed
guarantees.

For more detail, see [Getting Started](https://trimwise.readthedocs.io/en/latest/getting-started/),
[Configuration and API](https://trimwise.readthedocs.io/en/latest/configuration-and-api/), or the
[API Reference](https://trimwise.readthedocs.io/en/latest/api-reference/).

Trimwise is available under the [MIT License](LICENSE) and maintained by [AATBIT Labs](https://aatbit.com).
