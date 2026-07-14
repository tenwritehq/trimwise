# API Reference

This page documents Trimwise **{{ trimwise_version }}** directly from its public Python source.
Signatures, type annotations, and descriptions therefore stay aligned with the installed API.
Expand **Source code** under any entry to inspect its implementation and source line numbers.

Trimwise intentionally exports only the six objects below. Internal segmentation, measurement,
ranking, semantic-adapter, and orchestration helpers are not part of the compatibility promise.

## Trimming

Use [`Trimmer`][trimwise.Trimmer] for synchronous or asynchronous trimming.

::: trimwise.Trimmer

## Configuration

Use [`TrimConfig`][trimwise.TrimConfig] to configure token encoding, managed embeddings, MMR,
and omission markers.

::: trimwise.TrimConfig

## Result

Every call returns an immutable [`TrimResult`][trimwise.TrimResult] containing the excerpt,
measured counts, resolved strategy, and trimming status.

::: trimwise.TrimResult

## Strategies

[`Strategy`][trimwise.Strategy] selects structural, lexical, semantic, hybrid, or automatic
ranking behavior.

::: trimwise.Strategy

## Budget units

[`BudgetUnit`][trimwise.BudgetUnit] chooses token, word, or character measurement.

::: trimwise.BudgetUnit

## Semantic backend errors

[`SemanticBackendError`][trimwise.SemanticBackendError] reports embedding import,
initialization, callback, and inference failures without silently changing strategy.

::: trimwise.SemanticBackendError
