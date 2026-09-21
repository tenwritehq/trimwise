---
title: Observe Trimwise with OpenTelemetry
description: Add Trimwise spans to your application's existing traces without giving the library control of exporters, endpoints, or credentials.
---

# Observe Trimwise with OpenTelemetry

Trimwise can add its work to your application's distributed traces. Tracing is optional: if your
application does not configure an OpenTelemetry SDK, the installed OpenTelemetry API is a no-op
and Trimwise sends nothing over the network.

Trimwise creates spans only. Your application chooses whether to collect them and owns the SDK,
exporter, collector endpoint, authentication headers, credentials, TLS, resources, sampling,
batching, flushing, and shutdown. These operational settings do not belong in `TrimConfig`, which
remains limited to trimming behavior. This follows the
[OpenTelemetry guidance for instrumented libraries](https://opentelemetry.io/docs/specs/otel/library-guidelines/):
libraries use the API, while the final application configures the SDK and exporters.

## What you get

Each public operation creates one span:

| Call | Span name |
| --- | --- |
| `trim()` and `atrim()` | `trimwise.trim` |
| `trim_context()` and `atrim_context()` | `trimwise.trim_context` |
| `atrim_many()` | `trimwise.trim_many` |

Sync and async forms use the same names so a dashboard does not need separate queries. A batch
creates one span for the whole call rather than one span per input.

When your application already has a current span, the Trimwise span becomes its child. This works
through the worker threads used by `atrim()` and `atrim_context()`, and through an awaited async
embedding callback.

Successful single-source and context spans use these attributes:

| Attribute | Meaning |
| --- | --- |
| `trimwise.strategy.requested` | Valid strategy passed by the caller, including `auto` |
| `trimwise.strategy.resolved` | Strategy actually used after resolving `auto` |
| `trimwise.unit` | `tokens`, `words`, or `characters` |
| `trimwise.limit` | Requested output ceiling |
| `trimwise.input.count` | Measured input size in `trimwise.unit` |
| `trimwise.output.count` | Measured output size in `trimwise.unit` |
| `trimwise.trimmed` | Whether any source text was removed |
| `trimwise.source.count` | Number of input sources for a context call |

A successful `trimwise.trim_many` span records `trimwise.batch.size` and whether any item was
trimmed. Batch inputs may use different units, so Trimwise does not add misleading aggregate input
or output counts.

Failed and cancelled calls set the span status to error and add `error.type`, such as
`builtins.ValueError`. Trimwise does not attach the exception message or stack trace.

## Send traces with OTLP/gRPC

Install the SDK and the gRPC exporter in your **application**:

```bash
python -m pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Configure OpenTelemetry once when your application starts:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


provider = TracerProvider(
    resource=Resource.create(
        {
            "service.name": "answer-api",
            "deployment.environment.name": "production",
        }
    )
)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

The exporter reads `OTEL_EXPORTER_OTLP_ENDPOINT`. Use your collector's TLS and authentication
settings in production. Call `provider.shutdown()` from your application's shutdown hook so the
batch processor can flush pending spans.

## Send traces with OTLP/HTTP

Use the HTTP exporter when your collector accepts OTLP over HTTP/protobuf:

```bash
python -m pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

The provider setup is the same except for the exporter import:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


provider = TracerProvider(resource=Resource.create({"service.name": "answer-api"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

With the base endpoint above, the HTTP exporter sends traces to `/v1/traces`. Call
`provider.shutdown()` when the application stops.

## See Trimwise inside an application trace

Once the provider is configured, use Trimwise normally:

```python
from opentelemetry import trace
from trimwise import Trimmer


app_tracer = trace.get_tracer("answer-api")
trimmer = Trimmer()

with app_tracer.start_as_current_span("answer.generate") as span:
    span.set_attribute("app.plan", "standard")
    result = trimmer.trim(
        document,
        limit=500,
        strategy="lexical",
        query="What caused the outage?",
    )
```

The trace has this shape:

```text
answer.generate
└── trimwise.trim
```

Put application-specific fields on your `Resource` or parent span as shown above. That keeps
deployment and tenant metadata under application control. Trimwise does not accept an unrestricted
custom-attribute dictionary.

## Privacy and cardinality

Trimwise records operation metadata, not content. Its spans never include:

- Input or output text.
- Queries.
- Context prefixes, suffixes, or separators.
- Embedding passages or vectors.
- Exception messages or stack traces.
- Collector endpoints, headers, credentials, or other configuration.

Do not put source text, queries, user IDs, request IDs, or other high-cardinality or sensitive
values on parent spans unless your own telemetry policy permits them. OpenTelemetry context flows
into embedding callbacks, so spans created by your callback can become descendants of the
Trimwise span; the callback remains responsible for its own attributes and privacy controls.

## What Trimwise does not configure

Trimwise depends only on `opentelemetry-api`. It does not install or configure:

- An OpenTelemetry SDK.
- OTLP/gRPC or OTLP/HTTP exporters.
- A collector or observability backend.
- Sampling, queues, retries, or batch limits.
- Metrics or log export.
- Global propagators or resource attributes.

Your collector can derive request counts, error rates, and duration histograms from spans if its
span-metrics connector is enabled. That is a collector choice rather than a Trimwise runtime
feature.

## Turn tracing off

Do not configure an OpenTelemetry SDK, or configure your application's sampler to drop these
spans. No Trimwise flag is required. The OpenTelemetry API safely returns no-op spans when no SDK
provider is installed.

## Troubleshooting

**No Trimwise spans appear:** verify that the SDK is configured before the first trim call, the
exporter package matches your collector protocol, and the exporter endpoint uses port `4317` for
gRPC or `4318` for HTTP by convention.

**Spans appear under the wrong service:** set `service.name` on the application's `Resource`.
Trimwise deliberately does not choose a service name.

**The process exits before spans arrive:** call `provider.shutdown()` during application shutdown.
Do not call it after every trim.

**You expected metrics or logs:** Trimwise emits trace spans only. Configure application logging
and collector-derived span metrics separately.

## Continue exploring

- Follow the [Getting Started guide](getting-started.md).
- Review every call and result in [Configuration and API Reference](configuration-and-api.md).
- Understand worker threads and callbacks in [Semantic Models and Async Usage](semantic-and-async.md).
- Read the [guarantees and limitations](guarantees-and-limitations.md).
