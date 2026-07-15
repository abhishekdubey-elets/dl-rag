"""Prometheus metrics registry and helpers.

Exposed at ``GET /metrics``. Use the module-level metric objects directly, or
the small context managers for timing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

# A dedicated registry keeps app metrics isolated and testable.
REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "dlrag_http_requests_total",
    "Total HTTP requests.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "dlrag_http_request_duration_seconds",
    "HTTP request latency.",
    labelnames=("method", "path"),
    buckets=(0.05, 0.1, 0.25, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

CHAT_LATENCY = Histogram(
    "dlrag_chat_latency_seconds",
    "End-to-end /api/chat latency.",
    buckets=(0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "dlrag_retrieval_latency_seconds",
    "Hybrid retrieval latency.",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0),
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "dlrag_llm_latency_seconds",
    "LLM completion latency.",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "dlrag_llm_tokens_total",
    "LLM tokens consumed.",
    labelnames=("kind",),  # prompt | completion
    registry=REGISTRY,
)

QUERY_TYPE_COUNT = Counter(
    "dlrag_query_type_total",
    "Queries by detected type.",
    labelnames=("query_type",),
    registry=REGISTRY,
)

LOW_CONFIDENCE_COUNT = Counter(
    "dlrag_low_confidence_total",
    "Answers returned with low/no evidence.",
    registry=REGISTRY,
)

INGESTED_DOCS = Counter(
    "dlrag_ingested_documents_total",
    "Documents ingested.",
    registry=REGISTRY,
)

INDEX_SIZE = Gauge(
    "dlrag_indexed_chunks",
    "Chunks currently indexed (best-effort gauge).",
    registry=REGISTRY,
)


@contextmanager
def observe(histogram: Histogram, *labels: str) -> Iterator[None]:
    """Time a code block into ``histogram`` (with optional label values)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        target = histogram.labels(*labels) if labels else histogram
        target.observe(time.perf_counter() - start)
