"""Prometheus-compatible metrics with no-op fallback.

If ``prometheus-client`` is not installed, all metric operations become no-ops
and the ``/metrics`` endpoint returns a short plain-text status line.
"""

from __future__ import annotations

from typing import Iterable, Tuple

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


class _NoopMetric:
    def labels(self, *_: object, **__: object) -> "_NoopMetric":
        return self

    def inc(self, _: float = 1.0) -> None:
        return None

    def observe(self, _: float) -> None:
        return None


class Metrics:
    """Single-instance metrics container."""

    def __init__(self) -> None:
        if _PROM_AVAILABLE:
            self.registry = CollectorRegistry()
            self.queries_total = Counter(
                "graphrag_queries_total",
                "Total queries processed",
                labelnames=("failure_type",),
                registry=self.registry,
            )
            self.ingest_total = Counter(
                "graphrag_ingest_total",
                "Total ingest invocations",
                registry=self.registry,
            )
            self.ingest_documents = Counter(
                "graphrag_ingest_documents_total",
                "Documents successfully ingested",
                registry=self.registry,
            )
            self.query_latency = Histogram(
                "graphrag_query_latency_seconds",
                "End-to-end query latency",
                buckets=_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.module_latency = Histogram(
                "graphrag_module_latency_seconds",
                "Per-module latency in the pipeline",
                labelnames=("module",),
                buckets=_LATENCY_BUCKETS,
                registry=self.registry,
            )
            self.errors_total = Counter(
                "graphrag_stage_errors_total",
                "Errors raised inside protected pipeline stages",
                labelnames=("stage",),
                registry=self.registry,
            )
        else:
            self.registry = None
            self.queries_total = _NoopMetric()
            self.ingest_total = _NoopMetric()
            self.ingest_documents = _NoopMetric()
            self.query_latency = _NoopMetric()
            self.module_latency = _NoopMetric()
            self.errors_total = _NoopMetric()

    def observe_modules(self, timings_ms: Iterable[Tuple[str, float]]) -> None:
        for name, ms in timings_ms:
            self.module_latency.labels(module=name).observe(ms / 1000.0)

    def render(self) -> tuple[bytes, str]:
        if not _PROM_AVAILABLE or self.registry is None:
            return (b"# prometheus-client not installed\n", CONTENT_TYPE_LATEST)
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


METRICS = Metrics()
