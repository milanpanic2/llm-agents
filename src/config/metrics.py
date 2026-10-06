import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import metrics

logger = logging.getLogger(__name__)

# One meter for the whole service. Instruments are created here at import time
# from a proxy meter and get rebound to the real MeterProvider once
# init_telemetry() / init_worker_telemetry() installs it at startup -> so these
# module-level singletons are safe, same idea as the engine in connection.py.
#
# Nothing extra is needed to reach Prometheus: everything recorded on these
# instruments rides the OTLP metric pipeline already set up in telemetry.py:
#   MeterProvider -> OTLPMetricExporter(:4317) -> Alloy otelcol.receiver.otlp
#   -> otelcol.exporter.prometheus -> prometheus.remote_write -> Prometheus.
_meter = metrics.get_meter("llm-agents")


# --- generic WFQ queue metrics (recorded inside WFQEngine) -------------------
# Keyed by `table` (= the queue) and `status`, so every feature running on the
# engine shares one dashboard. Add a new queue/table and it shows up for free.
WFQ_TASKS_TOTAL = _meter.create_counter(
    "wfq_tasks_total",
    description="WFQ tasks finished, labelled by table and status (done/failed)",
)
WFQ_TASK_DURATION = _meter.create_histogram(
    "wfq_task_duration_seconds", unit="s",
    description="Per-task handler wall time, labelled by table",
)
WFQ_TASKS_INFLIGHT = _meter.create_up_down_counter(
    "wfq_tasks_inflight",
    description="WFQ tasks currently being handled, labelled by table",
)


# --- domain metrics: LLM usage (recorded at the agent call site) -------------
LLM_TOKENS_TOTAL = _meter.create_counter(
    "llm_tokens_total",
    description="LLM tokens, labelled by model and direction (input/output)",
)
LLM_REQUEST_DURATION = _meter.create_histogram(
    "llm_request_duration_seconds", unit="s",
    description="LLM request wall time, labelled by model",
)


@contextmanager
def record_duration(histogram, attributes: dict | None = None) -> Generator[None]:
    """Time the wrapped block and record the elapsed seconds into `histogram`.
    Sync context manager, but fine to wrap an awaited call -- it just measures
    wall time around whatever runs inside the `with`."""
    started = time.perf_counter()
    try:
        yield
    finally:
        histogram.record(time.perf_counter() - started, attributes or {})
