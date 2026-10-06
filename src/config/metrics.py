import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import metrics

logger = logging.getLogger(__name__)


_meter = metrics.get_meter("llm-agents")


WFQ_TASKS_TOTAL = _meter.create_counter(
    "wfq_tasks",
    description="WFQ tasks finished, labelled by table and status (done/failed)",
)
WFQ_TASK_DURATION = _meter.create_histogram(
    "wfq_task_duration", unit="s",
    description="Per-task handler wall time, labelled by table",
)
WFQ_TASKS_INFLIGHT = _meter.create_up_down_counter(
    "wfq_tasks_inflight",
    description="WFQ tasks currently being handled, labelled by table",
)
LLM_TOKENS_TOTAL = _meter.create_counter(
    "llm_tokens",
    description="LLM tokens, labelled by model and direction (input/output)",
)
LLM_REQUEST_DURATION = _meter.create_histogram(
    "llm_request_duration", unit="s",
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
