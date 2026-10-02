import logging
import sys

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config.settings import settings

OTEL_ENDPOINT = settings.otel_endpoint

# trace/span ids injected by LoggingInstrumentor so Grafana can jump log <-> trace
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] trace=%(otelTraceID)s span=%(otelSpanID)s %(message)s"


def init_telemetry(app, db_engine):
    """Full telemetry for the FastAPI process."""
    resource = _resource()
    export_traces_to_tempo_via_alloy(resource)
    export_metrics_to_prometheus_via_alloy(resource)
    setup_stdout_logging()
    instrument_fastapi_and_sqlalchemy(app, db_engine)


def init_worker_telemetry(db_engine):
    """Telemetry for the standalone worker process (no FastAPI app)."""
    resource = _resource()
    export_traces_to_tempo_via_alloy(resource)
    export_metrics_to_prometheus_via_alloy(resource)
    setup_stdout_logging()
    SQLAlchemyInstrumentor().instrument(engine=db_engine)


def _resource() -> Resource:
    return Resource.create({
        SERVICE_NAME: settings.service_name,
        SERVICE_VERSION: "0.1.0",
        "deployment.environment": settings.environment,
    })


def export_traces_to_tempo_via_alloy(resource: Resource):
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)


def export_metrics_to_prometheus_via_alloy(resource: Resource):
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True),
        export_interval_millis=15000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)


def setup_stdout_logging():
    """Logs go to stdout only; Alloy scrapes the container logs into Loki.
    LoggingInstrumentor injects otelTraceID / otelSpanID onto every record for trace correlation."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    # otelTraceID/otelSpanID are injected by LoggingInstrumentor, but records created
    # outside its factory (early startup, C-level) won't have them -> default to "0"
    # so the formatter never raises KeyError.
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, defaults={"otelTraceID": "0", "otelSpanID": "0"}))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    LoggingInstrumentor().instrument()


def instrument_fastapi_and_sqlalchemy(app, db_engine):
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument(engine=db_engine)
