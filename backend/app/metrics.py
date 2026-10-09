"""Prometheus metrics exporter for TrustLens backend."""
from typing import Dict
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics definitions
STUBS_IN_USE = Gauge(
    "trustlens_stubs_in_use",
    "Flag indicating whether stub models are currently in use",
    ["version"],
)
STUBS_IN_USE.labels(version="stub-0").set(1)

ACTIVE_SESSIONS = Gauge(
    "trustlens_active_sessions",
    "Number of currently active streaming sessions",
)

DROPPED_WINDOWS = Counter(
    "trustlens_dropped_windows_total",
    "Total number of dropped windows due to pipeline lag",
)

FFMPEG_RESTARTS = Counter(
    "trustlens_ffmpeg_restarts_total",
    "Total number of ffmpeg process restarts or crashes",
)

FAST_PATH_LATENCY = Histogram(
    "trustlens_fast_path_latency_seconds",
    "Latency of 1-second fast-path analysis per window",
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
)

SLOW_PATH_LATENCY = Histogram(
    "trustlens_slow_path_latency_seconds",
    "Latency of 3-second slow-path analysis (CLIP/WavLM)",
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0],
)


def get_prometheus_metrics() -> bytes:
    """Generate the latest Prometheus metrics in text format."""
    return generate_latest()
