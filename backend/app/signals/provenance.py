"""E1 Provenance signal module: Camera hardware checks, label analysis, and frame timing jitter."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from app.signals.thresholds import (
    VIRTUAL_CAMERA_KEYWORDS,
    JITTER_TOO_SMOOTH_STD_MS,
    FRAME_DROP_RATIO_HIGH,
)


@dataclass
class ProvenanceResult:
    source_risk: float
    flags: List[str] = field(default_factory=list)
    camera_label: Optional[str] = None
    matched_keyword: Optional[str] = None
    jitter_std_ms: Optional[float] = None
    frame_drop_ratio: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_provenance(
    camera_label: Optional[str] = None,
    frame_intervals_ms: Optional[List[float]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> ProvenanceResult:
    """
    Evaluate camera hardware provenance, virtual driver signatures, and frame delivery jitter.
    Pure function with no I/O.
    """
    flags = []
    risk = 0.0
    matched_keyword = None
    jitter_std = None
    drop_ratio = 0.0

    # 1. Virtual Camera Signature Check
    if camera_label:
        label_lower = camera_label.lower().strip()
        for kw in VIRTUAL_CAMERA_KEYWORDS:
            if kw in label_lower:
                flags.append("VIRTUAL_CAMERA")
                matched_keyword = kw
                risk = max(risk, 0.85)
                break

    # 2. Frame-Interval Jitter Statistics
    # Real physical webcams have hardware USB/sensor timing jitter (~3ms to 15ms).
    # Synthetic video loops or programmatic virtual webcams often produce unnaturally uniform delivery (< 1.0ms).
    if frame_intervals_ms and len(frame_intervals_ms) >= 10:
        intervals = np.array(frame_intervals_ms, dtype=np.float64)
        jitter_std = float(np.std(intervals))

        # Check for unnaturally smooth frame intervals
        if jitter_std < JITTER_TOO_SMOOTH_STD_MS:
            flags.append("JITTER_TOO_SMOOTH")
            risk = max(risk, 0.40)

        # Check for abnormal frame drop ratio (intervals > 2x median)
        median_interval = float(np.median(intervals))
        if median_interval > 0:
            dropped_frames = np.sum(intervals > 2.2 * median_interval)
            drop_ratio = float(dropped_frames / len(intervals))
            if drop_ratio > FRAME_DROP_RATIO_HIGH:
                flags.append("HIGH_FRAME_DROPS")
                risk = max(risk, 0.30)

    # 3. Settings Anomalies (e.g. impossible frame rate or zero resolution)
    if settings:
        fps = settings.get("frameRate")
        w = settings.get("width")
        h = settings.get("height")
        if fps and (fps > 120 or fps < 5):
            flags.append("ANOMALOUS_FPS")
            risk = max(risk, 0.35)
        if w is not None and h is not None and (w <= 0 or h <= 0):
            flags.append("INVALID_RESOLUTION")
            risk = max(risk, 0.50)

    return ProvenanceResult(
        source_risk=float(np.clip(risk, 0.0, 1.0)),
        flags=flags,
        camera_label=camera_label,
        matched_keyword=matched_keyword,
        jitter_std_ms=jitter_std,
        frame_drop_ratio=drop_ratio,
        details={
            "camera_label": camera_label,
            "jitter_std_ms": jitter_std,
            "frame_drop_ratio": drop_ratio,
            "flags": flags,
        },
    )
