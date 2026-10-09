"""E8 Quality module: Evaluates blur, lighting, face size, and outputs quality_trust in [0, 1]."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
from app.signals.thresholds import (
    QUALITY_BLUR_LAPLACIAN_MIN,
    QUALITY_BRIGHTNESS_MIN,
    QUALITY_BRIGHTNESS_MAX,
    QUALITY_FACE_RATIO_MIN,
    QUALITY_TRUST_MIN,
)


@dataclass
class QualityResult:
    quality_trust: float           # 0.0 (unusable) to 1.0 (studio grade)
    blur_score: float              # Laplacian variance
    brightness: float              # Mean luminance 0-255
    face_ratio: float              # Face visibility ratio
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def evaluate_quality(
    blur: float,
    brightness: float,
    face_ratio: float,
    frame_drop_ratio: float = 0.0,
) -> QualityResult:
    """
    Compute composite quality trust score to gate deepfake classifiers and prevent false rejections.
    Pure function, zero I/O.
    """
    flags = []

    # 1. Blur assessment (Laplacian variance)
    # < 30 is heavily blurred; > 120 is very crisp
    blur_factor = float(np.clip((blur - 25.0) / 95.0, 0.0, 1.0))
    if blur < QUALITY_BLUR_LAPLACIAN_MIN:
        flags.append("MOTION_BLUR")

    # 2. Lighting assessment
    # Optimal luminance is between 60 and 190
    if brightness < QUALITY_BRIGHTNESS_MIN:
        flags.append("POOR_LIGHTING_DARK")
    elif brightness > QUALITY_BRIGHTNESS_MAX:
        flags.append("POOR_LIGHTING_OVEREXPOSED")
    bright_factor = 1.0 - float(np.clip(abs(brightness - 128.0) / 100.0, 0.0, 1.0))

    # 3. Face presence
    if face_ratio < QUALITY_FACE_RATIO_MIN:
        flags.append("FACE_NOT_VISIBLE")
    face_factor = float(np.clip(face_ratio, 0.0, 1.0))

    # 4. Temporal continuity factor
    temporal_factor = 1.0 - float(np.clip(frame_drop_ratio, 0.0, 1.0))

    # Composite trust score
    trust = 0.40 * blur_factor + 0.30 * bright_factor + 0.20 * face_factor + 0.10 * temporal_factor
    if face_ratio < QUALITY_FACE_RATIO_MIN:
        trust = trust * 0.40
    quality_trust = float(np.clip(trust, 0.0, 1.0))

    return QualityResult(
        quality_trust=quality_trust,
        blur_score=blur,
        brightness=brightness,
        face_ratio=face_ratio,
        flags=flags,
        details={
            "blur_factor": blur_factor,
            "bright_factor": bright_factor,
            "face_factor": face_factor,
            "quality_trust": quality_trust,
        },
    )
