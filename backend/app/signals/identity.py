"""E4 Identity Consistency & Landmark Flicker signal module."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from app.signals.thresholds import (
    IDENTITY_FLICKER_MIN_SIM,
    IDENTITY_FLICKER_MAX_STD,
    LANDMARK_JITTER_ENERGY_MAX,
)


@dataclass
class IdentityResult:
    flicker_score: float           # 0.0 (consistent) to 1.0 (high flicker / identity shift)
    min_cosine_sim: float          # Minimum similarity between consecutive frames
    mean_cosine_sim: float         # Mean similarity between consecutive frames
    std_cosine_sim: float          # Variance in facial identity
    jump_count: int                # Number of abrupt identity discontinuities
    landmark_jitter: float         # High-frequency landmark energy
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_identity_consistency(
    embeddings: Optional[List[np.ndarray]] = None,
    landmarks_series: Optional[List[np.ndarray]] = None,
) -> IdentityResult:
    """
    Measure face embedding stability and landmark trajectory jitter across 16 frames.
    Pure function, zero I/O.
    """
    min_sim = 1.0
    mean_sim = 1.0
    std_sim = 0.0
    jumps = 0
    jitter = 0.0
    flags = []

    # 1. Face embedding continuity analysis
    if embeddings and len(embeddings) >= 2:
        sims = []
        for i in range(len(embeddings) - 1):
            e1 = embeddings[i].flatten()
            e2 = embeddings[i + 1].flatten()
            norm1 = float(np.linalg.norm(e1))
            norm2 = float(np.linalg.norm(e2))
            if norm1 > 1e-6 and norm2 > 1e-6:
                sim = float(np.dot(e1, e2) / (norm1 * norm2))
            else:
                sim = 1.0
            sims.append(sim)
            if sim < 0.65:
                jumps += 1

        sims_np = np.array(sims, dtype=np.float64)
        min_sim = float(np.min(sims_np))
        mean_sim = float(np.mean(sims_np))
        std_sim = float(np.std(sims_np))

        if min_sim < IDENTITY_FLICKER_MIN_SIM or std_sim > IDENTITY_FLICKER_MAX_STD or jumps >= 2:
            flags.append("IDENTITY_FLICKER")

    # 2. Landmark trajectory high-frequency jitter
    if landmarks_series and len(landmarks_series) >= 3:
        # High-pass via discrete second difference: d2 = x[t+1] - 2*x[t] + x[t-1]
        lms_arr = np.array([lm[:, :2] for lm in landmarks_series], dtype=np.float64)  # (T, N, 2)
        d2 = lms_arr[2:] - 2.0 * lms_arr[1:-1] + lms_arr[:-2]
        # Energy across all points and time steps
        jitter = float(np.mean(np.linalg.norm(d2, axis=-1)))

        if jitter > LANDMARK_JITTER_ENERGY_MAX:
            flags.append("LANDMARK_JITTER")

    # Composite flicker score (0.0 = stable real, 1.0 = unstable swap)
    sim_penalty = max(0.0, 1.0 - min_sim)
    jitter_penalty = min(1.0, jitter / (LANDMARK_JITTER_ENERGY_MAX + 1e-6))
    flicker_score = float(np.clip(0.6 * sim_penalty + 0.4 * jitter_penalty, 0.0, 1.0))

    return IdentityResult(
        flicker_score=flicker_score,
        min_cosine_sim=min_sim,
        mean_cosine_sim=mean_sim,
        std_cosine_sim=std_sim,
        jump_count=jumps,
        landmark_jitter=jitter,
        flags=flags,
        details={
            "min_sim": min_sim,
            "mean_sim": mean_sim,
            "std_sim": std_sim,
            "jumps": jumps,
            "jitter": jitter,
        },
    )
