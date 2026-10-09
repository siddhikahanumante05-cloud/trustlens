"""E6b Bilabial Lip Closure signal module.
Validates complete physical lip closure (aperture <= 0.12) at the onset of words starting with B, P, or M.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from app.signals.thresholds import BILABIAL_APERTURE_MAX
from app.signals.phrase import WordTimestamp


@dataclass
class BilabialViolation:
    word: str
    letter: str
    t_sec: float
    observed_aperture: float


@dataclass
class LipClosureResult:
    bilabial_aperture_err: float          # Max aperture violation over bilabials
    passed: bool
    violations: List[BilabialViolation] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_lip_closure(
    aperture_series: List[float],
    frame_timestamps: List[float],
    word_timestamps: List[WordTimestamp],
) -> LipClosureResult:
    """
    Evaluate mouth closure at acoustic onsets of bilabial phonemes (B, P, M).
    Pure function, zero I/O.
    """
    if not aperture_series or not frame_timestamps or not word_timestamps:
        return LipClosureResult(
            bilabial_aperture_err=0.0,
            passed=True,
            violations=[],
            flags=[],
            details={"reason": "no_words_or_frames"},
        )

    violations: List[BilabialViolation] = []
    max_aperture_err = 0.0
    ts_arr = np.array(frame_timestamps, dtype=np.float64)

    # Check words that begin with B, P, or M phonemes
    for wt in word_timestamps:
        word_clean = wt.word.strip().upper()
        if not word_clean:
            continue
        first_letter = word_clean[0]
        if first_letter in ["B", "P", "M"]:
            # Find frames in the temporal neighborhood [-120 ms, +120 ms] of word onset
            onset = wt.start_sec
            in_window = np.where((ts_arr >= onset - 0.12) & (ts_arr <= onset + 0.12))[0]

            if len(in_window) > 0:
                window_apertures = [aperture_series[idx] for idx in in_window]
                min_aperture = float(np.min(window_apertures))
            else:
                # Nearest frame
                nearest_idx = int(np.argmin(np.abs(ts_arr - onset)))
                min_aperture = float(aperture_series[nearest_idx])

            if min_aperture > BILABIAL_APERTURE_MAX:
                violation = BilabialViolation(
                    word=wt.word,
                    letter=first_letter,
                    t_sec=float(onset),
                    observed_aperture=min_aperture,
                )
                violations.append(violation)
                max_aperture_err = max(max_aperture_err, min_aperture - BILABIAL_APERTURE_MAX)

    flags = []
    if len(violations) > 0:
        flags.append("LIP_CLOSURE_MISSING")

    return LipClosureResult(
        bilabial_aperture_err=float(max_aperture_err),
        passed=(len(violations) == 0),
        violations=violations,
        flags=flags,
        details={
            "violations_count": len(violations),
            "max_err": max_aperture_err,
        },
    )
