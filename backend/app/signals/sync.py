"""E6c Audio-Visual Synchronization module.
Computes cross-correlation between audio RMS energy envelope and mouth aperture series.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from app.signals.thresholds import SYNC_OFFSET_MAX_MS, SYNC_CONF_MIN


@dataclass
class SyncResult:
    sync_conf: float               # Confidence / peak-to-average ratio in correlation
    offset_ms: float               # Estimated AV lag offset in milliseconds
    in_sync: bool
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def analyze_av_sync(
    audio_pcm: np.ndarray,
    mouth_apertures: List[float],
    sample_rate: int = 16000,
    fps: float = 15.0,
) -> SyncResult:
    """
    Lightweight CPU cross-correlation between audio RMS envelope and mouth aperture.
    Pure function, zero I/O.
    """
    if len(audio_pcm) < sample_rate or len(mouth_apertures) < 6:
        return SyncResult(
            sync_conf=2.0,
            offset_ms=0.0,
            in_sync=True,
            flags=[],
            details={"reason": "insufficient_samples"},
        )

    # 1. Compute audio RMS envelope aligned to video frames
    samples_per_frame = int(round(sample_rate / fps))
    n_frames = len(mouth_apertures)
    rms_envelope = []

    for i in range(n_frames):
        start = i * samples_per_frame
        end = min(len(audio_pcm), start + samples_per_frame)
        if start < len(audio_pcm) and end > start:
            segment = audio_pcm[start:end].astype(np.float64)
            rms = float(np.sqrt(np.mean(segment ** 2) + 1e-6))
        else:
            rms = 0.0
        rms_envelope.append(rms)

    env_arr = np.array(rms_envelope, dtype=np.float64)
    aperture_arr = np.array(mouth_apertures, dtype=np.float64)

    # High-pass / zero-center
    env_hp = env_arr - np.mean(env_arr)
    aperture_hp = aperture_arr - np.mean(aperture_arr)

    std_env = np.std(env_hp)
    std_ap = np.std(aperture_hp)

    if std_env < 1e-5 or std_ap < 1e-5:
        return SyncResult(
            sync_conf=2.0,
            offset_ms=0.0,
            in_sync=True,
            flags=[],
            details={"reason": "silent_audio_or_still_mouth"},
        )

    # 2. Cross-correlation over +/- 300 ms lags (shifts -4 to +4 frames at 15 fps)
    max_lag_frames = int(round(0.300 * fps))
    lags = list(range(-max_lag_frames, max_lag_frames + 1))
    corrs = []

    for shift in lags:
        if shift < 0:
            s_env = env_hp[:shift]
            s_ap = aperture_hp[-shift:]
        elif shift > 0:
            s_env = env_hp[shift:]
            s_ap = aperture_hp[:-shift]
        else:
            s_env = env_hp
            s_ap = aperture_hp

        if len(s_env) > 3:
            c = float(np.corrcoef(s_env, s_ap)[0, 1])
        else:
            c = 0.0
        corrs.append(c)

    corrs_arr = np.array(corrs, dtype=np.float64)
    best_idx = int(np.argmax(corrs_arr))
    best_lag = lags[best_idx]
    offset_ms = float(best_lag * (1000.0 / fps))

    # Peak-to-mean confidence
    peak_val = corrs_arr[best_idx]
    mean_val = float(np.mean(np.abs(corrs_arr))) + 1e-6
    sync_conf = float(max(0.0, peak_val / mean_val))

    flags = []
    if abs(offset_ms) > SYNC_OFFSET_MAX_MS or (sync_conf < SYNC_CONF_MIN and peak_val < 0.20):
        flags.append("SYNC_OFFSET")

    return SyncResult(
        sync_conf=sync_conf,
        offset_ms=offset_ms,
        in_sync=(len(flags) == 0),
        flags=flags,
        details={
            "peak_corr": peak_val,
            "sync_conf": sync_conf,
            "offset_ms": offset_ms,
        },
    )
