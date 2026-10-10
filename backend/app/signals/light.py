"""E2 Light Response signal module.
Measures active chromaticity reflections on face and neck vs challenge color sequence,
aligns using self-referenced white sync pulse, and computes cross-correlation and SNR.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.signals.thresholds import (
    LIGHT_CORR_FACE_THRESHOLD,
    LIGHT_SNR_MIN,
    LIGHT_MAX_LAG_SEARCH_MS,
    LIGHT_LATE_LAG_MS,
    LIGHT_NECK_FACE_CORR_MIN,
    LIGHT_NECK_FACE_RATIO_MIN,
)


@dataclass
class LightResult:
    light_score: float             # 0.0 (fake) to 1.0 (real authentic reflection)
    corr_face: float               # Cross-correlation of face reflection vs challenge
    lag_ms: float                  # Estimated reflection latency in ms
    neck_face_corr: float          # Correlation between neck and face reflection
    neck_face_ratio: float         # Power ratio of neck response vs face response
    snr: float                     # Signal-to-noise ratio of reflection
    sync_pulse_found: bool         # Whether the white sync pulse was identified
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def hex_to_chromaticity(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex string (e.g. #ff4d4d) to normalized (r, g, brightness)."""
    hex_clean = hex_color.lstrip("#")
    if len(hex_clean) != 6:
        return 0.333, 0.333, 128.0
    R = int(hex_clean[0:2], 16)
    G = int(hex_clean[2:4], 16)
    B = int(hex_clean[4:6], 16)
    total = R + G + B + 1e-6
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return R / total, G / total, brightness


def build_challenge_target_series(
    challenge_sequence: List[Dict[str, Any]],
    n_frames: int,
    fps: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample challenge color sequence onto n_frames at given fps."""
    r_target_list = []
    g_target_list = []
    y_target_list = []
    for step in challenge_sequence:
        c_r, c_g, c_y = hex_to_chromaticity(step.get("color", "#ffffff"))
        duration_frames = max(1, int(round((step.get("ms", 350) / 1000.0) * fps)))
        r_target_list.extend([c_r] * duration_frames)
        g_target_list.extend([c_g] * duration_frames)
        y_target_list.extend([c_y] * duration_frames)

    target_len = len(r_target_list)
    if target_len < n_frames:
        r_target_list.extend([r_target_list[-1]] * (n_frames - target_len))
        g_target_list.extend([g_target_list[-1]] * (n_frames - target_len))
        y_target_list.extend([y_target_list[-1]] * (n_frames - target_len))

    r_target = np.array(r_target_list[:n_frames], dtype=np.float64)
    g_target = np.array(g_target_list[:n_frames], dtype=np.float64)
    y_target = np.array(y_target_list[:n_frames], dtype=np.float64)
    return r_target, g_target, y_target


def analyze_light_response(
    face_rois: List[Dict[str, float]],
    neck_rois: Optional[List[Dict[str, float]]] = None,
    bg_rois: Optional[List[Dict[str, float]]] = None,
    timestamps: Optional[List[float]] = None,
    challenge_sequence: Optional[List[Dict[str, Any]]] = None,
    challenge_t0: Optional[float] = None,
    tap_mode: str = "agent",
) -> LightResult:
    """
    Evaluate physiological screen reflection on facial skin.
    Uses self-referenced timing from in-video sync pulse.
    Pure function, zero I/O.
    """
    if not face_rois or len(face_rois) < 6 or not challenge_sequence:
        return LightResult(
            light_score=0.5,
            corr_face=0.50,
            lag_ms=50.0,
            neck_face_corr=0.75,
            neck_face_ratio=0.80,
            snr=3.0,
            sync_pulse_found=False,
            flags=[],
            details={"reason": "no_challenge_or_insufficient_data"},
        )

    n_frames = len(face_rois)
    r_face = np.array([f.get("r", 0.333) for f in face_rois], dtype=np.float64)
    g_face = np.array([f.get("g", 0.333) for f in face_rois], dtype=np.float64)
    y_face = np.array([0.299 * f.get("R", 128) + 0.587 * f.get("G", 128) + 0.114 * f.get("B", 128) for f in face_rois], dtype=np.float64)

    # 1. Target color series
    r_target, g_target, y_target = build_challenge_target_series(challenge_sequence, n_frames=n_frames)

    # 2. Self-referenced white sync pulse alignment
    sync_pulse_found = False
    sync_shift = 0
    if len(y_face) >= 6:
        face_lum_diff = np.diff(y_face)
        if len(face_lum_diff) > 0:
            peak_idx = int(np.argmax(face_lum_diff))
            if 0 <= peak_idx < len(y_face) // 2:
                sync_pulse_found = True
                sync_shift = peak_idx

    # 3. High-pass filter & centering across all color channels (r, g, y)
    r_face_hp = r_face - np.mean(r_face)
    g_face_hp = g_face - np.mean(g_face)
    y_face_hp = y_face - np.mean(y_face)

    r_target_hp = r_target - np.mean(r_target)
    g_target_hp = g_target - np.mean(g_target)
    y_target_hp = y_target - np.mean(y_target)

    # Apply sync pulse alignment shift if detected
    if sync_pulse_found and sync_shift > 0:
        # Pre-align target sequence to sync pulse onset
        r_target_hp = np.roll(r_target_hp, sync_shift)
        g_target_hp = np.roll(g_target_hp, sync_shift)
        y_target_hp = np.roll(y_target_hp, sync_shift)

    # 4. Normalized cross-correlation across causal lag search using all color channels
    def _channel_corr(s_obs: np.ndarray, s_tgt: np.ndarray, max_shifts: int) -> Tuple[float, int]:
        best_c = -1.0
        best_s = 0
        for shift in range(0, max_shifts + 1):
            cur_obs = s_obs[shift:] if shift > 0 else s_obs
            cur_tgt = s_tgt[:-shift] if shift > 0 else s_tgt
            if len(cur_obs) >= 8 and np.std(cur_obs) > 1e-5 and np.std(cur_tgt) > 1e-5:
                c = float(np.corrcoef(cur_obs, cur_tgt)[0, 1])
                if c > best_c:
                    best_c = c
                    best_s = shift
        return best_c, best_s

    max_lag_frames = min(5, int((LIGHT_MAX_LAG_SEARCH_MS / 1000.0) * 15.0))
    corr_r, shift_r = _channel_corr(r_face_hp, r_target_hp, max_lag_frames)
    corr_g, shift_g = _channel_corr(g_face_hp, g_target_hp, max_lag_frames)
    corr_y, shift_y = _channel_corr(y_face_hp, y_target_hp, max_lag_frames)

    # Multi-channel weighted correlation (R channel has strongest skin reflectance contrast)
    weights = []
    corrs = []
    if corr_r > -1.0:
        corrs.append(corr_r)
        weights.append(0.60)
    if corr_g > -1.0:
        corrs.append(corr_g)
        weights.append(0.20)
    if corr_y > -1.0:
        corrs.append(corr_y)
        weights.append(0.20)

    if weights:
        w_sum = sum(weights)
        corr_face = float(sum(c * (w / w_sum) for c, w in zip(corrs, weights)))
        best_lag_frames = shift_r
    else:
        corr_face = float(max(-1.0, min(1.0, corr_r)))
        best_lag_frames = 0

    corr_face = float(max(-1.0, min(1.0, corr_face)))
    estimated_lag_ms = float(best_lag_frames * (1000.0 / 15.0))

    # 5. Neck vs Face check
    neck_face_corr = 0.80
    neck_face_ratio = 0.85
    if neck_rois and len(neck_rois) == n_frames:
        r_neck = np.array([n.get("r", 0.333) for n in neck_rois], dtype=np.float64)
        r_neck_hp = r_neck - np.mean(r_neck)
        if np.std(r_neck_hp) > 1e-6 and np.std(r_face_hp) > 1e-6:
            neck_face_corr = float(np.corrcoef(r_neck_hp, r_face_hp)[0, 1])
            neck_var = float(np.var(r_neck_hp))
            face_var = float(np.var(r_face_hp)) + 1e-6
            neck_face_ratio = float(min(neck_var / face_var, face_var / (neck_var + 1e-6)))

    # 6. SNR estimation
    signal_power = float(np.var(r_face_hp))
    residual_noise = float(np.mean((r_face_hp - corr_face * r_target_hp) ** 2)) + 1e-6
    snr = float(signal_power / residual_noise)

    # 7. Evaluate flags
    flags = []
    if corr_face < LIGHT_CORR_FACE_THRESHOLD:
        flags.append("NO_LIGHT_REACTION")

    if neck_face_corr < LIGHT_NECK_FACE_CORR_MIN:
        flags.append("NECK_FACE_MISMATCH")

    if tap_mode != "agent" and estimated_lag_ms > LIGHT_LATE_LAG_MS:
        flags.append("LATE_LIGHT_REACTION")

    # Composite light confidence score (1.0 = authentic, 0.0 = fake)
    corr_norm = np.clip((corr_face + 1.0) / 2.0, 0.0, 1.0)
    neck_norm = np.clip((neck_face_corr + 1.0) / 2.0, 0.0, 1.0)
    light_score = float(np.clip(0.65 * corr_norm + 0.35 * neck_norm, 0.0, 1.0))

    return LightResult(
        light_score=light_score,
        corr_face=corr_face,
        lag_ms=estimated_lag_ms,
        neck_face_corr=neck_face_corr,
        neck_face_ratio=neck_face_ratio,
        snr=snr,
        sync_pulse_found=sync_pulse_found,
        flags=flags,
        details={
            "corr_face": corr_face,
            "neck_face_corr": neck_face_corr,
            "estimated_lag_ms": estimated_lag_ms,
            "snr": snr,
            "tap_mode": tap_mode,
        },
    )


def simulate_light_response(
    is_real: bool,
    challenge_sequence: List[Dict[str, Any]],
    n_frames: int = 16,
    snr_level: float = 3.0,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Synthetic generator for testing and ROC/AUC benchmark.
    - Real: face signal = challenge * reflectance + Gaussian noise; neck follows face.
    - Fake: face signal has zero/inverted/uncorrelated response; neck may not match.
    """
    rng = np.random.RandomState(seed)
    target_r, target_g, target_y = build_challenge_target_series(challenge_sequence, n_frames=n_frames)

    noise_std = 0.03 / max(0.1, snr_level)

    if is_real:
        reflectance = 0.50
        noise = rng.normal(0, noise_std, size=n_frames)
        face_r = 0.35 + reflectance * (target_r - np.mean(target_r)) + noise
        neck_noise = rng.normal(0, noise_std, size=n_frames)
        neck_r = 0.35 + 0.90 * reflectance * (target_r - np.mean(target_r)) + neck_noise
        face_y = 130.0 + 40.0 * (target_y - np.mean(target_y))
    else:
        noise = rng.normal(0, noise_std * 2.0, size=n_frames)
        # Deepfake overlay: no screen reflection on face
        face_r = 0.35 + noise
        neck_noise = rng.normal(0, noise_std, size=n_frames)
        neck_r = 0.35 + 0.20 * (target_r - np.mean(target_r)) + neck_noise
        face_y = 130.0 + noise * 5.0

    face_rois = [{"r": float(r), "g": 0.33, "b": 0.33, "R": float(y), "G": float(y * 0.9), "B": float(y * 0.8)} for r, y in zip(face_r, face_y)]
    neck_rois = [{"r": float(r), "g": 0.33, "b": 0.33, "R": 140.0, "G": 120.0, "B": 110.0} for r in neck_r]
    return face_rois, neck_rois
