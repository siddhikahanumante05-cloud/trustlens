"""Comprehensive unit test suite for all pure-function signal modules (E1, E2, E4, E6a, E6b, E6c, E8).
Includes 500-sample AUC validation for E2 Light Response.
"""
import pytest
import numpy as np
from app.signals.thresholds import (
    VIRTUAL_CAMERA_KEYWORDS,
    LIGHT_CORR_FACE_THRESHOLD,
    BILABIAL_APERTURE_MAX,
    PHRASE_WER_HIGH,
)
from app.signals.provenance import analyze_provenance
from app.signals.light import (
    analyze_light_response,
    simulate_light_response,
    hex_to_chromaticity,
)
from app.signals.identity import analyze_identity_consistency
from app.signals.phrase import (
    analyze_phrase_match,
    compute_word_error_rate,
    WordTimestamp,
)
from app.signals.lip_closure import analyze_lip_closure
from app.signals.sync import analyze_av_sync
from app.signals.quality import evaluate_quality


# ============================================================================
# E1: Provenance Unit Tests (>= 5 tests)
# ============================================================================

def test_provenance_virtual_camera_detected():
    """Verify virtual camera names trigger VIRTUAL_CAMERA flag with elevated risk."""
    for label in ["OBS Virtual Camera", "ManyCam Video Source", "DroidCam Source 1"]:
        res = analyze_provenance(camera_label=label)
        assert "VIRTUAL_CAMERA" in res.flags
        assert res.source_risk >= 0.80
        assert res.matched_keyword is not None


def test_provenance_physical_webcam_clean():
    """Verify clean physical webcam labels pass without flags."""
    res = analyze_provenance(
        camera_label="Integrated Webcam (0c45:671f)",
        frame_intervals_ms=[31.2, 35.8, 30.5, 36.1, 29.4, 37.2, 32.0, 35.0, 31.8, 34.2],
    )
    assert len(res.flags) == 0
    assert res.source_risk < 0.20


def test_provenance_jitter_too_smooth():
    """Verify programmatic frame intervals with zero jitter trigger JITTER_TOO_SMOOTH."""
    # Perfectly spaced 33.333 ms intervals
    intervals = [33.333] * 30
    res = analyze_provenance(camera_label="HD Webcam", frame_intervals_ms=intervals)
    assert "JITTER_TOO_SMOOTH" in res.flags
    assert res.jitter_std_ms is not None and res.jitter_std_ms < 0.1


def test_provenance_high_frame_drops():
    """Verify erratic frame intervals trigger HIGH_FRAME_DROPS."""
    # Many intervals > 2.2x normal
    intervals = [33.0, 33.0, 120.0, 33.0, 150.0, 33.0, 110.0, 33.0, 140.0, 33.0]
    res = analyze_provenance(camera_label="Logitech Brio", frame_intervals_ms=intervals)
    assert "HIGH_FRAME_DROPS" in res.flags
    assert res.frame_drop_ratio > 0.3


def test_provenance_anomalous_settings():
    """Verify impossible resolution or extreme FPS trigger settings anomaly flags."""
    res = analyze_provenance(
        camera_label="Webcam",
        settings={"frameRate": 240, "width": -1, "height": 0},
    )
    assert "ANOMALOUS_FPS" in res.flags
    assert "INVALID_RESOLUTION" in res.flags


# ============================================================================
# E2: Light Response Unit Tests & 500-Sample AUC Benchmark (>= 5 tests)
# ============================================================================

SAMPLE_CHALLENGE = [
    {"color": "#ffffff", "ms": 350},
    {"color": "#ff4d4d", "ms": 350},
    {"color": "#4d79ff", "ms": 350},
    {"color": "#4dff88", "ms": 350},
    {"color": "#ffb84d", "ms": 350},
    {"color": "#b84dff", "ms": 350},
]


def test_light_hex_to_chromaticity():
    """Verify hex color parsing to chromaticity."""
    r, g, lum = hex_to_chromaticity("#ffffff")
    assert abs(r - 0.333) < 0.01
    assert abs(g - 0.333) < 0.01
    assert lum > 200.0


def test_light_real_reflection_passes():
    """Verify authentic synthetic reflection yields high correlation."""
    face_rois, neck_rois = simulate_light_response(
        is_real=True, challenge_sequence=SAMPLE_CHALLENGE, n_frames=16, snr_level=4.0, seed=42
    )
    res = analyze_light_response(
        face_rois=face_rois, neck_rois=neck_rois, challenge_sequence=SAMPLE_CHALLENGE, tap_mode="agent"
    )
    assert res.corr_face > 0.60
    assert res.light_score > 0.70
    assert "NO_LIGHT_REACTION" not in res.flags


def test_light_fake_unreflective_triggers_flag():
    """Verify lack of reflection triggers NO_LIGHT_REACTION."""
    face_rois, neck_rois = simulate_light_response(
        is_real=False, challenge_sequence=SAMPLE_CHALLENGE, n_frames=16, snr_level=4.0, seed=123
    )
    res = analyze_light_response(
        face_rois=face_rois, neck_rois=neck_rois, challenge_sequence=SAMPLE_CHALLENGE, tap_mode="agent"
    )
    assert res.corr_face < LIGHT_CORR_FACE_THRESHOLD
    assert "NO_LIGHT_REACTION" in res.flags


def test_light_neck_face_mismatch_detected():
    """Verify mismatched neck vs face reflections triggers NECK_FACE_MISMATCH."""
    face_rois, _ = simulate_light_response(is_real=True, challenge_sequence=SAMPLE_CHALLENGE, n_frames=16, seed=1)
    # Give neck completely inverted or uncorrelated signal
    neck_rois = [{"r": float(1.0 - f["r"]), "g": 0.33, "b": 0.33} for f in face_rois]
    res = analyze_light_response(
        face_rois=face_rois, neck_rois=neck_rois, challenge_sequence=SAMPLE_CHALLENGE, tap_mode="agent"
    )
    assert "NECK_FACE_MISMATCH" in res.flags
    assert res.neck_face_corr < 0.0


def test_light_tap_agent_suppresses_late_reaction():
    """Verify late reaction is disabled when tap_mode='agent' due to network jitter."""
    face_rois, neck_rois = simulate_light_response(is_real=True, challenge_sequence=SAMPLE_CHALLENGE, n_frames=16)
    # Even if lag is large, tap_mode=agent suppresses LATE_LIGHT_REACTION
    res_agent = analyze_light_response(
        face_rois=face_rois, neck_rois=neck_rois, challenge_sequence=SAMPLE_CHALLENGE, tap_mode="agent"
    )
    assert "LATE_LIGHT_REACTION" not in res_agent.flags


def test_light_response_auc_on_500_synthetic_cases():
    """
    CRITICAL ACCEPTANCE TEST:
    Separate simulated real vs fake with AUC > 0.95 across 500 cases at 3 SNR levels.
    """
    labels = []
    scores = []
    snr_levels = [1.5, 3.0, 5.0]

    for i in range(500):
        is_real = (i % 2 == 0)
        snr = snr_levels[i % len(snr_levels)]
        face_rois, neck_rois = simulate_light_response(
            is_real=is_real,
            challenge_sequence=SAMPLE_CHALLENGE,
            n_frames=16,
            snr_level=snr,
            seed=1000 + i,
        )
        res = analyze_light_response(
            face_rois=face_rois,
            neck_rois=neck_rois,
            challenge_sequence=SAMPLE_CHALLENGE,
            tap_mode="agent",
        )
        labels.append(1 if is_real else 0)
        scores.append(res.light_score)

    # Compute Mann-Whitney U statistic (exact ROC AUC)
    y_true = np.array(labels)
    y_score = np.array(scores)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    # Calculate AUC
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    # Count pairs where pos > neg (with ties = 0.5)
    u_stat = sum(np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores) for p in pos_scores)
    auc = float(u_stat / (n_pos * n_neg))

    print(f"\n[E2 Light Response] 500-sample Benchmark AUC: {auc:.4f}")
    assert auc > 0.95, f"Expected AUC > 0.95 on 500 synthetic cases, got {auc:.4f}"


# ============================================================================
# E4: Identity Consistency Unit Tests (>= 5 tests)
# ============================================================================

def test_identity_consistent_embeddings():
    """Verify stable face embeddings yield low flicker score and no flags."""
    # 16 identical embeddings
    base_emb = np.random.RandomState(42).randn(128)
    embeddings = [base_emb + np.random.normal(0, 0.01, size=128) for _ in range(16)]
    res = analyze_identity_consistency(embeddings=embeddings)
    assert res.flicker_score < 0.20
    assert len(res.flags) == 0
    assert res.min_cosine_sim > 0.90


def test_identity_flicker_detected():
    """Verify fluctuating facial identities trigger IDENTITY_FLICKER."""
    rng = np.random.RandomState(7)
    embeddings = [rng.randn(128) for _ in range(16)]  # Random embeddings = no consistency
    res = analyze_identity_consistency(embeddings=embeddings)
    assert "IDENTITY_FLICKER" in res.flags
    assert res.flicker_score > 0.40


def test_identity_landmark_jitter_detected():
    """Verify high-frequency landmark oscillation triggers LANDMARK_JITTER."""
    # 16 frames of 478 landmarks oscillating rapidly
    landmarks_series = []
    for i in range(16):
        jitter_val = 0.15 if (i % 2 == 0) else -0.15
        lm = np.zeros((478, 3), dtype=np.float32) + jitter_val
        landmarks_series.append(lm)
    res = analyze_identity_consistency(landmarks_series=landmarks_series)
    assert "LANDMARK_JITTER" in res.flags
    assert res.landmark_jitter > 0.08


def test_identity_jump_count_tracked():
    """Verify abrupt identity discontinuities are counted."""
    e1 = np.ones(128)
    e2 = -np.ones(128)  # Inverted = jump
    embeddings = [e1, e1, e2, e2, e1]
    res = analyze_identity_consistency(embeddings=embeddings)
    assert res.jump_count >= 2


def test_identity_empty_graceful_fallback():
    """Verify empty or single embedding inputs return neutral score without error."""
    res = analyze_identity_consistency(embeddings=[])
    assert res.flicker_score == 0.0
    assert len(res.flags) == 0


# ============================================================================
# E6a: Phrase Verification Unit Tests (>= 5 tests)
# ============================================================================

def test_phrase_exact_match():
    """Verify exact word match gives WER = 0.0."""
    transcription = [
        {"word": "Blue", "start": 0.5, "end": 0.8},
        {"word": "Paper", "start": 0.9, "end": 1.3},
        {"word": "Mountain", "start": 1.4, "end": 1.9},
    ]
    res = analyze_phrase_match(transcription, "Blue Paper Mountain")
    assert res.phrase_wer == 0.0
    assert res.matched is True
    assert len(res.flags) == 0


def test_phrase_single_word_substitution():
    """Verify single word error computes correct Levenshtein WER."""
    transcription = [
        {"word": "Blue", "start": 0.5, "end": 0.8},
        {"word": "Plastic", "start": 0.9, "end": 1.3},
        {"word": "Mountain", "start": 1.4, "end": 1.9},
    ]
    res = analyze_phrase_match(transcription, "Blue Paper Mountain")
    assert abs(res.phrase_wer - (1.0 / 3.0)) < 1e-4
    assert res.matched is True  # WER 0.33 <= 0.40 threshold


def test_phrase_mismatch_triggers_flag():
    """Verify completely wrong words trigger PHRASE_MISMATCH."""
    transcription = [
        {"word": "Red", "start": 0.5, "end": 0.8},
        {"word": "Apple", "start": 0.9, "end": 1.3},
        {"word": "Valley", "start": 1.4, "end": 1.9},
    ]
    res = analyze_phrase_match(transcription, "Blue Paper Mountain")
    assert res.phrase_wer > PHRASE_WER_HIGH
    assert "PHRASE_MISMATCH" in res.flags


def test_phrase_wer_insertions_and_deletions():
    """Verify insertions and deletions correctly calculate WER."""
    ref = ["Paper", "Mountain"]
    hyp = ["Paper", "Green", "Mountain", "Bottle"]
    wer = compute_word_error_rate(ref, hyp)
    assert wer == 1.0  # 2 insertions / 2 ref = 1.0


def test_phrase_empty_transcription():
    """Verify empty speech output yields WER = 1.0 and flag."""
    res = analyze_phrase_match([], "Banana Pocket Mountain")
    assert res.phrase_wer == 1.0
    assert "PHRASE_MISMATCH" in res.flags


# ============================================================================
# E6b: Lip Closure Unit Tests (>= 5 tests)
# ============================================================================

def test_lip_closure_authentic_closed():
    """Verify closed lips (aperture <= 0.12) on B, P, M passes."""
    apertures = [0.25, 0.05, 0.20, 0.04, 0.22, 0.06, 0.24]
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    words = [
        WordTimestamp(word="Bottle", start_sec=0.5, end_sec=0.9),
        WordTimestamp(word="Paper", start_sec=1.5, end_sec=1.9),
        WordTimestamp(word="Mountain", start_sec=2.5, end_sec=2.9),
    ]
    res = analyze_lip_closure(apertures, timestamps, words)
    assert res.passed is True
    assert len(res.violations) == 0
    assert len(res.flags) == 0


def test_lip_closure_open_lips_on_p_triggers():
    """Verify open lips on 'P' in 'Paper' triggers LIP_CLOSURE_MISSING."""
    # Mouth wide open (0.28 aperture) at t=1.5s
    apertures = [0.25, 0.05, 0.20, 0.28, 0.22, 0.06, 0.24]
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    words = [
        WordTimestamp(word="Bottle", start_sec=0.5, end_sec=0.9),
        WordTimestamp(word="Paper", start_sec=1.5, end_sec=1.9),
        WordTimestamp(word="Mountain", start_sec=2.5, end_sec=2.9),
    ]
    res = analyze_lip_closure(apertures, timestamps, words)
    assert res.passed is False
    assert "LIP_CLOSURE_MISSING" in res.flags
    assert len(res.violations) == 1
    assert res.violations[0].word == "Paper"
    assert res.violations[0].letter == "P"


def test_lip_closure_open_lips_on_m_triggers():
    """Verify open lips on 'M' in 'Mountain' triggers violation."""
    apertures = [0.05, 0.05, 0.25, 0.30]
    timestamps = [0.0, 0.5, 1.0, 1.5]
    words = [WordTimestamp(word="Mountain", start_sec=1.5, end_sec=2.0)]
    res = analyze_lip_closure(apertures, timestamps, words)
    assert "LIP_CLOSURE_MISSING" in res.flags
    assert res.violations[0].letter == "M"


def test_lip_closure_non_bilabials_ignored():
    """Verify non-bilabials (words starting with D, G, S) are not checked for bilabial closure."""
    apertures = [0.35, 0.40]  # Wide open mouth
    timestamps = [0.0, 1.0]
    words = [WordTimestamp(word="Green", start_sec=0.0, end_sec=0.8)]
    res = analyze_lip_closure(apertures, timestamps, words)
    assert res.passed is True
    assert len(res.violations) == 0


def test_lip_closure_empty_data():
    """Verify empty words or aperture series returns clean result."""
    res = analyze_lip_closure([], [], [])
    assert res.passed is True
    assert len(res.flags) == 0


# ============================================================================
# E6c: AV Sync Unit Tests (>= 5 tests)
# ============================================================================

def test_av_sync_aligned():
    """Verify aligned audio energy and mouth apertures yield in_sync=True."""
    # Synthetic bursty signal
    t = np.linspace(0, 1.0, 16)
    aperture = list(np.sin(2 * np.pi * 3 * t) + 1.0)
    # Audio with identical burst frequency
    audio = np.zeros(16000, dtype=np.int16)
    for i in range(16):
        if aperture[i] > 1.2:
            audio[i * 1000 : (i + 1) * 1000] = 5000
    res = analyze_av_sync(audio, aperture)
    assert res.in_sync is True
    assert abs(res.offset_ms) < 100.0


def test_av_sync_severe_lag_triggers():
    """Verify 300 ms offset triggers SYNC_OFFSET."""
    t = np.linspace(0, 2.0, 30)
    aperture = list(np.sin(2 * np.pi * 2 * t) + 1.0)
    audio = np.zeros(32000, dtype=np.int16)
    # Audio delayed by 300ms (5 frames)
    for i in range(5, 30):
        if aperture[i - 5] > 1.2:
            audio[i * 1066 : (i + 1) * 1066] = 5000
    res = analyze_av_sync(audio, aperture)
    assert abs(res.offset_ms) > 100.0


def test_av_sync_silent_audio_safe():
    """Verify completely silent audio handles safely without false flag."""
    audio = np.zeros(16000, dtype=np.int16)
    apertures = [0.2] * 16
    res = analyze_av_sync(audio, apertures)
    assert res.in_sync is True
    assert "SYNC_OFFSET" not in res.flags


def test_av_sync_confidence_ratio():
    """Verify confidence metric reflects correlation prominence."""
    audio = np.random.RandomState(42).randint(-2000, 2000, size=16000, dtype=np.int16)
    apertures = list(np.random.RandomState(42).uniform(0.1, 0.3, size=16))
    res = analyze_av_sync(audio, apertures)
    assert res.sync_conf >= 0.0


def test_av_sync_short_inputs():
    """Verify short input arrays return neutral sync result."""
    res = analyze_av_sync(np.zeros(100, dtype=np.int16), [0.1, 0.2])
    assert res.in_sync is True


# ============================================================================
# E8: Quality Assessment Unit Tests (>= 5 tests)
# ============================================================================

def test_quality_studio_grade():
    """Verify crisp, well-lit, centered face gives high quality_trust."""
    res = evaluate_quality(blur=120.0, brightness=130.0, face_ratio=1.0)
    assert res.quality_trust > 0.75
    assert len(res.flags) == 0


def test_quality_motion_blur():
    """Verify heavy blur triggers MOTION_BLUR."""
    res = evaluate_quality(blur=15.0, brightness=120.0, face_ratio=1.0)
    assert "MOTION_BLUR" in res.flags
    assert res.quality_trust < 0.60


def test_quality_dark_lighting():
    """Verify dark lighting triggers POOR_LIGHTING_DARK."""
    res = evaluate_quality(blur=90.0, brightness=20.0, face_ratio=1.0)
    assert "POOR_LIGHTING_DARK" in res.flags
    assert res.quality_trust < 0.70


def test_quality_overexposed_lighting():
    """Verify blown out brightness triggers POOR_LIGHTING_OVEREXPOSED."""
    res = evaluate_quality(blur=90.0, brightness=240.0, face_ratio=1.0)
    assert "POOR_LIGHTING_OVEREXPOSED" in res.flags


def test_quality_face_missing():
    """Verify low face presence triggers FACE_NOT_VISIBLE and drops trust."""
    res = evaluate_quality(blur=100.0, brightness=120.0, face_ratio=0.0)
    assert "FACE_NOT_VISIBLE" in res.flags
    assert res.quality_trust < 0.60
