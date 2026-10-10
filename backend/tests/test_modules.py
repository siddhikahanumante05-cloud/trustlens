"""Unit tests for TrustLens multi-modal expert modules and fusion layer."""
import pytest
import numpy as np

from app.modules.base import ModuleInputs, ModuleResult
from app.modules.quality import QualityGateModule
from app.modules.m1_source import SourceLivenessModule
from app.modules.m2_audio import AudioSpoofModule
from app.modules.m3_video import VideoAppearanceModule
from app.modules.m4_geometry import FaceGeometryModule
from app.modules.m5_speech_lips import SpeechLipModule
from app.runtime import ModelRuntime
from app.fusion import FusionEngine


def make_dummy_inputs(
    with_frames: bool = True,
    with_audio: bool = True,
    with_crops: bool = True,
    with_landmarks: bool = True,
) -> ModuleInputs:
    """Builds test inputs for module evaluation."""
    n_frames = 15 if with_frames else 0
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(n_frames)]
    timestamps = [i * (1.0 / 15.0) for i in range(n_frames)]

    audio = np.zeros(16000, dtype=np.int16) if with_audio else None
    crops = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(n_frames)] if with_crops else []
    landmarks = [np.zeros((478, 3), dtype=np.float32) for _ in range(n_frames)] if with_landmarks else []
    mouth_apertures = [0.05 for _ in range(n_frames)] if with_landmarks else []

    skin_rois = {
        "face": [{"r": 0.4, "g": 0.3, "b": 0.3, "y": 120.0} for _ in range(n_frames)],
        "neck": [{"r": 0.4, "g": 0.3, "b": 0.3, "y": 115.0} for _ in range(n_frames)],
        "background": [{"r": 0.2, "g": 0.2, "b": 0.2, "y": 50.0} for _ in range(n_frames)],
    }

    return ModuleInputs(
        frames=frames,
        timestamps=timestamps,
        audio_pcm=audio,
        t_sec=1.0,
        face_crops=crops,
        landmarks_series=landmarks,
        mouth_apertures=mouth_apertures,
        skin_rois=skin_rois,
        hand_occlusion_score=0.0,
        quality={"blur": 95.0, "brightness": 120.0, "face_ratio": 0.40, "quality_trust": 0.90},
        metadata={"camera_label": "FaceTime HD Camera", "frame_intervals_ms": [66.6] * 15},
        challenges=[],
        tap_mode="agent",
    )


def test_quality_gate_module():
    inputs = make_dummy_inputs()
    q_mod = QualityGateModule()
    res = q_mod.run(inputs)
    assert res.module_id == "Q_quality"
    assert res.status == "ok"
    assert "quality_trust" in res.features
    assert 0.0 <= res.features["quality_trust"] <= 1.0


def test_m1_source_module():
    inputs = make_dummy_inputs()
    m1 = SourceLivenessModule()
    res = m1.run(inputs)
    assert res.module_id == "M1_source"
    assert res.status == "ok"
    assert "source_risk" in res.features
    assert 0.0 <= res.risk <= 1.0


def test_m2_audio_skip_when_missing():
    inputs = make_dummy_inputs(with_audio=False)
    m2 = AudioSpoofModule()
    res = m2.run(inputs)
    assert res.module_id == "M2_audio"
    assert res.status == "skipped"
    assert res.confidence == 0.0
    assert len(res.findings) == 0


def test_m3_video_module():
    inputs = make_dummy_inputs()
    runtime = ModelRuntime()
    m3 = VideoAppearanceModule(runtime=runtime)
    res = m3.run(inputs)
    assert res.module_id == "M3_video"
    assert res.status == "ok"
    assert "edge_logit" in res.features
    assert "clip_logit" in res.features


def test_m3_video_skip_when_no_crops():
    inputs = make_dummy_inputs(with_crops=False)
    runtime = ModelRuntime()
    m3 = VideoAppearanceModule(runtime=runtime)
    res = m3.run(inputs)
    assert res.module_id == "M3_video"
    assert res.status == "skipped"
    assert res.confidence == 0.0


def test_m4_geometry_module():
    inputs = make_dummy_inputs()
    m4 = FaceGeometryModule()
    res = m4.run(inputs)
    assert res.module_id == "M4_geometry"
    assert res.status == "ok"
    assert "landmark_jitter" in res.features
    assert "static_face_risk" in res.features


def test_m4_geometry_static_photo_attack_detected():
    """Verify static photo presentation attack is flagged across consecutive windows."""
    inputs = make_dummy_inputs()
    # Populate with static face photo crops
    static_crop = np.full((160, 160, 3), 128, dtype=np.uint8)
    static_crop[40:120, 40:120] = [180, 160, 140]
    inputs.face_crops = [static_crop.copy() for _ in range(16)]

    m4 = FaceGeometryModule()
    # Window 1: warning / low dynamics
    res1 = m4.run(inputs)
    assert res1.features["static_face_risk"] >= 0.35

    # Window 2: confirmed static 2D photo attack
    res2 = m4.run(inputs)
    assert res2.features["static_face_risk"] >= 0.70
    assert any("Static 2D Photo Attack" in f.title for f in res2.findings)


def test_m4_geometry_skip_when_no_landmarks():
    inputs = make_dummy_inputs(with_landmarks=False)
    m4 = FaceGeometryModule()
    res = m4.run(inputs)
    assert res.module_id == "M4_geometry"
    assert res.status == "skipped"
    assert res.confidence == 0.0


def test_m5_speech_lips_module():
    inputs = make_dummy_inputs()
    m5 = SpeechLipModule()
    res = m5.run(inputs)
    assert res.module_id == "M5_speech_lips"
    assert res.status == "ok"
    assert "phrase_wer" in res.features
    assert "sync_conf" in res.features


def test_m5_speech_lips_skip_when_audio_missing():
    inputs = make_dummy_inputs(with_audio=False)
    m5 = SpeechLipModule()
    res = m5.run(inputs)
    assert res.module_id == "M5_speech_lips"
    assert res.status == "skipped"
    assert res.confidence == 0.0


def test_fusion_with_missing_modalities():
    engine = FusionEngine()
    
    # Simulate Audio (M2) and Lips (M5) skipped due to missing audio
    q_res = ModuleResult("Q_quality", "ok", 0.0, 1.0, {"quality_trust": 0.85})
    m1_res = ModuleResult("M1_source", "ok", 0.05, 0.9, {"source_risk": 0.05, "light_corr_face": 0.55})
    m2_res = ModuleResult("M2_audio", "skipped", 0.0, 0.0, {})
    m3_res = ModuleResult("M3_video", "ok", 0.10, 0.8, {"edge_logit": -1.2, "clip_logit": -0.8})
    m4_res = ModuleResult("M4_geometry", "ok", 0.05, 0.8, {"landmark_jitter": 0.02})
    m5_res = ModuleResult("M5_speech_lips", "skipped", 0.0, 0.0, {})

    fusion_res = engine.fuse_modules([q_res, m1_res, m2_res, m3_res, m4_res, m5_res])
    
    # Missing audio must NOT crash and must NOT be treated as fake
    assert fusion_res.category in ("Authentic", "Suspicious")
    assert fusion_res.smoothed_probability < 0.60
    assert "spoof_max" in fusion_res.missing_features


def test_m1_source_mobile_screen_replay_detection():
    """Verify SourceLivenessModule flags mobile screen YouTube replay via specular reflection."""
    inputs = make_dummy_inputs()
    # Inject specular glass reflection patch across crops
    for crop in inputs.face_crops:
        crop[50:70, 50:70] = [252, 252, 254]
        crop[::2, :, 0] = np.clip(crop[::2, :, 0].astype(int) + 30, 0, 255)
        crop[1::2, :, 2] = np.clip(crop[1::2, :, 2].astype(int) + 30, 0, 255)

    # Move landmarks to induce reflection decoupling
    for i, lm in enumerate(inputs.landmarks_series):
        lm[:, 0] = 0.5 + (i * 0.02)

    m1 = SourceLivenessModule()
    res = m1.run(inputs)
    assert res.module_id == "M1_source"
    assert res.features["replay_score"] >= 0.50
    assert res.features["screen_glare"] >= 0.40
    assert res.risk >= 0.50
    replay_findings = [f for f in res.findings if "Mobile Screen" in f.title or "Screen Glass" in f.title]
    assert len(replay_findings) > 0


def test_fusion_hard_gate_screen_replay():
    """Verify FusionEngine hard-gates to 'Likely deepfake' on mobile screen replay."""
    engine = FusionEngine()
    q_res = ModuleResult("Q_quality", "ok", 0.0, 1.0, {"quality_trust": 0.85})
    m1_res = ModuleResult(
        "M1_source",
        "ok",
        0.80,
        0.95,
        {
            "source_risk": 0.10,
            "replay_score": 0.75,
            "screen_glare": 0.70,
            "light_corr_face": 0.50,
        },
    )
    m2_res = ModuleResult("M2_audio", "skipped", 0.0, 0.0, {})
    m3_res = ModuleResult("M3_video", "ok", 0.10, 0.8, {"edge_logit": -1.0, "clip_logit": -0.8})
    m4_res = ModuleResult("M4_geometry", "ok", 0.05, 0.8, {"landmark_jitter": 0.02})
    m5_res = ModuleResult("M5_speech_lips", "skipped", 0.0, 0.0, {})

    fusion_res = engine.fuse_modules([q_res, m1_res, m2_res, m3_res, m4_res, m5_res])
    assert fusion_res.category == "Likely deepfake"
    assert fusion_res.in_deepfake_state is True
    assert fusion_res.check_states["source"] == "bad"
    assert "Mobile screen replay detected" in (fusion_res.action_recommended or "")

