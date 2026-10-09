"""Tests for Milestone 5: FusionEngine, ExplainEngine, NarratorEngine, CaseStore, and Runtime."""
import os
import shutil
import tempfile
import pytest
import numpy as np

from app.fusion import FusionEngine, WindowFeatures
from app.explain import ExplainEngine
from app.narrator import NarratorEngine
from app.store import CaseStore
from app.runtime import ModelRuntime
from app.schemas import FindingEvent


def test_fusion_feature_imputation_and_monotonicity():
    engine = FusionEngine()

    # 1. Neutral features should yield low / authentic risk
    neutral_feat = WindowFeatures()
    res_neutral = engine.process_window(neutral_feat)
    assert res_neutral.category == "Authentic"
    assert res_neutral.smoothed_probability < 0.40

    # 2. Add deepfake indicators (high source risk, severe lip closure error, bad light correlation)
    engine.reset()
    fake_feat = WindowFeatures(
        source_risk=0.85,
        light_corr_face=0.05,
        edge_logit=1.8,
        bilabial_aperture_err=0.45,
    )
    res_fake = engine.process_window(fake_feat)
    assert res_fake.raw_logit > res_neutral.raw_logit
    assert res_fake.smoothed_probability > res_neutral.smoothed_probability


def test_fusion_hysteresis_state_machine():
    engine = FusionEngine(ema_alpha=1.0)  # instantaneous smoothing for state machine test

    # Single window with high risk should NOT enter Likely deepfake (needs 2 consecutive)
    fake_feat = WindowFeatures(
        source_risk=0.90,
        light_corr_face=0.02,
        edge_logit=2.5,
        bilabial_aperture_err=0.50,
    )
    res1 = engine.process_window(fake_feat)
    assert res1.smoothed_probability >= 0.70
    assert not res1.in_deepfake_state  # 1st window: not yet in deepfake state

    # Second window >= 0.70 enters Likely deepfake
    res2 = engine.process_window(fake_feat)
    assert res2.in_deepfake_state
    assert res2.category == "Likely deepfake"
    assert res2.action_recommended is not None  # Action emitted on entry

    # Next window drops to neutral band (0.65) - should STAY in Likely deepfake due to hysteresis
    mid_feat = WindowFeatures(
        source_risk=0.50,
        edge_logit=0.9,
    )
    res3 = engine.process_window(mid_feat)
    assert res3.in_deepfake_state
    assert res3.category == "Likely deepfake"

    # Needs 3 consecutive authentic windows (< 0.60) to exit
    auth_feat = WindowFeatures(source_risk=0.0, light_corr_face=0.85, edge_logit=-1.5)
    r_a1 = engine.process_window(auth_feat)
    assert r_a1.in_deepfake_state  # 1st low window: still locked
    r_a2 = engine.process_window(auth_feat)
    assert r_a2.in_deepfake_state  # 2nd low window: still locked
    r_a3 = engine.process_window(auth_feat)
    assert not r_a3.in_deepfake_state  # 3rd low window: unlocked!


def test_explain_deduplication_and_escalation():
    explain = ExplainEngine()

    # Window 1: medium severity source finding
    f1 = explain.generate_findings({"source_risk": 0.45}, t_sec=1.0)
    assert len(f1) == 1
    assert f1[0].id == "source"
    assert f1[0].severity == "medium"

    # Window 2: same feature value -> should be deduplicated (suppressed)
    f2 = explain.generate_findings({"source_risk": 0.45}, t_sec=2.0)
    assert len(f2) == 0

    # Window 3: escalates to high severity -> should emit
    f3 = explain.generate_findings({"source_risk": 0.85}, t_sec=3.0)
    assert len(f3) == 1
    assert f3[0].severity == "high"


def test_explain_heatmap_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        explain = ExplainEngine(cases_dir=tmp_dir)
        dummy_crops = [np.ones((160, 160, 3), dtype=np.uint8) * 120] * 16

        # Normal logit should not create heatmap
        ev_none = explain.generate_heatmap_evidence("test_sess", 1, dummy_crops, edge_logit=0.10)
        assert ev_none is None

        # High edge logit creates heatmap JPEG
        ev = explain.generate_heatmap_evidence("test_sess", 2, dummy_crops, edge_logit=1.20)
        assert ev is not None
        assert ev.kind == "heatmap"
        assert os.path.exists(os.path.join(tmp_dir, "test_sess", "evidence", "heatmap_w002.jpg"))


def test_narrator_summary_and_validation():
    narrator = NarratorEngine()
    narrator.mode = "template"
    findings = [
        FindingEvent(
            type="finding",
            id="face",
            severity="high",
            title="Deepfake Boundary Artifacts",
            detail="Blending seams detected",
            t=2.0,
        )
    ]

    summary, ai_gen = narrator.generate_summary(
        risk_score=0.82,
        category="Likely deepfake",
        findings=findings,
    )
    assert "82%" in summary
    assert "Likely deepfake" in summary
    assert "Deepfake Boundary Artifacts" in summary
    assert not ai_gen


def test_case_store_save_load_and_cleanup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = CaseStore(base_dir=tmp_dir)

        path = store.save_case(
            session_id="test_sess_01",
            consent_status="consented",
            stubs_in_use=True,
            risk_score=0.75,
            category="Likely deepfake",
            findings=[{"id": "face", "severity": "high"}],
            summary="Test summary",
            ai_generated_summary=False,
            windows=[{"window_id": 1, "risk": 0.75}],
        )
        assert os.path.exists(path)

        loaded = store.load_case("test_sess_01")
        assert loaded is not None
        assert loaded["risk_score"] == 0.75
        assert loaded["category"] == "Likely deepfake"
        assert loaded["consent_status"] == "consented"

        # Test TTL cleanup (setting ttl_hours to -1 evicts all cases)
        evicted = store.cleanup_expired_cases(ttl_hours=-1)
        assert evicted == 1
        assert store.load_case("test_sess_01") is None


def test_model_runtime_execution():
    runtime = ModelRuntime()
    dummy_crops = [np.ones((160, 160, 3), dtype=np.uint8) * 100] * 16
    dummy_audio = np.zeros(48000, dtype=np.int16)

    out = runtime.run_inference(dummy_crops, dummy_audio, quality_trust=0.90)
    assert isinstance(out.edge_logit, float)
    assert isinstance(out.spoof_max, float)
    runtime.close()
