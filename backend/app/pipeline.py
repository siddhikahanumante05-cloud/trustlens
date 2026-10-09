"""End-to-end Analysis Pipeline for TrustLens.
Unifies decoder, windower, preprocessor, signal extractors, model runtime,
fusion engine, explainability engine, and case persistence.
Shared by live WebSocket calls and POST /analyze/file.
"""
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from app.config import settings
from app.decoder import ChunkDecoder
from app.windower import Windower, AnalysisWindow
from app.preprocess import Preprocessor, PreprocessedWindow
from app.runtime import ModelRuntime, ModelOutputs
from app.fusion import FusionEngine, WindowFeatures, FusionResult
from app.explain import ExplainEngine
from app.narrator import NarratorEngine
from app.store import case_store
from app.schemas import (
    CheckEvent,
    RiskEvent,
    ActionEvent,
    SummaryEvent,
    FindingEvent,
    EvidenceEvent,
)
from app.signals.provenance import analyze_provenance
from app.signals.light import analyze_light_response
from app.signals.identity import analyze_identity_consistency
from app.signals.phrase import analyze_phrase_match
from app.signals.lip_closure import analyze_lip_closure
from app.signals.sync import analyze_av_sync
from app.signals.quality import evaluate_quality

logger = logging.getLogger("trustlens.pipeline")


class LivePipeline:
    """Manages an active real-time analysis session."""

    def __init__(self, session_id: str, tap_mode: str = "agent"):
        self.session_id = session_id
        self.tap_mode = tap_mode
        self.t_start = time.time()

        self.preprocessor = Preprocessor()
        self.runtime = ModelRuntime()
        self.fusion = FusionEngine()
        self.explain = ExplainEngine()
        self.narrator = NarratorEngine()

        self.meta_info: Dict[str, Any] = {}
        self.camera_label: Optional[str] = None
        self.frame_intervals_ms: Optional[List[float]] = None

        self.all_findings: List[FindingEvent] = []
        self.window_history: List[Dict[str, Any]] = []
        self.latest_result: Optional[FusionResult] = None
        self.latest_risk: float = 0.0
        self.latest_category: str = "Authentic"
        self.window_count: int = 0

    def update_metadata(self, meta: Dict[str, Any]):
        """Store camera label and timing jitter metadata from caller."""
        self.meta_info.update(meta)
        if "camera_label" in meta:
            self.camera_label = meta["camera_label"]
        if "frame_intervals_ms" in meta:
            self.frame_intervals_ms = meta["frame_intervals_ms"]

    def process_window(self, win: AnalysisWindow) -> Tuple[
        Dict[str, str],            # Check states
        float,                     # Smoothed risk score
        Optional[str],             # Action recommendation (if any)
        List[FindingEvent],        # Newly escalated findings
        Optional[EvidenceEvent],   # Heatmap evidence (if generated)
    ]:
        """Runs the full analysis stack on a single temporal window."""
        self.window_count += 1
        t_sec = float(win.timestamps[-1]) if win.timestamps else float(self.window_count)

        # 1. Preprocessing (face crops, landmarks, mouth aperture, ROIs)
        prep: PreprocessedWindow = self.preprocessor.process_window(
            frames=win.frames,
            timestamps=win.timestamps,
            audio=win.audio,
            challenges=win.challenges,
        )

        # 2. Extract Signals:
        # E1: Provenance
        prov_res = analyze_provenance(
            camera_label=self.camera_label,
            frame_intervals_ms=self.frame_intervals_ms,
        )

        # E2: Active Light Response
        challenge_seq = None
        challenge_t0 = None
        if prep.challenges:
            for ch in prep.challenges:
                if ch.get("kind") == "light":
                    challenge_seq = ch.get("sequence")
                    challenge_t0 = ch.get("t0")
                    break

        light_res = analyze_light_response(
            face_rois=prep.skin_rois.get("face", []),
            neck_rois=prep.skin_rois.get("neck", []),
            bg_rois=prep.skin_rois.get("background", []),
            timestamps=prep.timestamps,
            challenge_sequence=challenge_seq,
            challenge_t0=challenge_t0,
            tap_mode=self.tap_mode,
        )

        # E4: Identity Consistency & Jitter
        # Use mean face crop RGB as quick spatial embedding proxy if full model is in stub mode
        face_embeddings = [np.mean(crop, axis=(0, 1)) for crop in prep.face_crops] if prep.face_crops else []
        id_res = analyze_identity_consistency(
            embeddings=face_embeddings,
            landmarks_series=prep.landmarks_series,
        )

        # E6: Phrase & Bilabial Lip Closure
        expected_phrase = ""
        if prep.challenges:
            for ch in prep.challenges:
                if ch.get("kind") == "phrase":
                    expected_phrase = ch.get("expected", "")
                    break

        # Transcribed words stub or Whisper if available
        phrase_res = analyze_phrase_match(
            transcribed_words=[],
            expected_phrase=expected_phrase,
        )

        lip_res = analyze_lip_closure(
            aperture_series=prep.mouth_apertures,
            frame_timestamps=prep.timestamps,
            word_timestamps=phrase_res.word_timestamps,
        )

        # E6c: AV Sync Fallback
        sync_res = analyze_av_sync(
            audio_pcm=prep.audio,
            mouth_apertures=prep.mouth_apertures,
            sample_rate=16000,
            fps=15.0,
        )

        # E8: Quality Trust
        qual_res = evaluate_quality(
            blur=prep.quality["blur"],
            brightness=prep.quality["brightness"],
            face_ratio=prep.quality["face_ratio"],
        )

        # 3. Model Inference (Fast-path EdgeNet + Slow-path CLIP/WavLM)
        models_out: ModelOutputs = self.runtime.run_inference(
            face_crops=prep.face_crops,
            audio_pcm=prep.audio,
            quality_trust=qual_res.quality_trust,
        )

        # 4. Feature Vector Assembly
        features = WindowFeatures(
            source_risk=prov_res.source_risk,
            light_corr_face=light_res.corr_face,
            light_lag_ms=light_res.lag_ms,
            light_neck_face_ratio=light_res.neck_face_ratio,
            light_neck_face_corr=light_res.neck_face_corr,
            light_snr=light_res.snr,
            edge_logit=models_out.edge_logit,
            clip_logit=models_out.clip_logit,
            model_disagreement=models_out.model_disagreement,
            ood_score=models_out.ood_score,
            identity_flicker=1.0 - id_res.mean_cosine_sim,
            landmark_jitter=id_res.landmark_jitter,
            occlusion_artifact=prep.hand_occlusion_score,
            spoof_max=models_out.spoof_max,
            spoof_mean=models_out.spoof_mean,
            sync_conf=sync_res.sync_conf,
            sync_offset_ms=sync_res.offset_ms,
            phrase_wer=phrase_res.phrase_wer,
            bilabial_aperture_err=lip_res.bilabial_aperture_err,
            quality_trust=qual_res.quality_trust,
        )

        # 5. Multimodal Fusion
        fusion_res: FusionResult = self.fusion.process_window(features)
        self.latest_result = fusion_res
        self.latest_risk = fusion_res.smoothed_probability
        self.latest_category = fusion_res.category

        # 6. Explainability Findings
        new_findings = self.explain.generate_findings(
            features=fusion_res.feature_dict,
            t_sec=t_sec,
        )
        self.all_findings.extend(new_findings)

        # 7. Visual Heatmap Evidence (if suspicious)
        evidence = self.explain.generate_heatmap_evidence(
            session_id=self.session_id,
            window_idx=self.window_count,
            face_crops=prep.face_crops,
            edge_logit=models_out.edge_logit,
        )

        # Record window stats
        self.window_history.append({
            "window_id": self.window_count,
            "t_sec": round(t_sec, 2),
            "risk": round(fusion_res.smoothed_probability, 4),
            "category": fusion_res.category,
            "quality_trust": round(qual_res.quality_trust, 3),
            "checks": fusion_res.check_states,
        })

        return (
            fusion_res.check_states,
            fusion_res.smoothed_probability,
            fusion_res.action_recommended,
            new_findings,
            evidence,
        )

    def finalize(self, consent_status: str) -> Tuple[str, bool]:
        """Creates session summary and writes case record to disk."""
        summary_text, ai_generated = self.narrator.generate_summary(
            risk_score=self.latest_risk,
            category=self.latest_category,
            findings=self.all_findings,
        )

        findings_dicts = [f.model_dump() for f in self.all_findings]
        case_store.save_case(
            session_id=self.session_id,
            consent_status=consent_status,
            stubs_in_use=self.runtime.stubs_in_use,
            risk_score=self.latest_risk,
            category=self.latest_category,
            findings=findings_dicts,
            summary=summary_text,
            ai_generated_summary=ai_generated,
            windows=self.window_history,
        )

        return summary_text, ai_generated

    def close(self):
        """Clean up runtime resources."""
        self.runtime.close()
