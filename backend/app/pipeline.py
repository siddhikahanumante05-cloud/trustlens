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
from app.modules import (
    ModuleInputs,
    ModuleResult,
    QualityGateModule,
    SourceLivenessModule,
    AudioSpoofModule,
    VideoAppearanceModule,
    FaceGeometryModule,
    SpeechLipModule,
)

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

        # Expert Modality Modules
        self.m1_module = SourceLivenessModule()
        self.m2_module = AudioSpoofModule()
        self.m3_module = VideoAppearanceModule(runtime=self.runtime)
        self.m4_module = FaceGeometryModule()
        self.m5_module = SpeechLipModule()

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

        # 2. Execute Quality Gate
        qual_mod = QualityGateModule()
        qual_res = qual_mod.run(
            ModuleInputs(
                frames=win.frames,
                timestamps=win.timestamps,
                audio_pcm=win.audio,
                t_sec=t_sec,
                face_crops=prep.face_crops,
                landmarks_series=prep.landmarks_series,
                mouth_apertures=prep.mouth_apertures,
                skin_rois=prep.skin_rois,
                hand_occlusion_score=prep.hand_occlusion_score,
                quality=prep.quality,
                metadata={
                    "camera_label": self.camera_label,
                    "frame_intervals_ms": self.frame_intervals_ms,
                    **self.meta_info,
                },
                challenges=prep.challenges,
                tap_mode=self.tap_mode,
            )
        )
        prep.quality["quality_trust"] = qual_res.features.get("quality_trust", 0.85)

        # 3. Assemble shared ModuleInputs
        mod_inputs = ModuleInputs(
            frames=win.frames,
            timestamps=win.timestamps,
            audio_pcm=win.audio,
            t_sec=t_sec,
            face_crops=prep.face_crops,
            landmarks_series=prep.landmarks_series,
            mouth_apertures=prep.mouth_apertures,
            skin_rois=prep.skin_rois,
            hand_occlusion_score=prep.hand_occlusion_score,
            quality=prep.quality,
            metadata={
                "camera_label": self.camera_label,
                "frame_intervals_ms": self.frame_intervals_ms,
                **self.meta_info,
            },
            challenges=prep.challenges,
            tap_mode=self.tap_mode,
        )

        # 4. Run Multi-Modal Expert Modules
        m1_res = self.m1_module.run(mod_inputs)
        m2_res = self.m2_module.run(mod_inputs)
        m3_res = self.m3_module.run(mod_inputs)
        m4_res = self.m4_module.run(mod_inputs)
        m5_res = self.m5_module.run(mod_inputs)

        module_results = [qual_res, m1_res, m2_res, m3_res, m4_res, m5_res]

        # 5. Calibrated Multi-Modal Fusion (handles missing modalities & hard gating)
        fusion_res: FusionResult = self.fusion.fuse_modules(module_results)
        self.latest_result = fusion_res
        self.latest_risk = fusion_res.smoothed_probability
        self.latest_category = fusion_res.category

        # 6. Collect findings from expert modules plus explainability engine
        new_findings: List[FindingEvent] = []
        for m_res in [m1_res, m2_res, m3_res, m4_res, m5_res]:
            new_findings.extend(m_res.findings)

        explain_findings = self.explain.generate_findings(
            features=fusion_res.feature_dict,
            t_sec=t_sec,
        )
        # Deduplicate findings by title & approximate time
        existing_titles = {f.title for f in new_findings}
        for ef in explain_findings:
            if ef.title not in existing_titles:
                new_findings.append(ef)

        self.all_findings.extend(new_findings)

        # 7. Visual Heatmap Evidence (if suspicious)
        edge_logit = m3_res.features.get("edge_logit", 0.0)
        evidence = self.explain.generate_heatmap_evidence(
            session_id=self.session_id,
            window_idx=self.window_count,
            face_crops=prep.face_crops,
            edge_logit=edge_logit,
        )

        # Record window stats
        self.window_history.append({
            "window_id": self.window_count,
            "t_sec": round(t_sec, 2),
            "risk": round(fusion_res.smoothed_probability, 4),
            "category": fusion_res.category,
            "quality_trust": round(qual_res.features.get("quality_trust", 0.85), 3),
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
