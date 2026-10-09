"""M3: Video Appearance Module.
Answers: Do the face pixels look like a real capture, or a generated or blended face?
Evaluates high-frequency spatial-frequency artifacts (EdgeNet), blend traces (CLIP),
and appearance identity continuity.
"""
import time
from typing import Set, List
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.runtime import ModelRuntime


class VideoAppearanceModule(ExpertModule):
    module_id = "M3_video"
    cadence = "fast"

    def __init__(self, runtime: ModelRuntime):
        self.runtime = runtime

    def required_inputs(self) -> Set[str]:
        return {"face_crops"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        crops = inputs.face_crops
        if not crops or len(crops) == 0:
            return ModuleResult(
                module_id=self.module_id,
                status="skipped",
                risk=0.0,
                confidence=0.0,
                features={
                    "edge_logit": 0.0,
                    "clip_logit": 0.0,
                    "model_disagreement": 0.0,
                    "ood_score": 0.0,
                    "identity_flicker": 0.0,
                },
                findings=[],
                implementation="stub" if self.runtime.stubs_in_use else "trained",
                model_version="1.0-stub-0" if self.runtime.stubs_in_use else "1.0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Spatial & Frequency Models (EdgeNet fast path + CLIP slow path)
        quality_trust = float(inputs.quality.get("quality_trust", 0.85))
        model_out = self.runtime.run_inference(
            face_crops=crops,
            audio_pcm=inputs.audio_pcm,
            quality_trust=quality_trust,
        )

        # 2. Appearance identity stability across consecutive frames
        face_embeddings = [np.mean(crop, axis=(0, 1)) for crop in crops]
        sims = []
        for i in range(len(face_embeddings) - 1):
            e1 = face_embeddings[i].flatten()
            e2 = face_embeddings[i + 1].flatten()
            norm1 = float(np.linalg.norm(e1))
            norm2 = float(np.linalg.norm(e2))
            if norm1 > 1e-6 and norm2 > 1e-6:
                sims.append(float(np.dot(e1, e2) / (norm1 * norm2)))
        mean_sim = float(np.mean(sims)) if sims else 1.0
        identity_flicker = float(np.clip(1.0 - mean_sim, 0.0, 1.0))

        # Findings generation
        findings: List[FindingEvent] = []
        if model_out.edge_logit >= 1.2 or identity_flicker >= 0.50:
            findings.append(
                FindingEvent(
                    id="face",
                    severity="high",
                    title="Deepfake Boundary Artifacts",
                    detail="High-frequency frequency-domain blending seams and temporal embedding flicker detected.",
                    t=round(inputs.t_sec, 2),
                )
            )
        elif model_out.edge_logit >= 0.60 or identity_flicker >= 0.25:
            findings.append(
                FindingEvent(
                    id="face",
                    severity="medium",
                    title="Facial Instability Detected",
                    detail="Micro-jitter and spatial blending boundaries observed across consecutive frames.",
                    t=round(inputs.t_sec, 2),
                )
            )

        # Map edge_logit to [0, 1] probability
        prob = float(1.0 / (1.0 + np.exp(-np.clip(model_out.edge_logit, -5.0, 5.0))))
        risk = max(prob, identity_flicker * 1.2)
        risk = float(np.clip(risk, 0.0, 1.0))

        # Confidence is modulated by quality trust (poor lighting or blur lowers model confidence)
        confidence = float(np.clip(quality_trust, 0.20, 1.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=risk,
            confidence=confidence,
            features={
                "edge_logit": model_out.edge_logit,
                "clip_logit": model_out.clip_logit,
                "model_disagreement": model_out.model_disagreement,
                "ood_score": model_out.ood_score,
                "identity_flicker": identity_flicker,
            },
            findings=findings,
            implementation="stub" if self.runtime.stubs_in_use else "trained",
            model_version="1.0-stub-0" if self.runtime.stubs_in_use else "1.0",
            latency_ms=latency_ms,
        )
