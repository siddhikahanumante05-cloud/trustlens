"""M4: Face Geometry Module.
Answers: Does the face move like a real head: a rigid skull with muscle-driven deformation?
Evaluates landmark trajectory smoothness, second-difference jitter energy, and occlusion artifacts.
"""
import time
from typing import Set, List
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.signals.identity import analyze_identity_consistency


class FaceGeometryModule(ExpertModule):
    module_id = "M4_geometry"
    cadence = "fast"

    def required_inputs(self) -> Set[str]:
        return {"landmarks_series"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        landmarks = inputs.landmarks_series
        if not landmarks or len(landmarks) < 4:
            return ModuleResult(
                module_id=self.module_id,
                status="skipped",
                risk=0.0,
                confidence=0.0,
                features={
                    "landmark_jitter": 0.0,
                    "occlusion_artifact": 0.0,
                },
                findings=[],
                implementation="heuristic",
                model_version="1.0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Landmark Jitter & Motion Stability
        id_res = analyze_identity_consistency(
            embeddings=None,
            landmarks_series=landmarks,
        )

        # 2. Hand occlusion response
        occlusion = float(inputs.hand_occlusion_score)

        findings: List[FindingEvent] = []
        if id_res.landmark_jitter >= 0.08:
            findings.append(
                FindingEvent(
                    id="face",
                    severity="medium",
                    title="Facial Landmark Jitter Anomaly",
                    detail="Unnatural high-frequency motion jitter detected across facial keypoints.",
                    t=round(inputs.t_sec, 2),
                )
            )

        if occlusion >= 0.60:
            findings.append(
                FindingEvent(
                    id="face",
                    severity="medium",
                    title="Occlusion Reconstruction Seams",
                    detail="Facial boundary degradation observed during hand/object occlusion.",
                    t=round(inputs.t_sec, 2),
                )
            )

        # Risk mapping
        jitter_risk = float(np.clip(id_res.landmark_jitter / 0.10, 0.0, 1.0))
        occl_risk = float(np.clip(occlusion, 0.0, 1.0))
        risk = max(jitter_risk, occl_risk * 0.7)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=risk,
            confidence=0.80,
            features={
                "landmark_jitter": id_res.landmark_jitter,
                "occlusion_artifact": occlusion,
            },
            findings=findings,
            implementation="heuristic",
            model_version="1.0",
            latency_ms=latency_ms,
        )
