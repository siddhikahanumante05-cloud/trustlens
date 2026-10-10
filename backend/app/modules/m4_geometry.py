"""M4: Face Geometry Module.
Answers: Does the face move like a real head: a rigid skull with muscle-driven deformation?
Evaluates landmark trajectory smoothness, second-difference jitter energy, occlusion artifacts,
and biological facial liveness (detecting static 2D photo and print presentation attacks).
"""
import time
from typing import Set, List, Tuple
import numpy as np
import cv2

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.signals.identity import analyze_identity_consistency


class FaceGeometryModule(ExpertModule):
    module_id = "M4_geometry"
    cadence = "fast"

    def __init__(self):
        super().__init__()
        self._consecutive_static_windows = 0

    def required_inputs(self) -> Set[str]:
        return {"landmarks_series"}

    def _evaluate_liveness_dynamics(self, crops: List[np.ndarray]) -> Tuple[float, float, float, float]:
        """
        Evaluate non-rigid muscle-driven facial deformation vs rigid 2D planar motion.
        Compensates for global camera/hand translation using phase correlation.
        Applies Gaussian filter to eliminate CMOS webcam sensor thermal noise.
        """
        if not crops or len(crops) < 4:
            return 5.0, 5.0, 0.0, 0.5

        valid_crops = [c for c in crops if c is not None and c.size > 0 and np.max(c) > 10]
        if len(valid_crops) < 4:
            return 5.0, 5.0, 0.0, 0.5

        n = min(len(valid_crops), 16)
        # Apply Gaussian blur to remove high-frequency webcam sensor grain (noise std ~ 3-4)
        grays = []
        for c in valid_crops[:n]:
            g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32) if len(c.shape) == 3 else c.astype(np.float32)
            grays.append(cv2.GaussianBlur(g, (5, 5), 1.2))

        h, w = grays[0].shape

        eye_y0, eye_y1 = int(h * 0.20), int(h * 0.52)
        inner_y0, inner_y1 = int(h * 0.15), int(h * 0.85)
        inner_x0, inner_x1 = int(w * 0.15), int(w * 0.85)

        base = grays[0]
        rigid_disps = []
        nonrigid_diffs = []
        eye_diffs = []
        corrs = []

        for i in range(1, n):
            curr = grays[i]
            # Rigid translation compensation
            shift, _ = cv2.phaseCorrelate(base, curr)
            dx, dy = float(shift[0]), float(shift[1])
            disp = float(np.sqrt(dx**2 + dy**2))
            rigid_disps.append(disp)

            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            aligned = cv2.warpAffine(curr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            inner_base = base[inner_y0:inner_y1, inner_x0:inner_x1]
            inner_aligned = aligned[inner_y0:inner_y1, inner_x0:inner_x1]

            inner_diff = np.abs(inner_aligned - inner_base)
            nonrigid_diffs.append(float(np.mean(inner_diff)))

            eye_diff = np.abs(aligned[eye_y0:eye_y1, inner_x0:inner_x1] - base[eye_y0:eye_y1, inner_x0:inner_x1])
            eye_diffs.append(float(np.mean(eye_diff)))

            if np.std(inner_base) > 1e-4 and np.std(inner_aligned) > 1e-4:
                c_val = float(np.corrcoef(inner_base.flatten(), inner_aligned.flatten())[0, 1])
                corrs.append(c_val)

        mean_nonrigid = float(np.mean(nonrigid_diffs)) if nonrigid_diffs else 0.0
        max_eye = float(np.max(eye_diffs)) if eye_diffs else 0.0
        mean_rigid = float(np.mean(rigid_disps)) if rigid_disps else 0.0
        mean_corr = float(np.mean(corrs)) if corrs else 0.9

        return mean_nonrigid, max_eye, mean_rigid, mean_corr

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
                    "static_face_risk": 0.0,
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

        # 3. Biological Liveness & Static Photo Detection
        crops = inputs.face_crops or []
        mean_nonrigid, max_eye, mean_rigid, mean_corr = self._evaluate_liveness_dynamics(crops)

        static_face_risk = 0.0
        if crops and len(crops) >= 4 and np.max(crops[0]) > 10:
            is_statically_frozen = (mean_nonrigid < 1.70 and max_eye < 2.8) or (mean_corr >= 0.985 and max_eye < 3.0)
            is_rigid_card_moving = (mean_rigid >= 0.4 and mean_nonrigid < 1.70)

            if is_rigid_card_moving:
                self._consecutive_static_windows += 1
                static_face_risk = 0.88
                findings.append(
                    FindingEvent(
                        id="face",
                        severity="high",
                        title="Rigid 2D Photo Motion Detected",
                        detail="The face moves as a flat rigid plane with zero internal facial deformation. Consistent with a printed photo being moved.",
                        t=round(inputs.t_sec, 2),
                    )
                )
            elif is_statically_frozen:
                self._consecutive_static_windows += 1
                static_face_risk = 0.88 if self._consecutive_static_windows >= 2 else 0.75
                findings.append(
                    FindingEvent(
                        id="face",
                        severity="high",
                        title="Static 2D Photo Attack Detected",
                        detail="Zero biological motion, eye blinking, or micro-expression detected across video frames. Facial features are static.",
                        t=round(inputs.t_sec, 2),
                    )
                )
            else:
                self._consecutive_static_windows = max(0, self._consecutive_static_windows - 1)

        # Risk mapping
        jitter_risk = float(np.clip(id_res.landmark_jitter / 0.10, 0.0, 1.0))
        occl_risk = float(np.clip(occlusion, 0.0, 1.0))
        risk = max(jitter_risk, occl_risk * 0.7, static_face_risk)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=risk,
            confidence=0.80,
            features={
                "landmark_jitter": id_res.landmark_jitter,
                "occlusion_artifact": occlusion,
                "static_face_risk": static_face_risk,
                "facial_nonrigid_motion": mean_nonrigid,
            },
            findings=findings,
            implementation="heuristic",
            model_version="1.0",
            latency_ms=latency_ms,
        )
