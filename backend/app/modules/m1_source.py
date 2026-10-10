"""M1: Source and Liveness Module.
Answers: Is this stream coming from a live camera in front of this screen?
Evaluates camera provenance metadata, timing jitter, active screen illumination response,
and presentation attack detection (specular screen glass reflections, motion decoupling, Moiré).
"""
import time
from typing import Set, List
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.signals.provenance import analyze_provenance
from app.signals.light import analyze_light_response
from app.signals.presentation import analyze_presentation_replay


class SourceLivenessModule(ExpertModule):
    module_id = "M1_source"
    cadence = "fast"

    def required_inputs(self) -> Set[str]:
        return {"skin_rois", "timestamps"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        camera_label = inputs.metadata.get("camera_label")
        frame_intervals_ms = inputs.metadata.get("frame_intervals_ms")

        # 1. Provenance / Virtual Camera checks
        prov_res = analyze_provenance(
            camera_label=camera_label,
            frame_intervals_ms=frame_intervals_ms,
        )

        # 2. Active Screen Light Challenge
        challenge_seq = None
        challenge_t0 = None
        if inputs.challenges:
            for ch in inputs.challenges:
                if ch.get("kind") == "light":
                    challenge_seq = ch.get("sequence")
                    challenge_t0 = ch.get("t0")
                    break

        light_res = analyze_light_response(
            face_rois=inputs.skin_rois.get("face", []),
            neck_rois=inputs.skin_rois.get("neck", []),
            bg_rois=inputs.skin_rois.get("background", []),
            timestamps=inputs.timestamps,
            challenge_sequence=challenge_seq,
            challenge_t0=challenge_t0,
            tap_mode=inputs.tap_mode,
        )

        # 3. Presentation Attack / Mobile Screen Replay & Reflection Analysis
        presentation_res = analyze_presentation_replay(
            face_crops=inputs.face_crops,
            frames=inputs.frames,
            landmarks_series=inputs.landmarks_series,
            face_bboxes=inputs.face_bboxes,
        )

        # Findings generation
        findings: List[FindingEvent] = []
        if prov_res.source_risk >= 0.70:
            findings.append(
                FindingEvent(
                    id="source",
                    severity="high",
                    title="Virtual Camera Stream Detected",
                    detail="Software capture device or synthetic frame timing signature identified.",
                    t=round(inputs.t_sec, 2),
                )
            )
        elif prov_res.source_risk >= 0.35:
            findings.append(
                FindingEvent(
                    id="source",
                    severity="medium",
                    title="Suspicious Video Source Timing",
                    detail="Frame delivery jitter deviates from physical webcam hardware behavior.",
                    t=round(inputs.t_sec, 2),
                )
            )

        if presentation_res.replay_score >= 0.60:
            findings.append(
                FindingEvent(
                    id="source",
                    severity="high",
                    title="Mobile Screen Replay Detected",
                    detail="Specular glass reflections, motion decoupling, and display artifacts indicate a video replay on a mobile screen to the webcam.",
                    t=round(inputs.t_sec, 2),
                )
            )
        elif presentation_res.replay_score >= 0.35:
            findings.append(
                FindingEvent(
                    id="source",
                    severity="medium",
                    title="Screen Glass Reflection Glare",
                    detail="Planar specular reflection highlights and subpixel artifacts observed over facial region.",
                    t=round(inputs.t_sec, 2),
                )
            )

        if light_res.snr >= 2.0 and challenge_seq:
            if light_res.corr_face < 0.15:
                findings.append(
                    FindingEvent(
                        id="light",
                        severity="high",
                        title="Failed Active Light Challenge",
                        detail="Skin luminance did not modulate in sync with the randomized challenge sequence.",
                        t=round(inputs.t_sec, 2),
                    )
                )
            elif light_res.corr_face < 0.30:
                findings.append(
                    FindingEvent(
                        id="light",
                        severity="medium",
                        title="Weak Active Light Modulation",
                        detail="Low photometric correlation between challenge sequence and face reflection.",
                        t=round(inputs.t_sec, 2),
                    )
                )

        # Composite risk: provenance risk + light failure risk (if challenged) + screen replay risk
        if challenge_seq and light_res.snr >= 1.5:
            light_risk = float(np.clip(1.0 - (light_res.corr_face + 0.2) / 1.0, 0.0, 1.0))
            combined_risk = max(prov_res.source_risk, light_risk, presentation_res.replay_score)
            confidence = min(1.0, (light_res.snr / 3.0) * 0.9 + 0.1)
        else:
            combined_risk = max(prov_res.source_risk, presentation_res.replay_score)
            if presentation_res.replay_score >= 0.50:
                confidence = 0.90
            else:
                confidence = 0.85 if camera_label or frame_intervals_ms else 0.50

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=float(combined_risk),
            confidence=float(confidence),
            features={
                "source_risk": prov_res.source_risk,
                "replay_score": presentation_res.replay_score,
                "screen_glare": presentation_res.glare_score,
                "reflection_decoupling": presentation_res.decoupling_score,
                "planar_reflection": presentation_res.planar_score,
                "moire_score": presentation_res.moire_score,
                "chroma_lattice_score": presentation_res.chroma_lattice_score,
                "bezel_score": presentation_res.bezel_score,
                "light_corr_face": light_res.corr_face,
                "light_lag_ms": light_res.lag_ms,
                "light_neck_face_ratio": light_res.neck_face_ratio,
                "light_neck_face_corr": light_res.neck_face_corr,
                "light_snr": light_res.snr,
            },
            findings=findings,
            implementation="heuristic",
            model_version="1.0",
            latency_ms=latency_ms,
        )

