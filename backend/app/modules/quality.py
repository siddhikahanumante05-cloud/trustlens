"""Q: Quality Gate Module.
Evaluates stream reliability (blur, brightness, face coverage, dropped frames, audio energy)
to modulate trust in downstream expert modules.
Does not vote on real vs fake.
"""
import time
from typing import Set
from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.signals.quality import evaluate_quality


class QualityGateModule(ExpertModule):
    module_id = "Q_quality"
    cadence = "fast"

    def required_inputs(self) -> Set[str]:
        return {"quality"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        blur = float(inputs.quality.get("blur", 80.0))
        brightness = float(inputs.quality.get("brightness", 120.0))
        face_ratio = float(inputs.quality.get("face_ratio", 0.35))
        frame_drop_ratio = float(inputs.metadata.get("frame_drop_ratio", 0.0))

        qual_res = evaluate_quality(
            blur=blur,
            brightness=brightness,
            face_ratio=face_ratio,
            frame_drop_ratio=frame_drop_ratio,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Confidence is high if we can measure quality accurately
        confidence = 1.0 if len(inputs.frames) > 0 else 0.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok" if confidence > 0.0 else "skipped",
            risk=0.0,  # Quality gate does not assert deepfake risk
            confidence=confidence,
            features={
                "quality_trust": qual_res.quality_trust,
                "blur_score": qual_res.blur_score,
                "brightness": qual_res.brightness,
                "face_ratio": qual_res.face_ratio,
                "frame_drop_ratio": frame_drop_ratio,
            },
            findings=[],
            implementation="heuristic",
            model_version="1.0",
            latency_ms=latency_ms,
        )
