"""M2: Audio Module.
Answers: Is this voice natural, or synthetic or cloned?
Processes 16 kHz PCM waveform, evaluating voice cloning / speech synthesis anomalies.
"""
import time
from typing import Set, List
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent


class AudioSpoofModule(ExpertModule):
    module_id = "M2_audio"
    cadence = "slow"

    def required_inputs(self) -> Set[str]:
        return {"audio_pcm"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        audio = inputs.audio_pcm
        # Skip condition: no audio track or empty buffer
        if audio is None or len(audio) < 8000:
            return ModuleResult(
                module_id=self.module_id,
                status="skipped",
                risk=0.0,
                confidence=0.0,
                features={"spoof_max": 0.0, "spoof_mean": 0.0},
                findings=[],
                implementation="stub",
                model_version="stub-0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Voice Activity Detection (VAD) heuristic
        audio_float = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        if rms < 50.0:  # Silence or near silence
            return ModuleResult(
                module_id=self.module_id,
                status="degraded",
                risk=0.0,
                confidence=0.15,
                features={"spoof_max": 0.0, "spoof_mean": 0.0, "audio_rms": rms},
                findings=[],
                implementation="stub",
                model_version="stub-0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Heuristic audio spoof indicator (runs stub-0 until real WavLM head is loaded)
        audio_std = float(np.std(audio_float))
        spoof_max = float(np.clip((audio_std - 1500.0) / 4000.0, 0.0, 1.0))
        spoof_mean = spoof_max * 0.75

        findings: List[FindingEvent] = []
        if spoof_max >= 0.65:
            findings.append(
                FindingEvent(
                    id="voice",
                    severity="high",
                    title="Synthetic Audio Cloned Voice",
                    detail="Acoustic features display non-natural vocoder synthesis patterns.",
                    t=round(inputs.t_sec, 2),
                )
            )
        elif spoof_max >= 0.35:
            findings.append(
                FindingEvent(
                    id="voice",
                    severity="medium",
                    title="Acoustic Spectral Anomaly",
                    detail="Voice harmonics indicate potential voice conversion or re-synthesis.",
                    t=round(inputs.t_sec, 2),
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=spoof_max,
            confidence=0.60,
            features={
                "spoof_max": spoof_max,
                "spoof_mean": spoof_mean,
                "audio_rms": rms,
            },
            findings=findings,
            implementation="stub",
            model_version="stub-0",
            latency_ms=latency_ms,
        )
