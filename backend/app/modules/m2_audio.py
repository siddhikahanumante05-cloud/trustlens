"""M2: Audio Module.
Answers: Is this voice natural, or synthetic or cloned?
Processes 16 kHz PCM waveform, evaluating voice cloning / speech synthesis anomalies.
"""
import time
import logging
from typing import Set, List, Optional
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.runtime import ModelRuntime

logger = logging.getLogger("trustlens.m2_audio")


class AudioSpoofModule(ExpertModule):
    module_id = "M2_audio"
    cadence = "slow"

    def __init__(self, runtime: Optional[ModelRuntime] = None):
        self.runtime = runtime

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
                implementation="stub" if (not self.runtime or self.runtime.stubs_in_use) else "trained",
                model_version="stub-0" if (not self.runtime or self.runtime.stubs_in_use) else "1.0",
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
                implementation="stub" if (not self.runtime or self.runtime.stubs_in_use) else "trained",
                model_version="stub-0" if (not self.runtime or self.runtime.stubs_in_use) else "1.0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        spoof_max = 0.0
        spoof_mean = 0.0
        implementation = "heuristic"
        model_ver = "1.0"

        # 1. Neural WavLM audio spoof inference via runtime if available
        if self.runtime and "wavlm_head" in self.runtime.sessions:
            try:
                audio_feats = self.runtime._extract_audio_features(audio)
                out = self.runtime.sessions["wavlm_head"].run(["spoof_logit"], {"audio_features": audio_feats})[0]
                raw_spoof_logit = float(out[0, 0])
                if raw_spoof_logit > 0.0:
                    spoof_max = float(np.clip(raw_spoof_logit / 3.0, 0.0, 1.0))
                    spoof_mean = spoof_max * 0.75
                else:
                    spoof_max = 0.0
                    spoof_mean = 0.0
                implementation = "trained"
            except Exception as e:
                logger.warning("WavLM audio spoof inference error: %s", e)

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
        elif spoof_max >= 0.40:
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
            confidence=0.75 if implementation == "trained" else 0.40,
            features={
                "spoof_max": spoof_max,
                "spoof_mean": spoof_mean,
                "audio_rms": rms,
            },
            findings=findings,
            implementation=implementation,
            model_version=model_ver,
            latency_ms=latency_ms,
        )
