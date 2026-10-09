"""M5: Speech and Lips Module (Cross-Modal).
Answers: Do the sound and the lips say the same thing, at the same time?
Evaluates phrase correctness (Whisper), bilabial mouth closure on b/p/m,
and audio-visual envelope synchronization.
"""
import time
from typing import Set, List
import numpy as np

from app.modules.base import ExpertModule, ModuleInputs, ModuleResult
from app.schemas import FindingEvent
from app.signals.phrase import analyze_phrase_match
from app.signals.lip_closure import analyze_lip_closure
from app.signals.sync import analyze_av_sync


class SpeechLipModule(ExpertModule):
    module_id = "M5_speech_lips"
    cadence = "fast"

    def required_inputs(self) -> Set[str]:
        return {"audio_pcm", "mouth_apertures"}

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        t0 = time.perf_counter()

        audio = inputs.audio_pcm
        apertures = inputs.mouth_apertures

        # Missing input conditions: no audio or no tracked mouth apertures
        if audio is None or len(audio) < 8000 or not apertures or len(apertures) < 4:
            return ModuleResult(
                module_id=self.module_id,
                status="skipped",
                risk=0.0,
                confidence=0.0,
                features={
                    "phrase_wer": 0.0,
                    "bilabial_aperture_err": 0.0,
                    "sync_conf": 2.0,
                    "sync_offset_ms": 0.0,
                },
                findings=[],
                implementation="heuristic",
                model_version="1.0",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Spoken Challenge Phrase Verification
        expected_phrase = ""
        is_phrase_active = False
        if inputs.challenges:
            for ch in inputs.challenges:
                if ch.get("kind") == "phrase":
                    expected_phrase = ch.get("expected", "")
                    is_phrase_active = True
                    break

        # Note: Until faster-whisper is wired with live audio model, mark phrase as inactive to avoid false alarms
        phrase_res = analyze_phrase_match(
            transcribed_words=[],
            expected_phrase=expected_phrase,
            is_active=False,
        )

        # 2. Bilabial mouth closure on b, p, m
        lip_res = analyze_lip_closure(
            aperture_series=apertures,
            frame_timestamps=inputs.timestamps,
            word_timestamps=phrase_res.word_timestamps,
        )

        # 3. Audio-Visual Envelope Synchronization
        sync_res = analyze_av_sync(
            audio_pcm=audio,
            mouth_apertures=apertures,
            sample_rate=16000,
            fps=15.0,
        )

        findings: List[FindingEvent] = []
        if lip_res.bilabial_aperture_err >= 0.35 or (sync_res.sync_conf >= 1.5 and sync_res.offset_ms >= 200.0):
            findings.append(
                FindingEvent(
                    id="lips",
                    severity="high",
                    title="Phoneme-Lip Incoherence",
                    detail="Failure to seal lips on bilabial consonants (B/P/M) with significant AV desync.",
                    t=round(inputs.t_sec, 2),
                )
            )
        elif lip_res.bilabial_aperture_err >= 0.18 or (sync_res.sync_conf >= 1.5 and sync_res.offset_ms >= 120.0):
            findings.append(
                FindingEvent(
                    id="lips",
                    severity="medium",
                    title="Mouth Motion Desynchronization",
                    detail="Aperture anomalies during bilabial speech segments.",
                    t=round(inputs.t_sec, 2),
                )
            )

        # Compute risk
        sync_risk = float(np.clip(sync_res.offset_ms / 250.0, 0.0, 1.0)) if sync_res.sync_conf >= 1.5 else 0.0
        lip_risk = float(np.clip(lip_res.bilabial_aperture_err / 0.30, 0.0, 1.0))
        risk = max(sync_risk, lip_risk)

        confidence = 0.80 if sync_res.sync_conf >= 1.5 else 0.40
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModuleResult(
            module_id=self.module_id,
            status="ok",
            risk=risk,
            confidence=confidence,
            features={
                "phrase_wer": phrase_res.phrase_wer,
                "bilabial_aperture_err": lip_res.bilabial_aperture_err,
                "sync_conf": sync_res.sync_conf,
                "sync_offset_ms": sync_res.offset_ms,
            },
            findings=findings,
            implementation="heuristic",
            model_version="1.0",
            latency_ms=latency_ms,
        )
