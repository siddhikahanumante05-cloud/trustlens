"""Base module protocol and data contracts for TrustLens multi-modal expert modules."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Set, Protocol, Optional, Literal
from app.schemas import FindingEvent, Severity


@dataclass
class ModuleResult:
    module_id: str                      # e.g. "M1_source", "M2_audio", "M3_video", "M4_geometry", "M5_speech_lips", "Q_quality"
    status: Literal["ok", "skipped", "degraded", "error"]
    risk: float                         # 0.0 to 1.0 (only evaluated if status in ("ok", "degraded"))
    confidence: float                   # 0.0 to 1.0 (how much fusion should trust this module right now)
    features: Dict[str, float]          # Named features consumed by fusion and explanations
    findings: List[FindingEvent] = field(default_factory=list)
    implementation: Literal["trained", "heuristic", "stub"] = "heuristic"
    model_version: str = "v1.0"
    latency_ms: float = 0.0


@dataclass
class ModuleInputs:
    # Frame and timing data
    frames: List[Any]                   # Sampled RGB frames
    timestamps: List[float]             # Frame timestamps (sec)
    audio_pcm: Any                      # 16 kHz mono PCM numpy array
    t_sec: float                        # Current window timestamp (sec)

    # Preprocessed face and landmarks
    face_crops: List[Any]               # Aligned face crops (160x160x3)
    landmarks_series: List[Any]         # 478 MediaPipe / geometric landmarks per frame
    mouth_apertures: List[float]        # Normalized mouth apertures per frame
    skin_rois: Dict[str, List[Dict[str, float]]]  # "face", "neck", "background" chromaticity/luminance
    hand_occlusion_score: float         # 0.0 to 1.0

    # Quality assessments
    quality: Dict[str, float]           # blur, brightness, face_ratio, quality_trust

    # Challenges and metadata
    metadata: Dict[str, Any]            # Camera label, frame_intervals_ms, WebRTC stats
    challenges: List[Dict[str, Any]]    # Active light or phrase challenge events
    tap_mode: str = "agent"             # "agent" | "caller"


class ExpertModule(Protocol):
    module_id: str
    cadence: Literal["fast", "slow", "once"]

    def required_inputs(self) -> Set[str]:
        ...

    def run(self, inputs: ModuleInputs) -> ModuleResult:
        ...
