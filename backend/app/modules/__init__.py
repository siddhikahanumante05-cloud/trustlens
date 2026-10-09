"""TrustLens Multi-Modal Expert Modules."""
from app.modules.base import ModuleResult, ModuleInputs, ExpertModule
from app.modules.quality import QualityGateModule
from app.modules.m1_source import SourceLivenessModule
from app.modules.m2_audio import AudioSpoofModule
from app.modules.m3_video import VideoAppearanceModule
from app.modules.m4_geometry import FaceGeometryModule
from app.modules.m5_speech_lips import SpeechLipModule

__all__ = [
    "ModuleResult",
    "ModuleInputs",
    "ExpertModule",
    "QualityGateModule",
    "SourceLivenessModule",
    "AudioSpoofModule",
    "VideoAppearanceModule",
    "FaceGeometryModule",
    "SpeechLipModule",
]
