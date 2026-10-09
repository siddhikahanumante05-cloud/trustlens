"""Pydantic schemas for TrustLens WebSocket protocol and REST APIs."""
from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Client -> Server WebSocket Messages
# ============================================================================

class StartMessage(BaseModel):
    type: Literal["start"] = "start"
    client_time: int
    session_id: Optional[str] = None
    consent: Optional[bool] = None


class MetaSettings(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    frameRate: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class MetaMessage(BaseModel):
    type: Literal["meta"] = "meta"
    camera_label: Optional[str] = None
    settings: Optional[Union[MetaSettings, Dict[str, Any]]] = None
    frame_intervals_ms: Optional[List[float]] = None
    user_agent: Optional[str] = None


class LightStep(BaseModel):
    color: str  # Hex e.g. "#ffffff", "#ff4d4d"
    ms: int     # Duration in ms (>= 333 ms to satisfy <= 3 changes/sec)


class LightChallengeMessage(BaseModel):
    type: Literal["challenge"] = "challenge"
    kind: Literal["light"] = "light"
    t0: int
    sequence: List[LightStep]


class PhraseChallengeMessage(BaseModel):
    type: Literal["challenge"] = "challenge"
    kind: Literal["phrase"] = "phrase"
    expected: str
    t0: int


class ClockPingMessage(BaseModel):
    type: Literal["clock"] = "clock"
    client_time: int


ClientMessage = Union[
    StartMessage,
    MetaMessage,
    LightChallengeMessage,
    PhraseChallengeMessage,
    ClockPingMessage,
]


# ============================================================================
# Server -> Client WebSocket Messages (Contract Compliant)
# ============================================================================

CheckId = Literal["source", "light", "face", "voice", "lips"]
CheckState = Literal["running", "ok", "warn", "bad"]
Severity = Literal["low", "medium", "high"]


class CheckEvent(BaseModel):
    type: Literal["check"] = "check"
    id: CheckId
    state: CheckState


class FindingEvent(BaseModel):
    type: Literal["finding"] = "finding"
    id: str  # e.g. "source", "light", "face", "voice", "lips"
    severity: Severity
    title: str
    detail: str
    t: float  # seconds since start


class RiskEvent(BaseModel):
    type: Literal["risk"] = "risk"
    value: float  # 0.0 to 1.0


class ActionEvent(BaseModel):
    type: Literal["action"] = "action"
    text: str  # Plain English guidance for human analyst e.g. "Ask the caller to turn their head left..."


class SummaryEvent(BaseModel):
    type: Literal["summary"] = "summary"
    text: str
    ai_generated: bool = False


class EvidenceEvent(BaseModel):
    type: Literal["evidence"] = "evidence"
    kind: Literal["heatmap", "timeline"]
    url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ClockPongEvent(BaseModel):
    type: Literal["clock"] = "clock"
    server_time: int
    echo: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


ServerMessage = Union[
    CheckEvent,
    FindingEvent,
    RiskEvent,
    ActionEvent,
    SummaryEvent,
    EvidenceEvent,
    ClockPongEvent,
    ErrorEvent,
]


# ============================================================================
# REST API Request & Response Schemas
# ============================================================================

class SessionCreateResponse(BaseModel):
    session_id: str
    nonce: str
    phrase: str
    light_sequence: List[LightStep]
    expires_at: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    ffmpeg_present: bool
    stubs_in_use: bool
    version: str
    narrator_mode: str
    ollama_available: bool


class CaseRecord(BaseModel):
    session_id: str
    created_at: int
    consent_status: str  # "consented" | "bypassed" | "none"
    stubs_in_use: bool
    risk_score: Optional[float] = None
    findings: List[FindingEvent] = Field(default_factory=list)
    summary: Optional[str] = None
    ai_generated_summary: bool = False
