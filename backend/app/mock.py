"""Mock engine for TrustLens backend.
Emits scripted events matching the frontend contract with realistic timing.
"""
import asyncio
import time
from typing import Callable, Awaitable, List, Tuple, Dict, Any
from app.schemas import (
    CheckEvent,
    FindingEvent,
    RiskEvent,
    ActionEvent,
    SummaryEvent,
    CheckId,
)

# Standard checks evaluated in TrustLens
ALL_CHECKS: List[CheckId] = ["source", "light", "face", "voice", "lips"]

# Fake scenario events: (delay_ms, event_dict)
FAKE_TIMELINE = [
    (500, CheckEvent(type="check", id="source", state="running")),
    (3000, CheckEvent(type="check", id="source", state="bad")),
    (
        3100,
        FindingEvent(
            type="finding",
            id="source",
            severity="high",
            title="Video may not come from a real camera",
            detail='The camera name looks like "OBS Virtual Camera".',
            t=3.1,
        ),
    ),
    (3100, RiskEvent(type="risk", value=0.35)),
    (5000, CheckEvent(type="check", id="light", state="bad")),
    (
        5100,
        FindingEvent(
            type="finding",
            id="light",
            severity="high",
            title="Face did not react to screen light",
            detail="A real face reflects changing colors. This one stayed the same.",
            t=5.1,
        ),
    ),
    (5100, RiskEvent(type="risk", value=0.58)),
    (7000, CheckEvent(type="check", id="lips", state="bad")),
    (
        7100,
        FindingEvent(
            type="finding",
            id="lips",
            severity="medium",
            title="Mouth does not match the sound",
            detail='Lips stayed open on the "P" in "Paper" at 1.2 s.',
            t=7.1,
        ),
    ),
    (7100, RiskEvent(type="risk", value=0.74)),
    # User requirement: On "Likely deepfake" (risk >= 0.70), emit an ActionEvent
    (
        7200,
        ActionEvent(
            type="action",
            text="Ask the caller to turn their head left, then say the phrase again.",
        ),
    ),
    (9000, CheckEvent(type="check", id="face", state="warn")),
    (
        9100,
        FindingEvent(
            type="finding",
            id="face",
            severity="medium",
            title="Skin looks too smooth near the hairline",
            detail="Blending marks found where the face meets the hair.",
            t=9.1,
        ),
    ),
    (9100, RiskEvent(type="risk", value=0.86)),
    (11000, CheckEvent(type="check", id="voice", state="warn")),
    (
        11100,
        FindingEvent(
            type="finding",
            id="voice",
            severity="low",
            title="Voice may be computer-made",
            detail="Sound pattern is close to cloned voices (low confidence).",
            t=11.1,
        ),
    ),
    (11100, RiskEvent(type="risk", value=0.91)),
    (
        12000,
        SummaryEvent(
            type="summary",
            text="High probability of deepfake detected (Risk: 91%). Virtual camera signature identified alongside absent reflection to light challenge and mismatched bilabial lip closure.",
            ai_generated=False,
        ),
    ),
]

# Real scenario events: (delay_ms, event_dict)
REAL_TIMELINE = [
    (3000, CheckEvent(type="check", id="source", state="ok")),
    (4500, CheckEvent(type="check", id="light", state="ok")),
    (6000, CheckEvent(type="check", id="face", state="ok")),
    (7500, CheckEvent(type="check", id="voice", state="ok")),
    (9000, CheckEvent(type="check", id="lips", state="ok")),
    (9000, RiskEvent(type="risk", value=0.07)),
    (
        10000,
        SummaryEvent(
            type="summary",
            text="Authentic live caller confirmed (Risk: 7%). Natural illumination response, authentic hardware camera timing, and natural lip synchrony verified.",
            ai_generated=False,
        ),
    ),
]


class MockEngine:
    """Simulates real-time deepfake analysis pipeline with realistic timing."""

    def __init__(self, send_fn: Callable[[Dict[str, Any]], Awaitable[None]]):
        self.send_fn = send_fn
        self._running = False

    async def run(self, scenario: str = "fake", speed_multiplier: float = 1.0) -> None:
        """Execute the timeline for real or fake scenario."""
        self._running = True

        # 1. Warm-up state: Emit initial check: running on all channels + base risk 0.10
        for ch in ALL_CHECKS:
            await self.send_fn(CheckEvent(type="check", id=ch, state="running").model_dump())
            await asyncio.sleep(0.02 / speed_multiplier)

        await self.send_fn(RiskEvent(type="risk", value=0.10).model_dump())

        timeline = FAKE_TIMELINE if scenario == "fake" else REAL_TIMELINE
        last_ms = 0

        for delay_ms, event in timeline:
            if not self._running:
                break
            delta_ms = delay_ms - last_ms
            if delta_ms > 0:
                await asyncio.sleep((delta_ms / 1000.0) / speed_multiplier)
            last_ms = delay_ms

            if not self._running:
                break
            await self.send_fn(event.model_dump())

    def stop(self) -> None:
        """Stop the simulation."""
        self._running = False
