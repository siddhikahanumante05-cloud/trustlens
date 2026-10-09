"""WebSocket ingest and session handler."""
import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, status
from app.config import settings
from app.schemas import (
    StartMessage,
    ClockPongEvent,
    ErrorEvent,
    CheckEvent,
    RiskEvent,
    ActionEvent,
    SummaryEvent,
)
from app.mock import MockEngine, ALL_CHECKS
from app.metrics import ACTIVE_SESSIONS
from app.decoder import ChunkDecoder
from app.windower import Windower
from app.pipeline import LivePipeline

logger = logging.getLogger("trustlens.ws")

# Track active connections
active_connections: Dict[str, WebSocket] = {}


async def handle_ws_analyze(
    websocket: WebSocket,
    session_id: Optional[str] = None,
    mock: Optional[str] = None,
    speed: float = 1.0,
):
    """Handle incoming WebSocket connection for real-time video/audio analysis."""
    # 1. Enforce concurrent session limit
    if len(active_connections) >= settings.MAX_SESSIONS:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Max concurrent sessions reached")
        return

    await websocket.accept()
    conn_id = session_id or f"conn_{time.time_ns()}"
    active_connections[conn_id] = websocket
    ACTIVE_SESSIONS.inc()

    mock_engine: Optional[MockEngine] = None
    mock_task: Optional[asyncio.Task] = None
    decoder: Optional[ChunkDecoder] = None
    windower: Optional[Windower] = None
    pipeline: Optional[LivePipeline] = None
    pipeline_task: Optional[asyncio.Task] = None

    session_start_time = time.time()
    max_duration_sec = 600  # 10 minutes max session length
    is_mock = bool(mock) or settings.USE_MOCK
    mock_scenario = mock if mock in ["real", "fake"] else "fake"
    consent_status = "none"

    async def send_json(data: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        # 2. Wait for first message, which must be 'start'
        raw_msg = await websocket.receive_text()
        start_data = json.loads(raw_msg)
        start_msg = StartMessage(**start_data)

        # 3. Check consent
        if start_msg.consent is True:
            consent_status = "consented"
        elif is_mock or (settings.DEMO_BYPASS_CONSENT and settings.ENV in ["demo", "development"]):
            consent_status = "bypassed"
            logger.warning("Caller consent bypassed under demo/mock mode")
        else:
            # Reject by default
            err = ErrorEvent(
                type="error",
                code="CONSENT_REQUIRED",
                message="Analysis requires explicit consent from caller.",
            )
            await websocket.send_text(err.model_dump_json())
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Consent required")
            return

        # 4. Handle Mock Mode or Real Mode
        if is_mock:
            mock_engine = MockEngine(send_fn=send_json)
            # Run mock in background so client can still send messages (clock, meta, challenge)
            mock_task = asyncio.create_task(mock_engine.run(scenario=mock_scenario, speed_multiplier=speed))
        else:
            # Initialize live pipeline
            decoder = ChunkDecoder()
            windower = Windower(decoder=decoder, stride_sec=1.0)
            pipeline = LivePipeline(session_id=conn_id, tap_mode=settings.TAP)

            # Emit initial warm-up checks
            for ch in ALL_CHECKS:
                await send_json(CheckEvent(type="check", id=ch, state="running").model_dump())
            await send_json(RiskEvent(type="risk", value=0.0).model_dump())

            async def pipeline_worker():
                """Extract windows from windower and run real-time analysis."""
                while True:
                    await asyncio.sleep(1.0)
                    win = windower.extract_current_window()
                    if win is not None:
                        try:
                            checks, risk, action, findings, evidence = pipeline.process_window(win)
                            # 1. Update check states
                            for ch, st in checks.items():
                                await send_json(CheckEvent(type="check", id=ch, state=st).model_dump())
                            # 2. Update risk score
                            await send_json(RiskEvent(type="risk", value=round(risk, 3)).model_dump())
                            # 3. Emit findings
                            for f in findings:
                                await send_json(f.model_dump())
                            # 4. Emit action event if recommended
                            if action:
                                await send_json(ActionEvent(type="action", text=action).model_dump())
                            # 5. Emit visual evidence if available
                            if evidence:
                                await send_json(evidence.model_dump())
                        except Exception as pe:
                            logger.error("Error in pipeline window processing: %s", pe, exc_info=True)

            pipeline_task = asyncio.create_task(pipeline_worker())

        # 5. Ingest loop: handle binary chunks, meta, challenge, clock
        while True:
            if time.time() - session_start_time > max_duration_sec:
                break

            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            msg_bytes = message.get("bytes")
            msg_text = message.get("text")

            if msg_bytes:
                if len(msg_bytes) > 2 * 1024 * 1024:
                    logger.warning("Incoming binary chunk exceeded 2MB limit; dropping")
                    continue
                if decoder:
                    decoder.feed_chunk(msg_bytes)
                continue

            if msg_text:
                if len(msg_text) > 2 * 1024 * 1024:
                    logger.warning("Incoming text message exceeded 2MB limit; dropping")
                    continue
                try:
                    payload = json.loads(msg_text)
                    msg_type = payload.get("type")

                    if msg_type == "clock":
                        client_time = payload.get("client_time", 0)
                        server_time = int(time.time() * 1000)
                        pong = ClockPongEvent(type="clock", server_time=server_time, echo=client_time)
                        await websocket.send_text(pong.model_dump_json())

                    elif msg_type == "meta":
                        logger.debug("Received meta event: %s", payload)
                        if pipeline:
                            pipeline.update_metadata(payload)

                    elif msg_type == "challenge":
                        logger.debug("Received challenge event: %s", payload)
                        if windower:
                            windower.register_challenge(payload)

                except Exception as e:
                    logger.warning("Error processing text frame: %s", e)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", conn_id)
    except Exception as e:
        logger.error("WebSocket exception in session %s: %s", conn_id, e)
    finally:
        if mock_engine:
            mock_engine.stop()
        if mock_task and not mock_task.done():
            mock_task.cancel()
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()
        if pipeline:
            try:
                summary_text, ai_gen = pipeline.finalize(consent_status=consent_status)
                await send_json(SummaryEvent(type="summary", text=summary_text, ai_generated=ai_gen).model_dump())
            except Exception as fe:
                logger.warning("Error finalizing session summary: %s", fe)
            pipeline.close()
        if windower:
            windower.stop()
        if decoder:
            decoder.close()
        active_connections.pop(conn_id, None)
        ACTIVE_SESSIONS.dec()
