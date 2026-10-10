"""FastAPI main entrypoint for TrustLens backend."""
import os
import shutil
import logging
import json
import time
from contextlib import asynccontextmanager
from typing import Optional
import httpx
from fastapi import FastAPI, WebSocket, Query, UploadFile, File, Form, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST

from app.config import settings
from app.schemas import (
    SessionCreateResponse,
    HealthResponse,
    CaseRecord,
)
from app.session import session_service
from app.metrics import get_prometheus_metrics
from app.ws import handle_ws_analyze
from app.decoder import ChunkDecoder
from app.windower import Windower
from app.pipeline import LivePipeline
from app.store import case_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trustlens.main")

# State tracking for runtime dependencies
runtime_state = {
    "ollama_available": False,
    "qwen_model_found": False,
    "stubs_in_use": True,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup checks
    logger.info("Starting TrustLens backend on %s:%s", settings.HOST, settings.PORT)
    logger.info("Memory profile: %s | Device: %s | Threads: %s", settings.MEMORY_PROFILE, settings.DEVICE, settings.ORT_THREADS)

    # Prune expired case folders
    evicted = case_store.cleanup_expired_cases()
    if evicted:
        logger.info("Evicted %d expired cases during startup.", evicted)

    # Check Ollama status
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                runtime_state["ollama_available"] = True
                models_info = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models_info]
                runtime_state["qwen_model_found"] = any(settings.OLLAMA_MODEL in name for name in model_names)
                logger.info(
                    "Ollama detected at %s (qwen2.5:0.5b found: %s)",
                    settings.OLLAMA_URL,
                    runtime_state["qwen_model_found"],
                )
    except Exception:
        logger.info("Local Ollama not reachable at %s. Defaulting to template narrator.", settings.OLLAMA_URL)

    # Check neural model files presence without blocking event loop
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    has_video = os.path.exists(os.path.join(models_dir, "edgenet.pt")) or os.path.exists(os.path.join(models_dir, "edgenet_int8.onnx"))
    has_audio = os.path.exists(os.path.join(models_dir, "wavlm_head.pt")) or os.path.exists(os.path.join(models_dir, "wavlm_head_int8.onnx"))
    runtime_state["stubs_in_use"] = not (has_video and has_audio)

    yield
    logger.info("Shutting down TrustLens backend.")


app = FastAPI(
    title="TrustLens Analysis API",
    description="Real-time multi-signal deepfake detection engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root status and documentation endpoint."""
    return {
        "service": "TrustLens Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "/health",
            "session": "POST /session",
            "file_analysis": "POST /analyze/file",
            "websocket_stream": "WS /ws/analyze",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check verifying models and ffmpeg availability."""
    ffmpeg_found = shutil.which("ffmpeg") is not None
    return HealthResponse(
        status="ok" if ffmpeg_found else "degraded",
        models_loaded=True,
        ffmpeg_present=ffmpeg_found,
        stubs_in_use=runtime_state["stubs_in_use"],
        version="0.1.0-stub-0",
        narrator_mode=settings.NARRATOR_MODE,
        ollama_available=runtime_state["ollama_available"] and runtime_state["qwen_model_found"],
    )


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(content=get_prometheus_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.post("/session", response_model=SessionCreateResponse)
async def create_session():
    """Generate a challenge session with cryptographic nonce, b/p/m phrase, and HMAC light sequence."""
    return session_service.create_session()


@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    meta_json: Optional[str] = Form(None),
):
    """
    Direct file upload analysis for rehearsal mode, offline demos, and benchmark replay.
    Supports video plus optional metadata sidecar (max 50 MB).
    Uses the exact same decoder, windower, and pipeline code as the live call.
    """
    max_bytes = 50 * 1024 * 1024  # 50 MB
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds maximum size of 50 MB",
        )

    session_id = f"file_{int(time.time())}_{file.filename}"
    decoder = ChunkDecoder(clip_mode=True)
    pipeline = LivePipeline(session_id=session_id, tap_mode=settings.TAP)

    try:
        decoder.feed_chunk(content)
        windower = Windower(decoder=decoder, stride_sec=1.0)

        # Parse optional metadata
        if meta_json:
            try:
                meta_obj = json.loads(meta_json)
                if isinstance(meta_obj, list):
                    for item in meta_obj:
                        windower.register_challenge(item)
                elif isinstance(meta_obj, dict):
                    windower.register_challenge(meta_obj)
                    pipeline.update_metadata(meta_obj)
            except Exception as e:
                logger.warning("Failed to parse metadata sidecar: %s", e)

        # Extract and analyze all available temporal windows
        windows_processed = 0
        latest_checks = {}

        windows = windower.extract_all_windows()
        for win in windows:
            checks, risk, action, findings, evidence = pipeline.process_window(win)
            latest_checks = checks
            windows_processed += 1

        summary_text, ai_gen = pipeline.finalize(consent_status="consented")
        stats = decoder.get_buffer_stats()

        return {
            "status": "analyzed",
            "session_id": session_id,
            "filename": file.filename,
            "bytes": len(content),
            "total_frames_decoded": stats["total_frames_decoded"],
            "total_audio_samples": stats["total_audio_samples_decoded"],
            "windows_analyzed": windows_processed,
            "risk_score": round(pipeline.latest_risk, 4),
            "category": pipeline.latest_category,
            "checks": latest_checks,
            "findings": [f.model_dump() for f in pipeline.all_findings],
            "summary": summary_text,
            "ai_generated_summary": ai_gen,
            "stubs_in_use": pipeline.runtime.stubs_in_use,
        }
    finally:
        decoder.close()
        windower.stop()
        pipeline.close()


@app.get("/cases/{session_id}", response_model=CaseRecord)
async def get_case(session_id: str):
    """Retrieve audit case record and evidence JSON."""
    case_data = case_store.load_case(session_id)
    if not case_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case record '{session_id}' not found.",
        )

    return CaseRecord(
        session_id=case_data.get("session_id", session_id),
        created_at=case_data.get("created_at", 0),
        consent_status=case_data.get("consent_status", "consented"),
        stubs_in_use=case_data.get("stubs_in_use", True),
        risk_score=case_data.get("risk_score", 0.0),
        findings=case_data.get("findings", []),
        summary=case_data.get("summary", ""),
        ai_generated_summary=case_data.get("ai_generated_summary", False),
    )


@app.websocket("/ws/analyze")
async def ws_analyze_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    mock: Optional[str] = Query(None),
    speed: float = Query(1.0),
):
    """WebSocket endpoint for real-time video/audio analysis."""
    await handle_ws_analyze(websocket=websocket, session_id=session_id, mock=mock, speed=speed)


@app.websocket("/ws/caller/{session_id}")
async def ws_caller_endpoint(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket endpoint for caller-side direct stream and challenge relay (TAP=caller)."""
    await handle_ws_analyze(websocket=websocket, session_id=session_id, mock=None)
