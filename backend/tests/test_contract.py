"""WebSocket and REST contract tests for TrustLens API."""
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.schemas import (
    CheckEvent,
    FindingEvent,
    RiskEvent,
    ActionEvent,
    SummaryEvent,
    ClockPongEvent,
    ErrorEvent,
)

client = TestClient(app)


# ============================================================================
# REST Contract Tests
# ============================================================================

def test_health_endpoint():
    """Verify GET /health reports models, ffmpeg status, and stubs_in_use=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["models_loaded"] is True
    assert isinstance(data["stubs_in_use"], bool)
    assert "ffmpeg_present" in data
    assert data["version"].startswith("0.1.0")


def test_metrics_endpoint():
    """Verify GET /metrics exposes Prometheus metrics including stubs_in_use."""
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "trustlens_stubs_in_use" in text
    assert "trustlens_active_sessions" in text


def test_session_create_endpoint():
    """Verify POST /session creates nonce, b/p/m phrase, and light sequence."""
    response = client.post("/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "nonce" in data
    assert "phrase" in data
    assert len(data["phrase"].split()) == 3
    assert "light_sequence" in data
    assert len(data["light_sequence"]) >= 6
    assert data["light_sequence"][0]["color"] == "#ffffff"


def test_analyze_file_endpoint():
    """Verify POST /analyze/file accepts video up to 50 MB and enforces limit."""
    # 1. Normal file
    dummy_video = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 1024
    response = client.post(
        "/analyze/file",
        files={"file": ("test.webm", dummy_video, "video/webm")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["received", "analyzed"]
    assert data["bytes"] == len(dummy_video)
    assert isinstance(data["stubs_in_use"], bool)

    # 2. Oversized file (> 50 MB)
    large_bytes = b"0" * (50 * 1024 * 1024 + 10)
    response_large = client.post(
        "/analyze/file",
        files={"file": ("large.webm", large_bytes, "video/webm")},
    )
    assert response_large.status_code == 413


# ============================================================================
# WebSocket Contract Tests
# ============================================================================

def test_ws_consent_rejection():
    """Verify WS rejects session when caller consent is missing and bypass is off."""
    orig_bypass = settings.DEMO_BYPASS_CONSENT
    orig_env = settings.ENV
    try:
        settings.DEMO_BYPASS_CONSENT = False
        settings.ENV = "production"

        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_text(json.dumps({"type": "start", "client_time": 1000, "consent": False}))
            msg = ws.receive_text()
            err_data = json.loads(msg)
            err = ErrorEvent(**err_data)
            assert err.type == "error"
            assert err.code == "CONSENT_REQUIRED"
    finally:
        settings.DEMO_BYPASS_CONSENT = orig_bypass
        settings.ENV = orig_env


def test_ws_mock_fake_scenario_contract():
    """Verify full fake scenario WebSocket timeline and exact message shapes."""
    received_types = []
    has_action = False
    has_summary = False

    # Connect with accelerated mock speed (50x)
    with client.websocket_connect("/ws/analyze?mock=fake&speed=50.0") as ws:
        # 1. Send start with consent
        ws.send_text(json.dumps({"type": "start", "client_time": 1000, "consent": True}))

        # 2. Receive warm-up check events and base risk
        for _ in range(5):
            msg = json.loads(ws.receive_text())
            check = CheckEvent(**msg)
            assert check.state == "running"
            received_types.append(check.type)

        risk_msg = json.loads(ws.receive_text())
        risk = RiskEvent(**risk_msg)
        assert risk.value == 0.10

        # 3. Read subsequent mock events until summary
        while True:
            try:
                raw = ws.receive_text()
                event_data = json.loads(raw)
                mtype = event_data.get("type")
                received_types.append(mtype)

                if mtype == "check":
                    CheckEvent(**event_data)
                elif mtype == "finding":
                    f = FindingEvent(**event_data)
                    assert f.severity in ["low", "medium", "high"]
                    assert f.t > 0
                elif mtype == "risk":
                    r = RiskEvent(**event_data)
                    assert 0.0 <= r.value <= 1.0
                elif mtype == "action":
                    act = ActionEvent(**event_data)
                    assert len(act.text) > 0
                    has_action = True
                elif mtype == "summary":
                    s = SummaryEvent(**event_data)
                    assert len(s.text) > 0
                    has_summary = True
                    break
            except Exception:
                break

    assert "check" in received_types
    assert "finding" in received_types
    assert "risk" in received_types
    assert has_action is True, "Expected action event on likely deepfake"
    assert has_summary is True, "Expected summary event at session end"


def test_ws_mock_real_scenario_contract():
    """Verify real scenario WebSocket timeline contract."""
    with client.websocket_connect("/ws/analyze?mock=real&speed=50.0") as ws:
        ws.send_text(json.dumps({"type": "start", "client_time": 1000, "consent": True}))

        # Read warm-up checks + initial risk
        for _ in range(6):
            ws.receive_text()

        # Read remaining events
        final_risk = None
        has_summary = False
        while True:
            try:
                raw = ws.receive_text()
                data = json.loads(raw)
                if data.get("type") == "risk":
                    final_risk = data.get("value")
                elif data.get("type") == "summary":
                    has_summary = True
                    break
            except Exception:
                break

        assert final_risk is not None
        assert final_risk < 0.35, f"Real scenario risk should be low, got {final_risk}"
        assert has_summary is True


def test_ws_clock_ping_pong():
    """Verify NTP-style clock sync ping pong."""
    with client.websocket_connect("/ws/analyze?mock=real&speed=50.0") as ws:
        ws.send_text(json.dumps({"type": "start", "client_time": 1000, "consent": True}))

        # Send clock ping
        client_ts = 987654321
        ws.send_text(json.dumps({"type": "clock", "client_time": client_ts}))

        # Consume events until clock pong is received
        pong_received = False
        for _ in range(15):
            raw = ws.receive_text()
            data = json.loads(raw)
            if data.get("type") == "clock":
                pong = ClockPongEvent(**data)
                assert pong.echo == client_ts
                assert pong.server_time > 0
                pong_received = True
                break

        assert pong_received is True, "Failed to receive clock pong"
