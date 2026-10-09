# TrustLens Backend

Real-time analysis backend for the TrustLens video-call KYC verification system. Analyzes received caller video and audio within 3–4 seconds to detect synthetic face swaps, virtual cameras, replay attacks, and voice cloning.

---

## 1. Quick Start

### Local Development (Python)

```bash
# From workspace root or backend/
cd backend
py -m pip install -r requirements.txt

# Run backend
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (CPU)

```bash
# Build and run with docker-compose from repository root
docker compose up backend
```

---

## 2. API & WebSocket Contract

### REST Endpoints

- **`POST /session`**: Generates a cryptographic nonce, a random 3-word phrase starting with b/p/m bilabial phonemes (e.g. *Banana Pocket Mountain*), and a 6–8 step HMAC light sequence starting with a `#ffffff` sync pulse.
- **`GET /health`**: Verifies system status, ffmpeg presence, model loading, and reports `stubs_in_use: true`.
- **`GET /metrics`**: Prometheus text metrics format (`trustlens_stubs_in_use{version="stub-0"} 1`, active sessions, dropped windows, pipeline latencies).
- **`POST /analyze/file`**: Multipart video upload (+ optional JSON metadata) for rehearsal mode and benchmark replays (max 50 MB).
- **`GET /cases/{session_id}`**: Retrieves audit case JSON and evidence features.

### WebSocket Endpoint: `ws://localhost:8000/ws/analyze`

Query Parameters:
- `?mock=fake` or `?mock=real`: Runs scripted deterministic timelines for UI testing.
- `?session_id=<id>`: Binds analysis to pre-generated challenge session.
- `?speed=<float>`: Acceleration multiplier (useful for fast contract testing).

#### Inbound Messages (Client -> Server)
```json
// 1. Initial Handshake & Biometric Consent
{"type":"start", "client_time": 1720000000000, "session_id": "abc...", "consent": true}

// 2. Binary chunks (1-second WebM fragments)
<binary data>

// 3. Metadata & Hardware Diagnostics
{"type":"meta", "camera_label": "HD Pro Webcam C920", "frame_intervals_ms": [33.3, 33.4, ...]}

// 4. Challenges (Light sequence & Phrase)
{"type":"challenge", "kind":"light", "t0": 1720000005000, "sequence":[{"color":"#ffffff","ms":350}, ...]}
{"type":"challenge", "kind":"phrase", "expected":"Banana Pocket Mountain", "t0": 1720000005000}

// 5. Clock Sync Ping
{"type":"clock", "client_time": 1720000001234}
```

#### Outbound Events (Server -> Client)
```json
// Channel Check Updates
{"type":"check", "id":"source|light|face|voice|lips", "state":"running|ok|warn|bad"}

// Plain-English Findings
{"type":"finding", "id":"source", "severity":"high", "title":"Video may not come from a real camera", "detail":"The camera name looks like \"OBS Virtual Camera\".", "t": 3.1}

// Calibrated Risk Score (0.0 to 1.0)
{"type":"risk", "value": 0.86}

// Analyst Recommendation (on Likely Deepfake >= 0.70)
{"type":"action", "text": "Ask the caller to turn their head left, then say the phrase again."}

// Case Narrative Summary
{"type":"summary", "text": "High probability of deepfake detected (Risk: 86%)...", "ai_generated": false}

// Clock Sync Pong
{"type":"clock", "server_time": 1720000001250, "echo": 1720000001234}
```

---

## 3. Privacy & Caller Consent

Biometric analysis is strictly gated on caller consent:
- Inbound `start` message must carry `"consent": true`.
- If missing or `false`, the server rejects with:
  `{"type":"error", "code":"CONSENT_REQUIRED", "message":"Analysis requires explicit consent from caller."}` and terminates the socket.
- **Demo Mode Bypass**: Setting `DEMO_BYPASS_CONSENT=true` when `ENV=demo` logs the bypass and flags the case as `"bypassed"`.

---

## 4. Stub Models Policy (Milestone 1–3)

While AI weights are undergoing training in Milestone 4:
- All models operate behind `stub-0` interfaces.
- `/health`, `/metrics`, and `/cases/{id}` explicitly export `stubs_in_use: true`.
- The benchmark harness rejects accuracy reporting whenever stubs are active.

---

## 5. Running Tests

```bash
# Run contract and session unit test suite
py -m pytest backend/tests -v
```
