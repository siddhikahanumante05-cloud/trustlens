# TrustLens UI (Vite + React + Tailwind v4)

Minimal video-call screen. The caller's video is analysed by a backend, and the reasons pop up on the right.

## Run
```
npm install
cp .env.example .env
npm run dev
```
Open the URL Vite prints. Allow camera and microphone.

## How it works
1. **Lobby:** pick "A real person" or "A deepfake" (only drives the mock engine). Optionally upload a video to play as the caller.
2. **Call:** you (webcam) + the caller (clip or placeholder). The caller's stream is analysed.
3. **Right panel:** risk verdict, 5 checks, and "Why this may be fake" cards that appear as the backend finds them.

## Connect your backend (the open ends)
Set `VITE_USE_MOCK=false` in `.env`. Edit `src/api.js` (`runBackend`) if your protocol differs.

- Client opens `WS {VITE_WS_URL}/ws/analyze`
- Client sends: one JSON text message `{ "type": "start", "client_time": 123 }`, then **1-second binary chunks** (`video/webm`, video + audio of the caller).
- Server sends JSON text messages:
```json
{ "type": "check",   "id": "source|light|face|voice|lips", "state": "running|ok|warn|bad" }
{ "type": "finding", "id": "lips", "severity": "low|medium|high", "title": "Mouth does not match the sound", "detail": "Lips stayed open on the P in Paper at 1.2 s.", "t": 7.1 }
{ "type": "risk",    "value": 0.74 }
```
Map these to TrustLens v2 channels: `source`=E1, `light`=E2, `face`=E3/E4, `voice`=E5, `lips`=E6. `risk` = fused risk (0 to 1). Verdict bands: under 0.35 real, 0.35 to 0.7 closer look, over 0.7 likely deepfake.

## Files
- `src/api.js`: config, check list, mock engine, backend client (the one file your backend teammate edits)
- `src/Lobby.jsx`, `src/Call.jsx`, `src/Panel.jsx`: the three screens/parts

## Not included yet (easy next steps)
- Real two-person calls (WebRTC, for example PeerJS). Replace `remoteFrom()` in `Call.jsx` with the remote peer's stream.
- Random phrase + light-flash challenge UI, heatmap evidence frames, LLM summary card.
