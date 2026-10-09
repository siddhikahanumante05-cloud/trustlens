// ---------- Backend open ends (see README for the full contract) ----------
export const CFG = {
  mock: import.meta.env.VITE_USE_MOCK !== 'false',
  api: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  ws: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
}

export const CHECKS = [
  { id: 'source', label: 'Camera source' },
  { id: 'light', label: 'Reaction to light' },
  { id: 'face', label: 'Face texture' },
  { id: 'voice', label: 'Voice' },
  { id: 'lips', label: 'Lips match sound' },
]

/**
 * Start analysing the remote person's stream.
 * onEvent receives:
 *  { type:'check',   id, state:'running'|'ok'|'warn'|'bad' }
 *  { type:'finding', id, severity:'low'|'medium'|'high', title, detail, t }
 *  { type:'risk',    value: 0..1 }
 * Returns a stop() function.
 */
export function startAnalysis({ stream, scenario, onEvent }) {
  return CFG.mock ? runMock(scenario, onEvent) : runBackend(stream, onEvent)
}

// REAL BACKEND: sends 1-second video/audio chunks over a WebSocket, receives events.
function runBackend(stream, onEvent) {
  const ws = new WebSocket(`${CFG.ws}/ws/analyze`)
  let rec
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'start', client_time: Date.now() }))
    rec = new MediaRecorder(stream, { mimeType: 'video/webm' })
    rec.ondataavailable = (e) => e.data.size && ws.readyState === 1 && ws.send(e.data)
    rec.start(1000)
  }
  ws.onmessage = (m) => { try { onEvent(JSON.parse(m.data)) } catch { /* ignore */ } }
  return () => { rec?.state !== 'inactive' && rec?.stop(); ws.close() }
}

// MOCK ENGINE: lets you demo the UI with no backend.
const FAKE_SCRIPT = [
  [500, { type: 'check', id: 'source', state: 'running' }],
  [3000, { type: 'check', id: 'source', state: 'bad' }],
  [3100, { type: 'finding', id: 'source', severity: 'high', title: 'Video may not come from a real camera', detail: 'The camera name looks like "OBS Virtual Camera".' }],
  [3100, { type: 'risk', value: 0.35 }],
  [5000, { type: 'check', id: 'light', state: 'bad' }],
  [5100, { type: 'finding', id: 'light', severity: 'high', title: 'Face did not react to screen light', detail: 'A real face reflects changing colors. This one stayed the same.' }],
  [5100, { type: 'risk', value: 0.58 }],
  [7000, { type: 'check', id: 'lips', state: 'bad' }],
  [7100, { type: 'finding', id: 'lips', severity: 'medium', title: 'Mouth does not match the sound', detail: 'Lips stayed open on the "P" in "Paper" at 1.2 s.' }],
  [7100, { type: 'risk', value: 0.74 }],
  [9000, { type: 'check', id: 'face', state: 'warn' }],
  [9100, { type: 'finding', id: 'face', severity: 'medium', title: 'Skin looks too smooth near the hairline', detail: 'Blending marks found where the face meets the hair.' }],
  [9100, { type: 'risk', value: 0.86 }],
  [11000, { type: 'check', id: 'voice', state: 'warn' }],
  [11100, { type: 'finding', id: 'voice', severity: 'low', title: 'Voice may be computer-made', detail: 'Sound pattern is close to cloned voices (low confidence).' }],
  [11100, { type: 'risk', value: 0.91 }],
]
const REAL_SCRIPT = [
  [3000, { type: 'check', id: 'source', state: 'ok' }], [4500, { type: 'check', id: 'light', state: 'ok' }],
  [6000, { type: 'check', id: 'face', state: 'ok' }], [7500, { type: 'check', id: 'voice', state: 'ok' }],
  [9000, { type: 'check', id: 'lips', state: 'ok' }], [9000, { type: 'risk', value: 0.07 }],
]

function runMock(scenario, onEvent) {
  CHECKS.forEach((c) => onEvent({ type: 'check', id: c.id, state: 'running' }))
  onEvent({ type: 'risk', value: 0.1 })
  const script = scenario === 'fake' ? FAKE_SCRIPT : REAL_SCRIPT
  const ids = script.map(([ms, e]) => setTimeout(() => onEvent({ ...e, t: ms / 1000 }), ms))
  return () => ids.forEach(clearTimeout)
}
