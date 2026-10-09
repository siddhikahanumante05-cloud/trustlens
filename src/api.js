// ---------- Backend API & WebSocket Client ----------
export const CFG = {
  mock: import.meta.env.VITE_USE_MOCK === 'true',
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
 * Fetch a new challenge session with cryptographic nonce, phrase, and light sequence.
 */
export async function createSession() {
  try {
    const res = await fetch(`${CFG.api}/session`, { method: 'POST' })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch (err) {
    console.warn('Backend /session unreachable, using local fallback:', err)
    return {
      session_id: `local_${Date.now()}`,
      nonce: 'demo_nonce',
      phrase: 'Peter Piper picked a peck of pickled peppers',
      light_sequence: [
        { color: '#ffffff', ms: 350 },
        { color: '#38bdf8', ms: 400 },
        { color: '#4ade80', ms: 400 },
        { color: '#f43f5e', ms: 400 },
      ],
      expires_at: Date.now() + 300000,
    }
  }
}

/**
 * Start analysing the remote person's stream.
 * onEvent receives:
 *  { type:'check',   id, state:'running'|'ok'|'warn'|'bad' }
 *  { type:'finding', id, severity:'low'|'medium'|'high', title, detail, t }
 *  { type:'risk',    value: 0..1 }
 *  { type:'action',  text }
 *  { type:'summary', text, ai_generated }
 *  { type:'evidence', kind, url, data }
 * Returns an object with:
 *  stop(), sendMeta(meta), sendChallenge(challenge)
 */
export function startAnalysis({ stream, scenario, onEvent, consent = true, sessionId }) {
  if (CFG.mock) {
    return runMock(scenario, onEvent)
  }
  return runBackend({ stream, onEvent, consent, sessionId })
}

// REAL BACKEND: sends 1-second video/audio chunks over a WebSocket, receives events.
function runBackend({ stream, onEvent, consent, sessionId }) {
  const wsUrl = sessionId ? `${CFG.ws}/ws/analyze?session_id=${sessionId}` : `${CFG.ws}/ws/analyze`
  const ws = new WebSocket(wsUrl)
  const pendingMessages = []
  const sendOrQueue = (msg) => {
    const payload = typeof msg === 'string' ? msg : JSON.stringify(msg)
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(payload)
    } else {
      pendingMessages.push(payload)
    }
  }

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: 'start',
      client_time: Date.now(),
      consent: Boolean(consent),
      session_id: sessionId,
    }))

    while (pendingMessages.length > 0) {
      const msg = pendingMessages.shift()
      ws.send(msg)
    }

    if (stream && stream.getTracks().length > 0) {
      try {
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
          ? 'video/webm;codecs=vp8,opus'
          : 'video/webm'
        rec = new MediaRecorder(stream, { mimeType })
        rec.ondataavailable = (e) => {
          if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(e.data)
          }
        }
        rec.start(1000)
      } catch (e) {
        console.warn('MediaRecorder error:', e)
      }
    }
  }

  ws.onmessage = (m) => {
    try {
      const data = JSON.parse(m.data)
      onEvent(data)
    } catch {
      // ignore
    }
  }

  return {
    stop: () => {
      if (rec && rec.state !== 'inactive') {
        rec.stop()
      }
      ws.close()
    },
    sendMeta: (meta) => {
      sendOrQueue({ type: 'meta', ...meta })
    },
    sendChallenge: (challenge) => {
      sendOrQueue({ type: 'challenge', ...challenge })
    },
  }
}

// MOCK ENGINE: lets you demo the UI with no backend.
const FAKE_SCRIPT = [
  [500, { type: 'check', id: 'source', state: 'running' }],
  [2500, { type: 'check', id: 'source', state: 'bad' }],
  [2600, { type: 'finding', id: 'source', severity: 'high', title: 'Video may not come from a real camera', detail: 'The camera name looks like "OBS Virtual Camera".' }],
  [2600, { type: 'risk', value: 0.35 }],
  [4500, { type: 'check', id: 'light', state: 'bad' }],
  [4600, { type: 'finding', id: 'light', severity: 'high', title: 'Face did not react to screen light', detail: 'A real face reflects changing colors. This one stayed the same.' }],
  [4600, { type: 'risk', value: 0.65 }],
  [6500, { type: 'check', id: 'lips', state: 'bad' }],
  [6600, { type: 'finding', id: 'lips', severity: 'high', title: 'Mouth does not match the sound', detail: 'Lips stayed open on the "P" in "Paper" at 1.2 s.' }],
  [6600, { type: 'risk', value: 0.78 }],
  [6700, { type: 'action', text: 'Ask the caller to turn their head left, then say the phrase again.' }],
  [8500, { type: 'check', id: 'face', state: 'warn' }],
  [8600, { type: 'finding', id: 'face', severity: 'medium', title: 'Skin looks too smooth near the hairline', detail: 'Blending marks found where the face meets the hair.' }],
  [8600, { type: 'risk', value: 0.88 }],
]

const REAL_SCRIPT = [
  [2000, { type: 'check', id: 'source', state: 'ok' }],
  [3500, { type: 'check', id: 'light', state: 'ok' }],
  [5000, { type: 'check', id: 'face', state: 'ok' }],
  [6500, { type: 'check', id: 'voice', state: 'ok' }],
  [8000, { type: 'check', id: 'lips', state: 'ok' }],
  [8000, { type: 'risk', value: 0.08 }],
]

function runMock(scenario, onEvent) {
  CHECKS.forEach((c) => onEvent({ type: 'check', id: c.id, state: 'running' }))
  onEvent({ type: 'risk', value: 0.1 })
  const script = scenario === 'fake' ? FAKE_SCRIPT : REAL_SCRIPT
  const ids = script.map(([ms, e]) => setTimeout(() => onEvent({ ...e, t: ms / 1000 }), ms))
  return {
    stop: () => ids.forEach(clearTimeout),
    sendMeta: () => {},
    sendChallenge: () => {},
  }
}
