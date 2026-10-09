import { useEffect, useRef, useState } from 'react'
import { CHECKS, startAnalysis } from './api.js'
import Panel from './Panel.jsx'

// Placeholder "other person": an animated face drawn on a canvas.
function placeholderStream() {
  const c = document.createElement('canvas'); c.width = 640; c.height = 480
  const x = c.getContext('2d'); let t = 0, on = true
  const draw = () => {
    if (!on) return
    t += 0.05
    x.fillStyle = '#1f2733'; x.fillRect(0, 0, 640, 480)
    x.fillStyle = '#9aa7b8'; x.beginPath(); x.arc(320, 220, 110, 0, 7); x.fill()
    x.fillStyle = '#14181f'; x.beginPath(); x.arc(280, 200, 10, 0, 7); x.arc(360, 200, 10, 0, 7); x.fill()
    x.fillRect(290, 270, 60, 4 + Math.abs(Math.sin(t * 3)) * 22)
    requestAnimationFrame(draw)
  }
  draw()
  const s = c.captureStream(25); s.stop = () => { on = false }
  return s
}

async function remoteFrom(file) {
  if (!file) return placeholderStream()
  const v = document.createElement('video')
  v.src = URL.createObjectURL(file); v.loop = true; v.playsInline = true
  await v.play()
  return (v.captureStream || v.mozCaptureStream).call(v)
}

function Tile({ stream, muted, className = '', label }) {
  const ref = useRef(null)
  useEffect(() => { if (ref.current) ref.current.srcObject = stream || null }, [stream])
  return (
    <div className={`relative overflow-hidden bg-black ${className}`}>
      <video ref={ref} autoPlay playsInline muted={muted} className="h-full w-full object-cover" />
      <span className="absolute left-3 top-3 rounded bg-black/55 px-2 py-1 text-xs text-white">{label}</span>
    </div>
  )
}

export default function Call({ setup, onEnd }) {
  const [self, setSelf] = useState(null)
  const [remote, setRemote] = useState(null)
  const [err, setErr] = useState('')
  const [mic, setMic] = useState(true)
  const [cam, setCam] = useState(true)
  const [secs, setSecs] = useState(0)
  const [checks, setChecks] = useState(() => Object.fromEntries(CHECKS.map((c) => [c.id, 'running'])))
  const [findings, setFindings] = useState([])
  const [risk, setRisk] = useState(0)

  useEffect(() => {
    let stops = [], dead = false
    ;(async () => {
      try {
        const me = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        const other = await remoteFrom(setup.file)
        if (dead) return
        setSelf(me); setRemote(other)
        const stopAnalysis = startAnalysis({
          stream: other, scenario: setup.scenario,
          onEvent: (e) => {
            if (e.type === 'check') setChecks((c) => ({ ...c, [e.id]: e.state }))
            if (e.type === 'finding') setFindings((f) => [...f, e])
            if (e.type === 'risk') setRisk(e.value)
          },
        })
        stops = [() => me.getTracks().forEach((t) => t.stop()), () => other.stop?.(), stopAnalysis]
      } catch (e) { setErr(e.message || 'Could not start the call.') }
    })()
    const timer = setInterval(() => setSecs((s) => s + 1), 1000)
    return () => { dead = true; clearInterval(timer); stops.forEach((s) => s()) }
  }, [setup])

  const toggle = (kind, on, set) => { self?.getTracks().filter((t) => t.kind === kind).forEach((t) => (t.enabled = !on)); set(!on) }
  const mmss = `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`
  const btn = 'rounded-full px-4 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white'

  return (
    <div className="flex h-screen flex-col lg:flex-row">
      <section className="relative flex min-h-[55vh] flex-1 flex-col bg-stage p-3 lg:p-4">
        {err ? (
          <div className="m-auto max-w-sm text-center text-white">
            <p className="font-semibold">We could not start the call</p>
            <p className="mt-2 text-sm text-slate-300">{err}. Allow camera and microphone access in your browser, then try again.</p>
            <button onClick={onEnd} className={`${btn} mt-4 bg-white text-ink`}>Back</button>
          </div>
        ) : (
          <>
            <div className="relative flex-1 overflow-hidden rounded-2xl">
              <Tile stream={remote} className="h-full w-full" label={`Caller · ${mmss}`} />
              <Tile stream={self} muted className="absolute bottom-3 right-3 h-28 w-40 rounded-xl border border-white/30" label="You" />
            </div>
            <div className="mt-3 flex justify-center gap-3">
              <button onClick={() => toggle('audio', mic, setMic)} className={`${btn} ${mic ? 'bg-white/15 text-white' : 'bg-white text-ink'}`}>{mic ? 'Mute' : 'Unmute'}</button>
              <button onClick={() => toggle('video', cam, setCam)} className={`${btn} ${cam ? 'bg-white/15 text-white' : 'bg-white text-ink'}`}>{cam ? 'Camera off' : 'Camera on'}</button>
              <button onClick={onEnd} className={`${btn} bg-bad text-white`}>End call</button>
            </div>
          </>
        )}
      </section>
      <Panel checks={checks} findings={findings} risk={risk} />
    </div>
  )
}
