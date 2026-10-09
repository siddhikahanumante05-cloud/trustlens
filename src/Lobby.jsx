import { useState } from 'react'
import { CFG } from './api.js'

export default function Lobby({ onStart }) {
  const [mode, setMode] = useState('webcam') // 'webcam' | 'rehearsal' | 'agent' | 'caller'
  const [roomCode, setRoomCode] = useState(() => `KYC-${Math.floor(1000 + Math.random() * 9000)}`)
  const [scenario, setScenario] = useState('fake')
  const [file, setFile] = useState(null)
  const [consent, setConsent] = useState(true)

  const handleStart = () => {
    onStart({
      mode,
      roomCode: roomCode.trim().toUpperCase(),
      scenario,
      file,
      consent,
    })
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-8 px-6 py-12">
      <header className="text-center sm:text-left">
        <div className="inline-flex items-center gap-2 rounded-full border border-ok/20 bg-emerald-50 px-3 py-1 text-xs font-semibold text-ok">
          <span className="h-2 w-2 rounded-full bg-ok animate-pulse" />
          TrustLens KYC Forensics Engine
        </div>
        <h1 className="mt-3 text-3xl font-extrabold tracking-tight text-ink sm:text-4xl">
          Real-Time Video Call Liveness & Deepfake Verification
        </h1>
        <p className="mt-2 text-base text-slate-600">
          Multimodal biometric analysis running active screen light reflection, lip coherence, and frequency-domain boundary detection.
        </p>
      </header>

      {/* Mode Navigation Tabs */}
      <div className="grid grid-cols-4 gap-1.5 rounded-xl bg-slate-200/60 p-1.5 text-xs font-semibold">
        <button
          onClick={() => setMode('webcam')}
          className={`rounded-lg py-2 transition ${
            mode === 'webcam'
              ? 'bg-white text-ink shadow-xs'
              : 'text-slate-600 hover:text-ink'
          }`}
        >
          📷 Live Webcam
        </button>
        <button
          onClick={() => setMode('rehearsal')}
          className={`rounded-lg py-2 transition ${
            mode === 'rehearsal'
              ? 'bg-white text-ink shadow-xs'
              : 'text-slate-600 hover:text-ink'
          }`}
        >
          🎬 Sample Video
        </button>
        <button
          onClick={() => setMode('agent')}
          className={`rounded-lg py-2 transition ${
            mode === 'agent'
              ? 'bg-white text-ink shadow-xs'
              : 'text-slate-600 hover:text-ink'
          }`}
        >
          🛡️ Agent Station
        </button>
        <button
          onClick={() => setMode('caller')}
          className={`rounded-lg py-2 transition ${
            mode === 'caller'
              ? 'bg-white text-ink shadow-xs'
              : 'text-slate-600 hover:text-ink'
          }`}
        >
          📱 Caller Device
        </button>
      </div>

      {/* Mode 0: Direct Live Webcam */}
      {mode === 'webcam' && (
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div>
            <h2 className="text-base font-bold text-ink">Test Your Live Camera</h2>
            <p className="text-xs text-slate-500 mt-1">
              Turns on your webcam and microphone immediately. TrustLens will stream your feed directly into the backend AI pipeline, measuring your facial texture, screen reflection, and speech synchronization.
            </p>
          </div>
          <div className="pt-2">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-ok focus:ring-ok"
              />
              <span className="text-xs text-slate-600 leading-normal">
                <strong className="text-ink">Consent for Real-Time Analysis:</strong> Allow TrustLens to analyze camera stream for liveness and deepfake indicators.
              </span>
            </label>
          </div>
        </section>
      )}

      {/* Mode 1: Agent Station */}
      {mode === 'agent' && (
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div>
            <label className="block text-sm font-bold text-ink">Room Verification Code</label>
            <p className="text-xs text-slate-500 mb-2">Share this code with the caller to connect their phone or laptop camera.</p>
            <input
              type="text"
              value={roomCode}
              onChange={(e) => setRoomCode(e.target.value.toUpperCase())}
              className="w-full rounded-xl border border-slate-300 px-4 py-2.5 font-mono text-lg font-bold tracking-wider uppercase focus:border-ok focus:outline-none"
            />
          </div>

          <div className="pt-2">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-ok focus:ring-ok"
              />
              <span className="text-xs text-slate-600 leading-normal">
                <strong className="text-ink">Explicit Caller Consent:</strong> Caller has been notified that biometric liveness, photometric skin response, and speech synchronization are recorded and analyzed for fraud prevention.
              </span>
            </label>
          </div>
        </section>
      )}

      {/* Mode 2: Caller Device */}
      {mode === 'caller' && (
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div>
            <label className="block text-sm font-bold text-ink">Enter Room Code from Agent</label>
            <p className="text-xs text-slate-500 mb-2">Type the code displayed on the verification officer's screen.</p>
            <input
              type="text"
              value={roomCode}
              placeholder="e.g. KYC-1234"
              onChange={(e) => setRoomCode(e.target.value.toUpperCase())}
              className="w-full rounded-xl border border-slate-300 px-4 py-2.5 font-mono text-lg font-bold tracking-wider uppercase focus:border-ok focus:outline-none"
            />
          </div>
          <p className="text-xs text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200">
            During the call, your screen may illuminate briefly with soft colors to verify physical presence.
          </p>
        </section>
      )}

      {/* Mode 3: Rehearsal / Offline Demo */}
      {mode === 'rehearsal' && (
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div>
            <h2 className="text-sm font-bold text-ink mb-2">Preset Simulation Scenario</h2>
            <div className="grid grid-cols-2 gap-3">
              {[
                ['real', 'Authentic Person', 'Normal camera & physical response'],
                ['fake', 'Deepfake Ingest', 'Virtual cam, bad light modulation, lip desync'],
              ].map(([id, t, s]) => (
                <button
                  key={id}
                  onClick={() => setScenario(id)}
                  aria-pressed={scenario === id}
                  className={`rounded-xl border p-4 text-left transition ${
                    scenario === id
                      ? 'border-ok bg-emerald-50/40 ring-2 ring-ok/30'
                      : 'border-slate-300 hover:border-slate-400'
                  }`}
                >
                  <div className="font-semibold text-sm text-ink">{t}</div>
                  <div className="mt-1 text-xs text-slate-500">{s}</div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-bold text-ink">Custom Video File (Optional)</label>
            <p className="text-xs text-slate-500 mb-2">Upload a recorded MP4/WebM file to play as caller video.</p>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-xs file:mr-3 file:rounded-lg file:border-0 file:bg-ink file:px-3 file:py-2 file:text-white"
            />
          </div>
        </section>
      )}

      <button
        onClick={handleStart}
        className="w-full rounded-xl bg-ok px-6 py-3.5 font-bold text-white shadow-md hover:bg-ok/90 transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ok text-base"
      >
        {mode === 'webcam'
          ? 'Start Webcam Analysis'
          : mode === 'caller'
          ? 'Join Call as Caller'
          : 'Launch Verification Call'}
      </button>

      <footer className="text-center text-xs text-slate-400">
        Backend URL: <span className="font-mono text-slate-600">{CFG.api}</span> · Device Profile: CPU-optimized (8GB)
      </footer>
    </main>
  )
}
