import { useState } from 'react'
import { CFG } from './api.js'

export default function Lobby({ onStart }) {
  const [scenario, setScenario] = useState('fake')
  const [file, setFile] = useState(null)

  return (
    <main className="mx-auto flex min-h-screen max-w-xl flex-col justify-center gap-8 px-6 py-12">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">TrustLens</h1>
        <p className="mt-2 text-slate-600">Start a video call. We check the other person and tell you, in plain words, if they may be a deepfake.</p>
      </header>

      <section className="space-y-3">
        <h2 className="font-semibold">Who are you calling?</h2>
        <div className="grid grid-cols-2 gap-3">
          {[['real', 'A real person', 'Demo: should look safe'], ['fake', 'A deepfake', 'Demo: should raise warnings']].map(([id, t, s]) => (
            <button key={id} onClick={() => setScenario(id)} aria-pressed={scenario === id}
              className={`rounded-lg border p-4 text-left transition ${scenario === id ? 'border-ok bg-white ring-2 ring-ok/30' : 'border-slate-300 bg-white/60 hover:border-slate-400'}`}>
              <div className="font-medium">{t}</div><div className="text-sm text-slate-500">{s}</div>
            </button>
          ))}
        </div>
        <p className="text-sm text-slate-500">This choice only controls the built-in demo engine.{!CFG.mock && ' (Your backend is on, so it is ignored.)'}</p>
      </section>

      <section className="space-y-2">
        <h2 className="font-semibold">Video of the other person <span className="font-normal text-slate-500">(optional)</span></h2>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-ink file:px-3 file:py-2 file:text-white" />
        <p className="text-sm text-slate-500">Pick a clip to play as the caller. Without one, a simple placeholder face is used.</p>
      </section>

      <button onClick={() => onStart({ scenario, file })}
        className="rounded-lg bg-ok px-5 py-3 font-semibold text-white hover:bg-ok/90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ok">
        Start video call
      </button>
    </main>
  )
}
