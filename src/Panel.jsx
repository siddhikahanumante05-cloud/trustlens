import { CHECKS } from './api.js'

const verdict = (r) => (r < 0.35 ? ['Looks real', 'text-ok', 'bg-ok'] : r < 0.7 ? ['Needs a closer look', 'text-warn', 'bg-warn'] : ['Likely a deepfake', 'text-bad', 'bg-bad'])
const SEV = { high: 'border-bad', medium: 'border-warn', low: 'border-slate-400' }
const STATE = {
  running: ['Checking…', 'text-slate-500'], ok: ['Fine', 'text-ok'], warn: ['Unsure', 'text-warn'], bad: ['Problem', 'text-bad'],
}

export default function Panel({ checks, findings, risk }) {
  const [label, text, bar] = verdict(risk)
  const running = Object.values(checks).some((s) => s === 'running')
  return (
    <aside aria-label="Live analysis" className="flex w-full flex-col gap-6 overflow-y-auto border-l border-slate-200 bg-paper p-5 lg:w-[380px]">
      <div>
        <div className="flex items-baseline justify-between">
          <h2 className="text-lg font-bold">Live analysis</h2>
          <span className="text-sm text-slate-500" aria-live="polite">{running ? 'Analyzing…' : 'Done'}</span>
        </div>
        <p className={`mt-3 text-2xl font-bold ${text}`}>{label}</p>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200"
          role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(risk * 100)} aria-label="Deepfake risk">
          <div className={`h-full ${bar} transition-all duration-700`} style={{ width: `${Math.max(4, risk * 100)}%` }} />
        </div>
        <p className="mt-1 text-sm text-slate-500">Risk {Math.round(risk * 100)} out of 100</p>
      </div>

      <ul className="space-y-2 text-sm">
        {CHECKS.map((c) => {
          const [s, cls] = STATE[checks[c.id] || 'running']
          return <li key={c.id} className="flex justify-between"><span>{c.label}</span><span className={`font-medium ${cls}`}>{s}</span></li>
        })}
      </ul>

      <div className="min-h-0">
        <h3 className="mb-3 font-semibold">Why this may be fake</h3>
        {findings.length === 0 ? (
          <p className="text-sm text-slate-500">{running ? 'Listening and watching. Reasons will appear here in a few seconds.' : 'No problems found.'}</p>
        ) : (
          <ul className="space-y-3">
            {findings.map((f, i) => (
              <li key={i} className={`reason-in rounded-lg border-l-4 bg-white p-3 ${SEV[f.severity] || SEV.low}`}>
                <p className="font-medium">{f.title}</p>
                <p className="mt-1 text-sm text-slate-600">{f.detail}</p>
                {f.t != null && <p className="mt-1 text-xs text-slate-400">Found at {Math.round(f.t)} s</p>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  )
}
