import { useState } from 'react'
import { CHECKS } from './api.js'

const verdict = (r) => {
  if (r < 0.40) return ['Looks authentic', 'text-ok', 'bg-ok']
  if (r < 0.70) return ['Needs closer review', 'text-warn', 'bg-warn']
  return ['Likely a deepfake', 'text-bad', 'bg-bad']
}

const SEV = {
  high: 'border-bad bg-red-50/50 text-red-900',
  medium: 'border-warn bg-amber-50/50 text-amber-900',
  low: 'border-slate-400 bg-white text-slate-800',
}

const STATE = {
  running: ['Analyzing…', 'text-slate-500 bg-slate-100'],
  ok: ['Passed', 'text-ok bg-emerald-50'],
  warn: ['Suspicious', 'text-warn bg-amber-50'],
  bad: ['Failed', 'text-bad bg-red-50'],
}

export default function Panel({
  checks,
  findings,
  risk,
  action,
  evidence,
  onTriggerLightChallenge,
  onTriggerPhraseChallenge,
}) {
  const [label, textClass, barClass] = verdict(risk)
  const running = Object.values(checks).some((s) => s === 'running')
  const [previewEvidence, setPreviewEvidence] = useState(null)

  return (
    <aside
      aria-label="Live KYC Forensics Panel"
      className="flex w-full flex-col gap-5 overflow-y-auto border-l border-slate-200 bg-paper p-5 lg:w-[400px]"
    >
      {/* 1. Header & Live Meter */}
      <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-ink">Forensics Monitor</h2>
          <span className="flex items-center gap-1.5 text-xs text-slate-500 font-medium">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                running ? 'animate-pulse bg-emerald-500' : 'bg-slate-400'
              }`}
            />
            {running ? 'Live' : 'Standby'}
          </span>
        </div>

        <p className={`mt-2 text-2xl font-extrabold ${textClass}`}>{label}</p>

        <div
          className="mt-3 h-2.5 overflow-hidden rounded-full bg-slate-100"
          role="meter"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(risk * 100)}
          aria-label="Deepfake probability"
        >
          <div
            className={`h-full ${barClass} transition-all duration-500`}
            style={{ width: `${Math.max(4, risk * 100)}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-xs text-slate-500 font-medium">
          <span>Assessed Risk: {Math.round(risk * 100)}%</span>
          <span>Threshold: 70%</span>
        </div>
      </div>

      {/* 2. Action Recommendation Card (Human Decides) */}
      {action && (
        <div className="rounded-xl border-2 border-bad bg-red-50/80 p-4 shadow-sm animate-pulse">
          <div className="flex items-center gap-2">
            <span className="text-bad font-bold text-sm uppercase tracking-wide">
              ⚠️ Action Recommended
            </span>
          </div>
          <p className="mt-1 text-sm font-medium text-red-950">{action}</p>
          <div className="mt-3 flex gap-2">
            <button
              onClick={onTriggerLightChallenge}
              className="rounded-lg bg-bad px-3 py-1.5 text-xs font-semibold text-white shadow hover:bg-bad/90"
            >
              Start Light Challenge
            </button>
            <button
              onClick={onTriggerPhraseChallenge}
              className="rounded-lg bg-white border border-bad/30 px-3 py-1.5 text-xs font-semibold text-bad hover:bg-red-100/50"
            >
              Start Phrase Challenge
            </button>
          </div>
        </div>
      )}

      {/* 3. Signal Checks Checklist */}
      <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-400">
          Biometric & Hardware Signals
        </h3>
        <ul className="divide-y divide-slate-100 text-sm">
          {CHECKS.map((c) => {
            const [sLabel, sClass] = STATE[checks[c.id] || 'running']
            return (
              <li key={c.id} className="flex items-center justify-between py-2">
                <span className="font-medium text-slate-700">{c.label}</span>
                <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${sClass}`}>
                  {sLabel}
                </span>
              </li>
            )
          })}
        </ul>
      </div>

      {/* 4. Active Challenge Controls */}
      <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-400">
          Liveness Challenges
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={onTriggerLightChallenge}
            className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left text-xs font-medium text-slate-700 hover:bg-slate-100 transition"
          >
            <span className="block font-semibold text-ink">💡 Screen Flash</span>
            Test photometric skin reflection
          </button>
          <button
            onClick={onTriggerPhraseChallenge}
            className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left text-xs font-medium text-slate-700 hover:bg-slate-100 transition"
          >
            <span className="block font-semibold text-ink">🗣️ Bilabial Phrase</span>
            Test acoustic-lip coherence
          </button>
        </div>
      </div>

      {/* 5. Explainability Findings */}
      <div className="min-h-0 flex-1">
        <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-400">
          Forensic Observations ({findings.length})
        </h3>
        {findings.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-200 bg-white/50 p-4 text-center text-xs text-slate-400">
            {running
              ? 'Monitoring stream for spatial artifacts, timing jitter, and voice synthesis…'
              : 'No anomalies detected.'}
          </div>
        ) : (
          <ul className="space-y-2.5">
            {findings.map((f, i) => (
              <li
                key={i}
                className={`reason-in rounded-xl border-l-4 p-3 shadow-xs ${
                  SEV[f.severity] || SEV.low
                }`}
              >
                <div className="flex items-baseline justify-between">
                  <p className="font-semibold text-sm">{f.title}</p>
                  <span className="text-[10px] font-bold uppercase opacity-75">
                    {f.severity}
                  </span>
                </div>
                <p className="mt-1 text-xs opacity-90 leading-relaxed">{f.detail}</p>
                {f.t != null && (
                  <p className="mt-1 text-[10px] text-slate-400">
                    Timestamp: {Math.round(f.t)}s
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* 6. Visual Evidence Heatmaps */}
      {evidence && evidence.length > 0 && (
        <div className="rounded-xl border border-slate-200/80 bg-white p-4 shadow-sm">
          <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            Visual Evidence ({evidence.length})
          </h3>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {evidence.map((ev, idx) => (
              <button
                key={idx}
                onClick={() => setPreviewEvidence(ev.url)}
                className="group relative flex-shrink-0 overflow-hidden rounded-lg border border-slate-200 hover:border-bad transition"
              >
                <img
                  src={ev.url}
                  alt="Anomaly heatmap"
                  className="h-16 w-16 object-cover"
                />
                <span className="absolute inset-0 flex items-center justify-center bg-black/40 text-[10px] font-bold text-white opacity-0 group-hover:opacity-100 transition">
                  Zoom
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Modal for Zoomed Heatmap */}
      {previewEvidence && (
        <div
          onClick={() => setPreviewEvidence(null)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-xs"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="max-w-lg rounded-2xl bg-white p-4 shadow-2xl"
          >
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-bold text-sm text-ink">Facial Texture Anomaly Heatmap</h4>
              <button
                onClick={() => setPreviewEvidence(null)}
                className="text-slate-400 hover:text-slate-600 text-sm font-bold"
              >
                ✕
              </button>
            </div>
            <img
              src={previewEvidence}
              alt="Detailed forensic heatmap"
              className="rounded-lg w-full max-h-[70vh] object-contain"
            />
          </div>
        </div>
      )}
    </aside>
  )
}
