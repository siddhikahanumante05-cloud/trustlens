import { useEffect, useState } from 'react'

/**
 * Caller-side full-screen active illumination challenge overlay.
 * Displays randomized soft color steps with photosensitivity disclaimer
 * and instant skip/dismiss button.
 */
export default function FlashOverlay({ sequence, onComplete, onSkip }) {
  const [stepIndex, setStepIndex] = useState(0)

  useEffect(() => {
    if (!sequence || sequence.length === 0) {
      onComplete?.()
      return
    }

    if (stepIndex >= sequence.length) {
      onComplete?.()
      return
    }

    const currentStep = sequence[stepIndex]
    const timer = setTimeout(() => {
      setStepIndex((idx) => idx + 1)
    }, currentStep.ms || 350)

    return () => clearTimeout(timer)
  }, [stepIndex, sequence, onComplete])

  if (!sequence || stepIndex >= sequence.length) return null

  const currentColor = sequence[stepIndex]?.color || '#ffffff'

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col justify-between p-6 transition-colors duration-150"
      style={{ backgroundColor: currentColor }}
      role="alertdialog"
      aria-modal="true"
      aria-label="Screen illumination challenge"
    >
      <div className="flex items-center justify-between rounded-xl bg-black/70 px-4 py-2 text-white shadow-lg backdrop-blur">
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 animate-ping rounded-full bg-red-400" />
          <span className="text-xs font-semibold uppercase tracking-wider">
            Verification Challenge in Progress
          </span>
        </div>
        <button
          onClick={onSkip}
          className="rounded-lg bg-white/20 px-3 py-1 text-xs font-medium text-white transition hover:bg-white/30"
        >
          Skip Challenge
        </button>
      </div>

      <div className="mx-auto max-w-md rounded-2xl bg-black/75 p-6 text-center text-white shadow-2xl backdrop-blur">
        <h3 className="text-lg font-bold">Please Face Your Screen</h3>
        <p className="mt-2 text-sm text-slate-300">
          This test shines a soft, randomized sequence of colors to detect screen replays and synthetic video.
        </p>
        <p className="mt-4 text-xs text-amber-300">
          ⚠️ Photosensitivity note: Flashes occur at safe rates (≤ 3 changes/sec). Click Skip above if you are light sensitive.
        </p>
      </div>

      <div className="flex justify-center text-xs font-medium text-black/60">
        Step {stepIndex + 1} of {sequence.length}
      </div>
    </div>
  )
}
