import { useEffect, useRef, useState } from 'react'
import Peer from 'peerjs'
import { CHECKS, startAnalysis, createSession } from './api.js'
import Panel from './Panel.jsx'
import FlashOverlay from './FlashOverlay.jsx'

// Animated fallback placeholder face drawn on a canvas
function placeholderStream() {
  const c = document.createElement('canvas')
  c.width = 640
  c.height = 480
  const x = c.getContext('2d')
  let t = 0
  let on = true
  const draw = () => {
    if (!on) return
    t += 0.05
    x.fillStyle = '#1f2733'
    x.fillRect(0, 0, 640, 480)
    x.fillStyle = '#9aa7b8'
    x.beginPath()
    x.arc(320, 220, 110, 0, 7)
    x.fill()
    x.fillStyle = '#14181f'
    x.beginPath()
    x.arc(280, 200, 10, 0, 7)
    x.arc(360, 200, 10, 0, 7)
    x.fill()
    x.fillRect(290, 270, 60, 4 + Math.abs(Math.sin(t * 3)) * 22)
    requestAnimationFrame(draw)
  }
  draw()
  const s = c.captureStream(25)
  s.stop = () => {
    on = false
  }
  return s
}

async function remoteFrom(file) {
  if (!file) return placeholderStream()
  const v = document.createElement('video')
  v.src = URL.createObjectURL(file)
  v.loop = true
  v.playsInline = true
  await v.play()
  return (v.captureStream || v.mozCaptureStream).call(v)
}

function Tile({ stream, muted, className = '', label }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream || null
  }, [stream])
  return (
    <div className={`relative overflow-hidden bg-black ${className}`}>
      <video
        ref={ref}
        autoPlay
        playsInline
        muted={muted}
        className="h-full w-full object-cover"
      />
      <span className="absolute left-3 top-3 rounded-lg bg-black/60 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur-xs">
        {label}
      </span>
    </div>
  )
}

export default function Call({ setup, onEnd }) {
  const [selfStream, setSelfStream] = useState(null)
  const [remoteStream, setRemoteStream] = useState(null)
  const [peerStatus, setPeerStatus] = useState('Connecting…')
  const [err, setErr] = useState('')
  const [mic, setMic] = useState(true)
  const [cam, setCam] = useState(true)
  const [secs, setSecs] = useState(0)

  // Forensic analysis state
  const [checks, setChecks] = useState(() =>
    Object.fromEntries(CHECKS.map((c) => [c.id, 'running']))
  )
  const [findings, setFindings] = useState([])
  const [risk, setRisk] = useState(0)
  const [action, setAction] = useState(null)
  const [evidenceList, setEvidenceList] = useState([])
  const [summaryData, setSummaryData] = useState(null)

  // Challenge states
  const [activeFlashSequence, setActiveFlashSequence] = useState(null)
  const [activePhrasePrompt, setActivePhrasePrompt] = useState(null)

  const analysisHandleRef = useRef(null)
  const dataConnRef = useRef(null)
  const peerRef = useRef(null)

  const isAgent = setup.mode === 'agent'
  const isCaller = setup.mode === 'caller'
  const isRehearsal = setup.mode === 'rehearsal'
  const isWebcam = setup.mode === 'webcam'

  // 1. Setup Media & WebRTC Peer Connections
  useEffect(() => {
    let dead = false
    let localStream = null

    async function init() {
      try {
        // Get local webcam/mic (or placeholder if permissions denied/demo)
        try {
          localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: 15 },
            audio: true,
          })
        } catch (mediaErr) {
          console.warn('Webcam access not granted, using placeholder:', mediaErr)
          localStream = placeholderStream()
        }

        if (dead) return
        setSelfStream(localStream)

        // Send hardware metadata if available
        const videoTrack = localStream.getVideoTracks()[0]
        const cameraLabel = videoTrack?.label || 'Direct Webcam'

        // ----------------- Direct Live Webcam Mode -----------------
        if (isWebcam) {
          setRemoteStream(localStream)
          setPeerStatus('Live Webcam Active')

          analysisHandleRef.current = startAnalysis({
            stream: localStream,
            consent: setup.consent,
            onEvent: handleAnalysisEvent,
          })

          analysisHandleRef.current.sendMeta?.({
            camera_label: cameraLabel,
          })
          return
        }

        // ----------------- Rehearsal Mode -----------------
        if (isRehearsal) {
          const other = await remoteFrom(setup.file)
          if (dead) return
          setRemoteStream(other)
          setPeerStatus('Rehearsal Running')

          analysisHandleRef.current = startAnalysis({
            stream: other,
            scenario: setup.scenario,
            consent: setup.consent,
            onEvent: handleAnalysisEvent,
          })

          analysisHandleRef.current.sendMeta?.({
            camera_label: cameraLabel,
            frame_intervals_ms: [66.666],
          })
          return
        }

        // ----------------- Live PeerJS WebRTC Calling -----------------
        const peerId = isAgent
          ? `${setup.roomCode}-agent`
          : `${setup.roomCode}-caller`

        const peer = new Peer(peerId)
        peerRef.current = peer

        peer.on('open', () => {
          if (dead) return
          if (isAgent) {
            setPeerStatus(`Waiting for caller to enter code ${setup.roomCode}…`)
          } else {
            setPeerStatus('Connecting to Agent station…')
            // Caller initiates call to agent
            const call = peer.call(`${setup.roomCode}-agent`, localStream)
            call.on('stream', (agentStream) => {
              setRemoteStream(agentStream)
              setPeerStatus('Call Connected')
            })

            // Setup data channel to exchange challenge commands
            const conn = peer.connect(`${setup.roomCode}-agent`)
            dataConnRef.current = conn
            setupDataChannel(conn)
          }
        })

        if (isAgent) {
          // Agent listens for incoming caller stream
          peer.on('call', (incomingCall) => {
            incomingCall.answer(localStream)
            incomingCall.on('stream', (callerStream) => {
              setRemoteStream(callerStream)
              setPeerStatus('Caller Connected')

              // Start real-time analysis pipeline on caller's stream
              if (!analysisHandleRef.current) {
                analysisHandleRef.current = startAnalysis({
                  stream: callerStream,
                  consent: setup.consent,
                  onEvent: handleAnalysisEvent,
                })
                analysisHandleRef.current.sendMeta?.({
                  camera_label: cameraLabel,
                })
              }
            })
          })

          peer.on('connection', (conn) => {
            dataConnRef.current = conn
            setupDataChannel(conn)
          })
        }

        peer.on('error', (err) => {
          console.warn('PeerJS error:', err)
          setPeerStatus(`Peer status: ${err.type || 'disconnected'}`)
        })
      } catch (e) {
        setErr(e.message || 'Could not initialize video call.')
      }
    }

    init()

    const timer = setInterval(() => setSecs((s) => s + 1), 1000)

    return () => {
      dead = true
      clearInterval(timer)
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop())
      }
      if (peerRef.current) {
        peerRef.current.destroy()
      }
      if (analysisHandleRef.current) {
        analysisHandleRef.current.stop()
      }
    }
  }, [setup])

  // Data connection handlers between Agent and Caller
  const setupDataChannel = (conn) => {
    conn.on('open', () => {
      console.log('PeerJS data connection opened')
    })
    conn.on('data', (data) => {
      if (data.type === 'trigger_flash') {
        // Caller receives instruction to flash screen
        setActiveFlashSequence(data.sequence)
      } else if (data.type === 'trigger_phrase') {
        setActivePhrasePrompt(data.phrase)
      }
    })
  }

  // Forensic Event Dispatcher
  const handleAnalysisEvent = (e) => {
    if (e.type === 'check') {
      setChecks((prev) => ({ ...prev, [e.id]: e.state }))
    } else if (e.type === 'finding') {
      setFindings((prev) => {
        // Prevent duplicate IDs from cluttering view
        const exists = prev.some((p) => p.id === e.id && p.title === e.title)
        return exists ? prev : [...prev, e]
      })
    } else if (e.type === 'risk') {
      setRisk(e.value)
    } else if (e.type === 'action') {
      setAction(e.text)
    } else if (e.type === 'evidence') {
      setEvidenceList((prev) => [...prev, e])
    } else if (e.type === 'summary') {
      setSummaryData({ text: e.text, ai_generated: e.ai_generated })
    }
  }

  // Trigger Light Challenge
  const handleTriggerLight = async () => {
    const sess = await createSession()
    const sequence = sess.light_sequence || [
      { color: '#ffffff', ms: 350 },
      { color: '#38bdf8', ms: 400 },
      { color: '#4ade80', ms: 400 },
      { color: '#f43f5e', ms: 400 },
    ]

    // Send challenge message to backend
    analysisHandleRef.current?.sendChallenge?.({
      type: 'challenge',
      kind: 'light',
      t0: Date.now(),
      sequence,
    })

    // If caller device is connected via DataConnection, trigger their screen flash
    if (dataConnRef.current && dataConnRef.current.open) {
      dataConnRef.current.send({
        type: 'trigger_flash',
        sequence,
      })
    } else {
      // Rehearsal or local demo: flash local screen
      setActiveFlashSequence(sequence)
    }
  }

  // Trigger Phrase Challenge
  const handleTriggerPhrase = async () => {
    const sess = await createSession()
    const phrase = sess.phrase || 'Peter Piper picked a peck of pickled peppers'

    analysisHandleRef.current?.sendChallenge?.({
      type: 'challenge',
      kind: 'phrase',
      expected: phrase,
      t0: Date.now(),
    })

    if (dataConnRef.current && dataConnRef.current.open) {
      dataConnRef.current.send({
        type: 'trigger_phrase',
        phrase,
      })
    }
    setActivePhrasePrompt(phrase)
  }

  const toggle = (kind, on, set) => {
    selfStream?.getTracks().filter((t) => t.kind === kind).forEach((t) => (t.enabled = !on))
    set(!on)
  }

  const mmss = `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`
  const btn = 'rounded-full px-4 py-2 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2'

  return (
    <div className="flex h-screen flex-col lg:flex-row bg-stage">
      {/* Active Light Flash Overlay */}
      {activeFlashSequence && (
        <FlashOverlay
          sequence={activeFlashSequence}
          onComplete={() => setActiveFlashSequence(null)}
          onSkip={() => setActiveFlashSequence(null)}
        />
      )}

      {/* Main Video Call Stage */}
      <section className="relative flex min-h-[55vh] flex-1 flex-col p-3 lg:p-4">
        {/* High Risk Alert Banner */}
        {risk >= 0.70 && (
          <div className="mb-3 flex items-center justify-between rounded-xl bg-bad px-4 py-2.5 text-white shadow-lg animate-bounce">
            <div className="flex items-center gap-2">
              <span className="text-lg">🚨</span>
              <span className="font-bold text-sm tracking-wide">
                High Deepfake Probability Detected ({Math.round(risk * 100)}%)
              </span>
            </div>
            <span className="text-xs bg-white/20 px-2.5 py-1 rounded-md font-semibold">
              Action Required
            </span>
          </div>
        )}

        {/* Phrase Challenge Banner Prompt */}
        {activePhrasePrompt && (
          <div className="mb-3 flex items-center justify-between rounded-xl bg-amber-500 px-4 py-2.5 text-white shadow-lg">
            <div>
              <span className="text-xs uppercase font-bold tracking-wider opacity-90">
                Spoken Verification Phrase:
              </span>
              <p className="font-bold text-sm">"{activePhrasePrompt}"</p>
            </div>
            <button
              onClick={() => setActivePhrasePrompt(null)}
              className="text-xs bg-black/20 hover:bg-black/30 px-3 py-1.5 rounded-lg font-semibold"
            >
              Dismiss
            </button>
          </div>
        )}

        {err ? (
          <div className="m-auto max-w-sm text-center text-white">
            <p className="font-semibold text-lg">Unable to connect call</p>
            <p className="mt-2 text-sm text-slate-300">{err}</p>
            <button onClick={onEnd} className={`${btn} mt-4 bg-white text-ink hover:bg-slate-200`}>
              Back to Lobby
            </button>
          </div>
        ) : (
          <>
            <div className="relative flex-1 overflow-hidden rounded-2xl border border-white/10 shadow-2xl">
              <Tile
                stream={remoteStream}
                className="h-full w-full"
                label={`${isCaller ? 'Agent Station' : 'Caller'} · ${mmss} (${peerStatus})`}
              />
              <Tile
                stream={selfStream}
                muted
                className="absolute bottom-3 right-3 h-28 w-40 rounded-xl border border-white/30 shadow-lg"
                label={isCaller ? 'You (Caller)' : 'You (Agent)'}
              />
            </div>

            {/* Media & Session Control Bar */}
            <div className="mt-3 flex items-center justify-center gap-3">
              <button
                onClick={() => toggle('audio', mic, setMic)}
                className={`${btn} ${mic ? 'bg-white/15 text-white hover:bg-white/25' : 'bg-white text-ink'}`}
              >
                {mic ? 'Mute Mic' : 'Unmute Mic'}
              </button>
              <button
                onClick={() => toggle('video', cam, setCam)}
                className={`${btn} ${cam ? 'bg-white/15 text-white hover:bg-white/25' : 'bg-white text-ink'}`}
              >
                {cam ? 'Camera Off' : 'Camera On'}
              </button>
              <button
                onClick={onEnd}
                className={`${btn} bg-bad text-white hover:bg-bad/90 shadow-sm`}
              >
                End Call
              </button>
            </div>
          </>
        )}
      </section>

      {/* Forensics Panel (Agent or Rehearsal) */}
      {!isCaller && (
        <Panel
          checks={checks}
          findings={findings}
          risk={risk}
          action={action}
          evidence={evidenceList}
          onTriggerLightChallenge={handleTriggerLight}
          onTriggerPhraseChallenge={handleTriggerPhrase}
        />
      )}

      {/* Case Summary Modal when generated */}
      {summaryData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-xs">
          <div className="max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-lg text-ink">Verification Audit Summary</h3>
              {summaryData.ai_generated ? (
                <span className="rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-semibold text-purple-700">
                  ✨ AI-generated summary
                </span>
              ) : (
                <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
                  🛡️ System Verified Summary
                </span>
              )}
            </div>

            <p className="mt-4 text-sm text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-200">
              {summaryData.text}
            </p>

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setSummaryData(null)}
                className="rounded-lg bg-ink px-4 py-2 text-sm font-semibold text-white hover:bg-ink/90"
              >
                Close & Finish
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
