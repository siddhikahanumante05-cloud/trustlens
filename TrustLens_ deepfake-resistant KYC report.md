Mastercard, CSB-02. Technical report and execution plan.

# TrustLens

Catch deepfake injection in video KYC from a 3-second clip, of a person we have never seen, and say why.

## Summary

Build three honest agents, not eight stubs.

Your design is a team of small specialist agents fused into one calibrated risk score. The design is sound. The constraint is a CPU-only laptop and free Colab, so the plan below trains and ships only the parts that can be measured on real data, and marks the rest as not available in the UI.

- **Trained on DFDC:** a lightweight face-appearance classifier (M3) with temporal pooling and a frequency branch.
- **Built from your own recordings:** the screen-light challenge (M1) and landmark geometry checks (M4). DFDC has no flash or spoken-phrase signal, so these cannot be validated on it.
- **Stretch:** phrase and lip-closure check (M5), once Whisper timestamps are wired in.
- **Demo:** a Gradio app with a webcam clip, a mock onboarding flow, and finding cards that explain the score. A human makes the final call.
- **Delivery:** two steps. Step 1 builds and evaluates the detector on Colab. Step 2 integrates it into the onboarding flow on your laptop.

## Problem and constraints

Video KYC is being attacked by injection: a synthetic or face-swapped video is pushed into the app through a virtual camera or a hooked camera API. It can blink and smile on command, so ordinary liveness checks pass it. Presentation attacks (a photo or screen held to a real camera) are the easier problem.

| Constraint | What it forces |
| --- | --- |
| Short clip, about 3 seconds | No long-term temporal cues. Pulse estimation is information only. |
| No baseline of the real person | Single-session detection. No identity drift against a stored reference. |
| CPU-only laptop | Small models, int8 ONNX, one heavy job at a time. |
| Free Colab for training | Frozen or lightly tuned backbones, a data subset, checkpoints saved to Drive. |
| Explanations required | Every score must trace to named artifacts, labeled faithful or illustrative. |

## Dataset

The DFDC dataset from Meta ([ai.meta.com/datasets/dfdc](https://ai.meta.com/datasets/dfdc/)) holds video clips of paid actors, some manipulated with several face-swap methods, and a minority with manipulated audio. Labels are at video level.

### Use it for

- Training and testing the face-appearance agent on real versus synthetic faces.
- Frame-level and temporal artifact learning: blend seams, texture, flicker.
- Optional audio-fake examples for a voice agent.

### Do not expect it to give you

- Light-flash reactions, spoken challenge phrases, or camera metadata. These need your own recordings.
- Coverage of modern real-time face-swap tools. DFDC generators are from 2019 and 2020.
- Live webcam compression. Degrade both classes during training to approximate it.

### Practical plan

- The full set is hundreds of gigabytes. Start from the smaller preview or a sampled slice that fits free Colab disk and a 15 GB Drive.
- Check the licence and access terms on the Meta page before downloading, and keep the data out of your repository.
- Split by source video and actor, never by frame, so the same person never appears in both train and test.

## Architecture

Each agent answers one question and never reads another agent's output. Only fusion sees all of them.

| Agent | Question it answers | Version 1 | Evidence |
| --- | --- | --- | --- |
| M1 Source and light | Is this a live camera facing this screen right now? | Build | Own recordings |
| M3 Video appearance | Do the pixels look like a real capture? | Train | DFDC |
| M4 Face geometry | Does the face move like a real head? | Build | Own recordings, DFDC check |
| M5 Speech and lips | Do the sound and lips agree? | Stretch | Own recordings |
| M2 Audio | Is the voice natural? | Cut, shown as not available | None yet |
| Q Quality gate | How much should the others be trusted now? | Build | Heuristic |
| Fusion | What is the calibrated risk? | Train on agent scores | Held-out sessions |

### Rules every agent follows

- Missing input returns *skipped* with zero confidence. It is never read as fake.
- A stub reports itself as a stub, carries weight 0 in fusion, and shows as not available.
- Fusion is trained with whole agents randomly dropped, so it works with any subset.

## Scope decision

Your current repository has a working backend and frontend scaffold, but the face-texture, voice and identity models are stubs and no accuracy is measured. Adding more agents does not help a judge. A smaller system with real numbers does.

### Keep

- M3, the only agent that can be trained and benchmarked on DFDC.
- M1 light challenge: random colors from an HMAC of the session nonce. A pre-made fake cannot know them.
- M4 landmark jitter and occlusion checks.
- Quality gate, fusion, and the explanation layer.

### Defer

- CLIP blend-trace model, WavLM voice head, local LLM narrator, PeerJS two-device calls. Each adds RAM and failure modes without adding evidence.

### Fix before anything else

- Phrase check: an empty transcript gives an error rate of 1.0 and flags every caller. Report *skipped* until Whisper is wired in.
- Light check: apply the sync-pulse alignment and use all color channels, not red only.
- Verdict thresholds: code uses 0.40 and an older doc used 0.35. Choose one.
- Model card: remove AUC constants that were never measured.
- Repository: remove committed `cases/`, `node_modules/` and `dist/`.

## Detector design

### M3 appearance model

- **Input:** 16 evenly spaced face crops from the 3-second window, 224 by 224, aligned using landmarks.
- **Spatial branch:** MobileNetV3-Small pretrained on ImageNet, fine-tuned.
- **Frequency branch:** FFT magnitude of the crop, small CNN. Generators leave spectral traces.
- **Temporal head:** mean and standard deviation of per-frame embeddings, then a small GRU. The standard deviation captures flicker.
- **Data:** DFDC subset plus self-blended images made from real frames, so the model learns blend seams and not one generator's fingerprint.
- **Robustness:** random JPEG, blur, resize and bitrate degradation on both classes.
- **Calibration:** isotonic regression on a held-out actor split, so a score of 0.8 means roughly 80 percent.
- **Export:** ONNX with int8 quantization for CPU inference.

### M1 light challenge

- The caller screen shows a white sync pulse and then soft random colors, at most three changes per second, with a warning and a Skip button.
- Measure correlation between the expected sequence and the color of face, neck and background.
- A face swap on a real neck can show the face and neck reacting differently.
- In a Gradio page the flash and the recording are timed in the browser. Treat absolute lag as unreliable and score correlation and signal strength only.

### M4 geometry

- Second differences of 478 landmark tracks as a smoothness measure.
- Hand-over-face occlusion test, where face swaps often break.

### Fusion and verdict

- Regularized logistic regression on agent risks, scaled by agent confidence, then calibrated.
- Hard gates: a virtual-camera flag with no light reaction goes straight to human escalation. Too dark or too blurry goes to retry.
- Smoothing with hysteresis, so the badge does not flicker.

| Risk | Verdict | Action |
| --- | --- | --- |
| Below 0.40 | Looks real | Continue onboarding |
| 0.40 to 0.70 | Needs a closer look | Step-up check: head turn, repeat phrase |
| 0.70 or more | Likely a deepfake | Escalate to a human reviewer |

Thresholds are starting values to tune on held-out data, not results.

## Explanations

Each reason on screen carries a trust label so nobody mistakes a picture for proof.

| Explanation | Source | Trust |
| --- | --- | --- |
| Weight times value per agent | Fusion model | Faithful |
| Reason codes, such as no light reaction | Rules on real features | Faithful |
| Counterfactual: risk if the light were normal | Re-run of fusion | Faithful |
| Heatmap on the two most suspicious frames | Grad-CAM on M3 | Illustrative |
| Plain-English summary | Fixed template | Wording only |

Finding cards use names a reviewer understands: face did not react to screen light, face edges look artificial, face and neck react differently, identity changes between frames, face breaks when covered.

## Mock onboarding flow

One Gradio app on your laptop, with the webcam recorded as a short clip.

1. **Consent.** The caller accepts analysis of video, with no raw media stored.
2. **Mock ID and selfie.** Upload an ID image and capture a selfie. This is a stand-in for the real KYC step and is not analyzed.
3. **Challenge.** The page shows a random phrase and plays the color sequence while recording 3 to 4 seconds.
4. **Analysis.** The clip goes through preprocessing, agents and fusion on CPU.
5. **Result.** A risk meter, finding cards, the two heatmaps and a recommended action.
6. **Decision.** Approve, step-up check, or send to a reviewer. The call is never rejected automatically.

A second tab, *Attack lab*, lets you upload a prerecorded or synthetic clip and compare it against the live result. This is where the demo makes its point: the same face passes appearance checks and fails the light challenge.

Gradio is the better fit than Streamlit here because its webcam recording component gives a complete clip with no extra WebRTC wiring.

## Execution plan

### Step 1 Build and measure the detector

Where: free Colab. Effort: about 5 working days.

#### Tasks

- Get DFDC access, pull a subset, verify labels, split by actor.
- Extract faces and landmarks, cache 16-frame tensors to Drive.
- Train M3 with degradation and self-blended images. Checkpoint every epoch.
- Calibrate. Run per-agent evaluation, compression sweep and error analysis.
- Build M1 and M4 as modules with the common result format.
- Record a small consented set: real, replay, virtual camera, and a face swap if possible.
- Train fusion on agent outputs. Validate on real sessions only.
- Export int8 ONNX and time it on your laptop CPU.

#### Outputs

- Trained and calibrated M3 as an ONNX file.
- M1, M4, quality gate and fusion modules with tests.
- An evaluation notebook with AUC, EER, calibration and latency.

#### Exit when

- Every reported number comes from a held-out actor split.
- M3 runs under 1.5 seconds per window at p95 on your CPU.
- No agent still labeled stub contributes to a score.

### Step 2 Integrate and demonstrate

Where: your laptop. Effort: about 4 working days.

#### Tasks

- Build the Gradio flow: consent, mock ID, challenge, recording, result.
- Wire the pipeline: clip decoding, windowing, agents, fusion, explainer.
- Render risk meter, finding cards, heatmaps and the recommended action.
- Add the Attack lab tab with prepared real and fake clips.
- Add the memory guard: one heavy job at a time, skip when RAM is low.
- Test missing inputs: no audio, no face, no flash, a dark room.
- Write the model card with measured results and honest limits.
- Script and rehearse a 5-minute demo with a backup recorded clip.

#### Outputs

- A runnable app: one command to start.
- Tables for attack acceptance, real-user rejection and ablation.
- Demo script, backup video and a short slide or README.

#### Exit when

- A stranger can run the flow from the README alone.
- Each finding card traces to a real feature value.
- The demo survives a bad webcam and a Wi-Fi drop.

Effort assumes one person, and will change once you tell me the deadline and team size.

## Evaluation

| Test | What it shows | Result |
| --- | --- | --- |
| AUC and EER per agent on 3 s windows | Each agent works alone on short clips | To be measured |
| Compression sweep: clean, medium, heavy | Holds up on real webcams | To be measured |
| Attack acceptance at 1, 2, 5 percent real-user rejection | The main security number | To be measured |
| Rejection by webcam quality, skin tone, glasses, lighting | Fairness to real users | To be measured |
| Leave-one-agent-out ablation | Each agent earns its place | To be measured |
| Missing-agent test | Sensible verdict with partial inputs | To be measured |
| Calibration error and reliability plot | Scores can be trusted | To be measured |
| Latency p50 and p95, peak RAM | Fits a CPU laptop | To be measured |

- Split by actor and source video. Tune only on validation data.
- Report 95 percent bootstrap confidence intervals. A small benchmark gives wide ones, and that is fine.
- Never quote an accuracy while any contributing agent is a stub.
- Label synthetic test clips as plumbing checks, not results.

## Threat model

| Level | Attacker | Main defense | Honest rating |
| --- | --- | --- | --- |
| L0 | Photo or screen to a real camera | Light check, quality | Strong |
| L1 | Prerecorded deepfake via virtual camera | Light check, camera flags | Strong |
| L2 | Real-time face swap on a live person | Face versus neck light, appearance, geometry | Medium, test it |
| L3 | Swap plus cloned voice | Phrase, lip closure | Medium, stretch |
| L4 | Attacker re-lights the fake to match the flash | Layered random challenges | Raises cost only |
| L5 | Compromised device or hooked camera API | Needs native attestation | Not solved in a web demo |

## Risks

| Risk | Likelihood | Response |
| --- | --- | --- |
| DFDC access or download is slow or blocked | Medium | Start with the preview slice. Request access on day one. |
| Colab disconnects mid-training | High | Checkpoint each epoch to Drive. Cache features, not videos. |
| Model overfits to DFDC generators | High | Self-blended images, degradation, report the cross-source gap. |
| Light check weak in a bright room | Medium | Quality gate lowers its weight. Ask for lower ambient light. |
| Flash and recording are misaligned in the browser | Medium | Sync pulse alignment. Score correlation, not lag. |
| Demo fails on stage | Medium | Backup clip, offline-only path, no dependency on public services. |
| Bias across skin tone or lighting | Medium | Report slice results. Say so if the sample is too small. |

## Privacy and limits

- Consent comes from the person on camera. Analysis does not start without it.
- Raw video and audio are not stored. Only features, findings and hashes are kept, and they expire.
- All processing is local. No clip is sent to an outside API.
- Check local video-KYC rules, such as RBI V-CIP, GDPR or FinCEN guidance, and confirm exact clauses before citing them.
- A compromised device (L5) cannot be solved in a browser.
- A fast adaptive attacker can weaken the light check.
- Newer generators than those in DFDC can fool M3. The gap will be measured and reported, not hidden.
- Three seconds is short. The demo benchmark will be small.

## Open questions

Answer what you can. Short answers are enough. Each one changes the plan.

### Data

1. Do you already have DFDC access, or must you apply now? Which version: preview, a sample, or full?
2. How much Google Drive space is free, and do you have a Kaggle account for a mirrored copy?
3. Is a self-blended-image augmentation acceptable, or must training use only DFDC fakes?
4. Do you have any other datasets in mind for a cross-source test, such as Celeb-DF or FaceForensics++?

### Scope

5. Is audio mandatory for judging, or is a strong video-only result enough?
6. Do you want the light challenge to be the headline feature, or a supporting one?
7. Is the existing TrustLens repository your own work to reuse, or should this start clean?
8. Should the two-device call and PeerJS stay out of version 1?

### Demo

9. Who is the audience: judges reading a report, a live demo, or both? How long is the demo slot?
10. Will the demo run on your laptop, or must it also be hosted online?
11. Can you record yourself and two or three friends with consent, for the real and replay sets?
12. Can you run a free face-swap tool on Colab to create a few live-swap test clips, or should we use prerecorded fakes only?

### Constraints

13. What is the deadline, and how many people are on the team?
14. Are there Mastercard judging criteria beyond the problem statement, such as accuracy targets, latency, or explainability?
15. How many hours of Colab GPU can you realistically get per day?
16. Any compliance framing that must appear in the report, for example India, the US or the EU?

Report generated for CSB-02. No accuracy figures are claimed in this document. Every number will come from a held-out evaluation in Step 1.