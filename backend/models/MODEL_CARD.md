# TrustLens Model Card: Multi-Signal Deepfake Detection Suite

## 1. Overview
The **TrustLens** model suite provides real-time verification of short ($3\text{-second}$) video and audio streams of unknown subjects during remote KYC video calls. It combines active challenge-response signals with lightweight spatio-temporal neural networks running entirely on CPU within an $8\text{ GB RAM}$ budget.

---

## 2. Model Architectures & Channels

| Channel | Architecture / Model | Input Modality | Target Latency (CPU) | Output |
|---|---|---|---|---|
| **E1** | Provenance Rules & Jitter Estimator | Metadata, frame intervals | $< 1\text{ ms}$ | `source_risk`, flags |
| **E2** | Chromaticity Reflection Tracker | Facial ROI chromaticity ($r, g$) | $< 5\text{ ms}$ | `light_score`, `corr_face`, `snr` |
| **E3a** | **EdgeNet** (MobileNetV3 + 2D FFT + GRU) | $16 \times (160\times 160\times 3)$ crops | $\approx 48\text{ ms}$ (int8) | `edge_logit` |
| **E3b** | **CLIP-LN + SBI** (ViT-B/32 LayerNorm) | $8 \times 512\text{-d}$ embeddings | $\approx 185\text{ ms}$ (int8) | `clip_logit`, `ood_score` |
| **E4** | Facial Identity & Landmark Trajectory | Landmark coordinates (478 pts) | $< 2\text{ ms}$ | `identity_flicker`, `landmark_jitter` |
| **E5** | **WavLM Head** (WavLM-base-plus ASP) | $1\text{ s}$ $16\text{ kHz}$ PCM audio | $\approx 32\text{ ms}$ (int8) | `spoof_prob` |
| **E6a** | Phrase Verifier (faster-whisper tiny) | Spoken challenge audio | $< 1200\text{ ms}$ (once/phrase) | `phrase_wer` |
| **E6b** | Bilabial Aperture Validator | Aperture series at $b/p/m$ onsets | $< 1\text{ ms}$ | `bilabial_aperture_err` |
| **E6c** | Cheap Cross-Correlation Sync | Audio RMS envelope vs mouth aperture | $< 2\text{ ms}$ | `sync_conf`, `offset_ms` |
| **E8** | Quality Trust Estimator | Laplacian blur, brightness, face coverage | $< 2\text{ ms}$ | `quality_trust` |

---

## 3. Data & Training Protocols

### DFDC Face Extraction & Isolation (Rule 1)
- **Dataset**: Deepfake Detection Challenge (DFDC) Parts 0–3.
- **Source Isolation**: Train and test partitions are split strictly on the `original` video field in `metadata.json`. Test clips never share an identity or original recording session with training clips.

### Dual-Class Degradation Pipeline (Rule 2)
Both real and fake training samples are subjected to identical degradation distributions:
- JPEG compression ($Q \in [20, 90]$)
- Rescaling (downscale to 240p/360p then bicubic upscale)
- Additive sensor noise, brightness and contrast variation
- Gaussian blur and directional motion blur
- WebRTC frame drop simulation (temporal freezing)

### Audio Spoofing Training
- **Dataset**: ASVspoof 2019 Logical Access (LA) partition.
- **Augmentation**: Multi-codec transcoding (Opus, MP3, AMR), 8kHz downsampling/resampling, room impulse response (RIR) convolution.

---

## 4. Benchmarks & Quantization Profile (INT8 on CPU)

Evaluated on 4 intra-op CPU threads (`ORT_THREADS=4`):

| Model | Weights Size (FP32 $\to$ INT8) | Clean AUC [95% CI] | Medium AUC [95% CI] | Heavy AUC [95% CI] | p95 Latency |
|---|---|---|---|---|---|
| **EdgeNet** | $42.0\text{ MB} \to 14.0\text{ MB}$ (target) | *Not yet measured (stub active)* | *Not yet measured* | *Not yet measured* | $\approx 48.0\text{ ms}$ |
| **CLIP-LN + SBI** | $95.0\text{ MB} \to 28.5\text{ MB}$ (target) | *Not yet measured (stub active)* | *Not yet measured* | *Not yet measured* | $\approx 185.0\text{ ms}$ |
| **WavLM Head** | $18.0\text{ MB} \to 5.5\text{ MB}$ (target) | *Not yet measured (stub active)* | *Not yet measured* | *Not yet measured* | $\approx 32.0\text{ ms}$ |
| **E2 Active Light** | *Heuristic (0 params)* | *Awaiting TrustLens-Bench* | *Awaiting TrustLens-Bench* | *Awaiting TrustLens-Bench* | $< 5.0\text{ ms}$ |

---

## 5. Development Stubs Notice

> [!WARNING]
> While models are training on Kaggle/Colab, the backend operates with stubs labeled `stub-0`.
> Whenever stubs are loaded:
> - `/metrics` reports `trustlens_stubs_in_use{version="stub-0"} 1`.
> - `case.json` flags `stubs_in_use: true`.
> - Replay benchmarks refuse to print accuracy numbers until real weights are supplied.

---

## 6. Limitations & Ethical Disclosures
1. **Device Compromise**: Cannot defend against a fully compromised caller device or virtual machine that can synthesize native hardware attestation.
2. **Adaptive Re-lighting**: Advanced real-time face-swap pipelines with sub-frame neural re-lighting may reduce E2 reflection contrast under strong studio lighting.
3. **Demographic Diversity**: Performance may fluctuate on demographic groups or skin undertones underrepresented in public benchmark datasets; quality-bin conformal thresholding is utilized to prevent disparate false rejection rates.
4. **No Deepfake Generation**: This codebase strictly implements detection routines and contains no tools or weights for deepfake generation.
