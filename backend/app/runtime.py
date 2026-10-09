"""ModelRuntime for CPU-scheduled deepfake inference.
Manages thread pools, ONNX sessions, fast-path / slow-path cascading, and latency recording.
"""
import os
import json
import time
import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from app.config import settings
from app.metrics import FAST_PATH_LATENCY, SLOW_PATH_LATENCY

logger = logging.getLogger("trustlens.runtime")


@dataclass
class ModelOutputs:
    edge_logit: float = 0.0
    clip_logit: float = 0.0
    spoof_max: float = 0.0
    spoof_mean: float = 0.0
    ood_score: float = 0.0
    model_disagreement: float = 0.0
    slow_path_fresh: bool = False
    stubs_in_use: bool = True


class ModelRuntime:
    """CPU-constrained model inference scheduler for 8 GB RAM laptop."""

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(__file__), "..", "models", "models_registry.json"
        )
        self.registry: Dict[str, Any] = {}
        self.stubs_in_use = True
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.ORT_THREADS,
            thread_name_prefix="model_worker",
        )
        self.sessions: Dict[str, Any] = {}
        self._slow_task: Optional[concurrent.futures.Future] = None
        self._last_slow_outputs = (0.0, 0.0, 0.0, 0.0)  # (clip_logit, spoof_max, spoof_mean, ood_score)
        self._window_counter = 0

        self._load_registry()
        self._init_models()

    def _load_registry(self):
        """Load model registry metadata."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    self.registry = json.load(f)
                    self.stubs_in_use = self.registry.get("stubs_in_use", True)
                    logger.info("Loaded models registry (stubs_in_use=%s)", self.stubs_in_use)
            except Exception as e:
                logger.warning("Failed to load models_registry.json: %s", e)

    def _init_models(self):
        """Warm up ONNX sessions or initialize stub-0 handlers."""
        models_dir = os.path.dirname(self.registry_path)
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = settings.ORT_THREADS
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            edgenet_path = os.path.join(models_dir, "edgenet_int8.onnx")
            if os.path.exists(edgenet_path):
                self.sessions["edgenet"] = ort.InferenceSession(
                    edgenet_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                logger.info("EdgeNet INT8 ONNX session initialized.")

            clip_path = os.path.join(models_dir, "clip_head_int8.onnx")
            if os.path.exists(clip_path):
                self.sessions["clip_sbi"] = ort.InferenceSession(
                    clip_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                logger.info("CLIP-SBI INT8 ONNX session initialized.")

            wavlm_path = os.path.join(models_dir, "wavlm_head_int8.onnx")
            if os.path.exists(wavlm_path):
                self.sessions["wavlm_head"] = ort.InferenceSession(
                    wavlm_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                logger.info("WavLM Head INT8 ONNX session initialized.")

            if len(self.sessions) >= 3:
                self.stubs_in_use = False
                logger.info("All 3 neural ONNX models loaded successfully. Stubs disabled.")
        except Exception as e:
            logger.warning("Could not initialize ONNX sessions: %s. Using stub-0.", e)

    def run_inference(
        self,
        face_crops: List[np.ndarray],
        audio_pcm: np.ndarray,
        quality_trust: float,
    ) -> ModelOutputs:
        """
        Cadence scheduler:
        - Fast path: runs every 1 s window (EdgeNet).
        - Slow path: runs every 3rd window in background (CLIP + WavLM).
        """
        self._window_counter += 1
        t_start = time.perf_counter()

        # 1. Fast Path (EdgeNet)
        edge_logit = self._run_edgenet(face_crops)
        fast_latency = time.perf_counter() - t_start
        FAST_PATH_LATENCY.observe(fast_latency)

        # 2. Slow Path (CLIP + WavLM) - triggered every 3rd window or window 1
        slow_fresh = False
        if self._window_counter % 3 == 0 or self._window_counter == 1:
            if self._slow_task is None or self._slow_task.done():
                self._slow_task = self.thread_pool.submit(
                    self._run_slow_path, face_crops, audio_pcm
                )

        # Check if slow path completed
        if self._slow_task is not None and self._slow_task.done():
            try:
                self._last_slow_outputs = self._slow_task.result()
                slow_fresh = True
            except Exception as e:
                logger.warning("Slow path execution error: %s", e)

        clip_logit, spoof_max, spoof_mean, ood_score = self._last_slow_outputs

        # Calculate model disagreement (e.g. EdgeNet vs CLIP)
        model_disagreement = float(abs(np.tanh(edge_logit) - np.tanh(clip_logit)))

        return ModelOutputs(
            edge_logit=edge_logit,
            clip_logit=clip_logit,
            spoof_max=spoof_max,
            spoof_mean=spoof_mean,
            ood_score=ood_score,
            model_disagreement=model_disagreement,
            slow_path_fresh=slow_fresh,
            stubs_in_use=self.stubs_in_use,
        )

    def _run_edgenet(self, face_crops: List[np.ndarray]) -> float:
        """Fast path: MobileNetV3-Small + Frequency branch."""
        if "edgenet" in self.sessions and len(face_crops) >= 16:
            try:
                crops_arr = np.array(face_crops[:16], dtype=np.float32) / 255.0
                crops_t = np.transpose(crops_arr, (0, 3, 1, 2))  # (16, 3, 160, 160)
                input_feed = {"frames": np.expand_dims(crops_t, axis=0)}
                out = self.sessions["edgenet"].run(["edge_logit"], input_feed)[0]
                return float(out[0, 0])
            except Exception as e:
                logger.warning("EdgeNet session run error: %s", e)

        # Neutral fallback when models are warming up or face crops absent
        return 0.0

    def _extract_audio_features(self, audio_pcm: np.ndarray) -> np.ndarray:
        """Extract (1, 50, 768) acoustic feature tensor from 16kHz audio for WavLM head."""
        if len(audio_pcm) < 16000:
            sig = np.zeros(16000, dtype=np.float32)
            if len(audio_pcm) > 0:
                sig[-len(audio_pcm):] = audio_pcm.astype(np.float32)
        else:
            sig = audio_pcm[-16000:].astype(np.float32)

        max_v = float(np.max(np.abs(sig))) + 1e-6
        sig = sig / max_v

        frame_size = 320
        num_frames = 50
        feats = np.zeros((1, num_frames, 768), dtype=np.float32)
        window = np.hanning(frame_size)

        for i in range(num_frames):
            start = i * frame_size
            chunk = sig[start : start + frame_size]
            if len(chunk) < frame_size:
                break
            fft_mag = np.abs(np.fft.rfft(chunk * window)) + 1e-6
            log_mag = np.log(fft_mag)
            norm_mag = (log_mag - np.mean(log_mag)) / (np.std(log_mag) + 1e-6)
            rep = int(np.ceil(768 / len(norm_mag)))
            feats[0, i, :] = np.tile(norm_mag, rep)[:768]

        return feats

    def _extract_clip_features(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """Extract (1, 8, 512) visual embedding features for CLIP-SBI head."""
        import cv2
        num_frames = 8
        indices = np.linspace(0, len(face_crops) - 1, num_frames, dtype=int)
        feats = np.zeros((1, num_frames, 512), dtype=np.float32)
        for i, idx in enumerate(indices):
            crop = face_crops[idx]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            low = cv2.resize(gray, (16, 16)).flatten()
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_low = cv2.resize(grad_mag, (16, 16)).flatten()
            combined = np.concatenate([low, grad_low])[:512]
            norm = float(np.linalg.norm(combined)) + 1e-6
            feats[0, i, :] = combined / norm
        return feats

    def _run_slow_path(self, face_crops: List[np.ndarray], audio_pcm: np.ndarray) -> Tuple[float, float, float, float]:
        """Slow path (runs every 3 s): CLIP-LN and WavLM Head."""
        t0 = time.perf_counter()
        clip_logit = 0.0
        spoof_max = 0.0
        spoof_mean = 0.0
        ood_score = 0.0

        # 1. CLIP-SBI Inference
        if "clip_sbi" in self.sessions and len(face_crops) >= 8:
            try:
                clip_feats = self._extract_clip_features(face_crops)
                out = self.sessions["clip_sbi"].run(["clip_logit"], {"clip_embeddings": clip_feats})[0]
                clip_logit = float(out[0, 0])
            except Exception as e:
                logger.warning("CLIP session error: %s", e)

        # 2. WavLM Head Audio Spoof Inference
        if "wavlm_head" in self.sessions and len(audio_pcm) >= 8000:
            try:
                audio_feats = self._extract_audio_features(audio_pcm)
                out = self.sessions["wavlm_head"].run(["spoof_logit"], {"audio_features": audio_feats})[0]
                raw_spoof_logit = float(out[0, 0])

                audio_std = float(np.std(audio_pcm)) if len(audio_pcm) > 0 else 0.0
                if audio_std < 50.0 or raw_spoof_logit <= 0.0:
                    spoof_max = 0.0
                    spoof_mean = 0.0
                else:
                    spoof_max = float(np.clip(raw_spoof_logit / 3.0, 0.0, 1.0))
                    spoof_mean = spoof_max * 0.75
            except Exception as e:
                logger.warning("WavLM session error: %s", e)

        slow_latency = time.perf_counter() - t0
        SLOW_PATH_LATENCY.observe(slow_latency)
        return (clip_logit, spoof_max, spoof_mean, ood_score)

    def close(self):
        """Shutdown thread pool and cleanup sessions."""
        self.thread_pool.shutdown(wait=False)
        self.sessions.clear()
