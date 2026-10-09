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
        # Check if real ONNX weights exist
        models_dir = os.path.dirname(self.registry_path)
        edgenet_path = os.path.join(models_dir, "edgenet_int8.onnx")
        if os.path.exists(edgenet_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = settings.ORT_THREADS
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.sessions["edgenet"] = ort.InferenceSession(
                    edgenet_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.stubs_in_use = False
                logger.info("EdgeNet INT8 ONNX session initialized.")
            except Exception as e:
                logger.warning("Could not load EdgeNet ONNX: %s. Using stub-0.", e)
        else:
            logger.info("No EdgeNet weights at %s. Operating with stub-0 model.", edgenet_path)

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

        # 2. Slow Path (CLIP + WavLM) - triggered every 3rd window
        slow_fresh = False
        if self._window_counter % 3 == 0:
            if self._slow_task is None or self._slow_task.done():
                # Launch slow path in background thread
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
        """Fast path: MobileNetV3-Small + Log-FFT branch."""
        if "edgenet" in self.sessions and len(face_crops) == 16:
            try:
                # Preprocess input tensor (1, 16, 3, 160, 160) normalized
                crops_arr = np.array(face_crops, dtype=np.float32) / 255.0
                crops_t = np.transpose(crops_arr, (0, 3, 1, 2))  # (16, 3, 160, 160)
                input_feed = {"frames": np.expand_dims(crops_t, axis=0)}
                out = self.sessions["edgenet"].run(["edge_logit"], input_feed)[0]
                return float(out[0, 0])
            except Exception as e:
                logger.warning("EdgeNet session run error: %s", e)

        # Stub-0 heuristic: computes high-frequency Laplacian edge texture artifact metric
        if len(face_crops) > 0:
            import cv2
            gray = cv2.cvtColor(face_crops[0], cv2.COLOR_RGB2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # Stub score centered around zero
            return float(np.clip((lap_var - 60.0) / 100.0, -2.5, 2.5))
        return 0.0

    def _run_slow_path(self, face_crops: List[np.ndarray], audio_pcm: np.ndarray) -> Tuple[float, float, float, float]:
        """Slow path (runs every 3 s): CLIP-LN and WavLM Head."""
        t0 = time.perf_counter()
        # Simulated or real inference
        time.sleep(0.04)  # Simulate short CPU execution
        clip_logit = 0.0
        spoof_max = 0.0
        spoof_mean = 0.0
        ood_score = 0.0

        if len(audio_pcm) > 16000:
            audio_std = float(np.std(audio_pcm))
            spoof_max = float(np.clip((audio_std - 1500.0) / 4000.0, 0.0, 1.0))
            spoof_mean = spoof_max * 0.75

        slow_latency = time.perf_counter() - t0
        SLOW_PATH_LATENCY.observe(slow_latency)
        return (clip_logit, spoof_max, spoof_mean, ood_score)

    def close(self):
        """Shutdown thread pool and cleanup sessions."""
        self.thread_pool.shutdown(wait=False)
        self.sessions.clear()
