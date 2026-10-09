"""Dual-class degradation augmentation pipeline.
CRITICAL RULE: Must be applied equally to BOTH real and fake classes in training.
Never degrade only one class, preventing the classifier from keying on compression artifacts.
"""
import random
from typing import List, Tuple
import cv2
import numpy as np


class DualClassDegradation:
    """Simulates webcam, network, and video-call compression artifacts."""

    def __init__(self, p: float = 0.85):
        self.p = p

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Input: (T, H, W, C) uint8 numpy array.
        Output: Degraded (T, H, W, C) uint8 numpy array.
        """
        if random.random() > self.p:
            return frames

        t, h, w, c = frames.shape
        out = frames.copy()

        # 1. Random Rescaling (Downscale-Upscale to simulate 240p/360p webcam streams)
        if random.random() < 0.6:
            scale_factor = random.uniform(0.4, 0.8)
            small_w = max(16, int(w * scale_factor))
            small_h = max(16, int(h * scale_factor))
            rescaled = []
            for i in range(t):
                low_res = cv2.resize(out[i], (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                back_res = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
                rescaled.append(back_res)
            out = np.array(rescaled, dtype=np.uint8)

        # 2. JPEG Compression artifacts (Quality 20 to 90)
        if random.random() < 0.7:
            q = random.randint(20, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            jpeg_frames = []
            for i in range(t):
                bgr = cv2.cvtColor(out[i], cv2.COLOR_RGB2BGR)
                _, enc = cv2.imencode(".jpg", bgr, encode_param)
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                jpeg_frames.append(rgb)
            out = np.array(jpeg_frames, dtype=np.uint8)

        # 3. Gaussian or Motion Blur
        if random.random() < 0.5:
            if random.random() < 0.5:
                # Gaussian blur
                k = random.choice([3, 5, 7])
                out = np.array([cv2.GaussianBlur(f, (k, k), 0) for f in out], dtype=np.uint8)
            else:
                # Motion blur in random angle
                size = random.randint(3, 9)
                kernel_motion = np.zeros((size, size))
                kernel_motion[int((size - 1) / 2), :] = np.ones(size)
                kernel_motion = kernel_motion / size
                out = np.array([cv2.filter2D(f, -1, kernel_motion) for f in out], dtype=np.uint8)

        # 4. Brightness, Contrast & Sensor Noise
        if random.random() < 0.6:
            alpha = random.uniform(0.7, 1.3)  # Contrast
            beta = random.uniform(-30, 30)    # Brightness
            noise_std = random.uniform(0, 12) # Additive sensor noise
            noisy_list = []
            for i in range(t):
                f_float = out[i].astype(np.float32) * alpha + beta
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, size=f_float.shape)
                    f_float += noise
                noisy_list.append(np.clip(f_float, 0, 255).astype(np.uint8))
            out = np.array(noisy_list, dtype=np.uint8)

        # 5. WebRTC Frame Drops & Temporal Freezes
        if random.random() < 0.35 and t > 4:
            # Randomly duplicate frame to simulate packet drop freeze
            drop_idx = random.randint(1, t - 2)
            out[drop_idx] = out[drop_idx - 1]
            if random.random() < 0.5:
                out[drop_idx + 1] = out[drop_idx]

        return out


def apply_degradation_pipeline(clip_batch: np.ndarray) -> np.ndarray:
    """Helper for DataLoader worker transform."""
    aug = DualClassDegradation(p=0.85)
    return aug(clip_batch)
