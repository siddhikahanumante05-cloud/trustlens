"""Windower module for extracting 3-second analysis windows with latest-wins queue."""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from app.config import settings
from app.decoder import ChunkDecoder
from app.metrics import DROPPED_WINDOWS

logger = logging.getLogger("trustlens.windower")

WINDOW_FRAMES_TOTAL = int(settings.WINDOW_SEC * settings.FPS)  # 45 frames for 3.0s at 15 fps
SAMPLED_FRAMES_COUNT = 16  # Exactly 16 sampled frames for models
WINDOW_AUDIO_SAMPLES = int(settings.WINDOW_SEC * 16000)  # 48,000 samples


@dataclass
class AnalysisWindow:
    """A 3.0s window ready for pre-processing and model inference."""
    window_id: int
    t_start: float
    t_end: float
    frames: List[np.ndarray]  # 16 sampled frames (224, 224, 3)
    timestamps: List[float]   # 16 frame timestamps
    audio: np.ndarray         # 48,000 samples (16kHz s16le)
    challenges: List[Dict[str, Any]] = field(default_factory=list)
    is_warmup: bool = False


class Windower:
    """Manages window sliding, frame sub-sampling, audio alignment, and latest-wins queue."""

    def __init__(self, decoder: ChunkDecoder, stride_sec: float = 1.0, max_queue_size: int = 2):
        self.decoder = decoder
        self.stride_sec = stride_sec
        self.max_queue_size = max_queue_size

        self.queue: asyncio.Queue[AnalysisWindow] = asyncio.Queue(maxsize=max_queue_size)
        self.registered_challenges: List[Dict[str, Any]] = []
        self._running = False
        self._window_counter = 0

    def register_challenge(self, challenge_event: Dict[str, Any]):
        """Register a light or phrase challenge for window overlap matching."""
        self.registered_challenges.append(challenge_event)

    def extract_current_window(self) -> Optional[AnalysisWindow]:
        """Sample a 3-second window from the decoder's ring buffers."""
        raw_frames = self.decoder.get_latest_frames(WINDOW_FRAMES_TOTAL)
        raw_audio = self.decoder.get_latest_audio(WINDOW_AUDIO_SAMPLES)

        # Warm-up check: First 3s require full video frames buffer
        if len(raw_frames) < WINDOW_FRAMES_TOTAL:
            return None

        # Sample exactly 16 evenly spaced frames
        frame_indices = np.linspace(0, len(raw_frames) - 1, SAMPLED_FRAMES_COUNT, dtype=int)
        sampled_frames = [raw_frames[i][1] for i in frame_indices]
        sampled_timestamps = [raw_frames[i][0] for i in frame_indices]

        # Audio padding if during warm-up
        if len(raw_audio) < WINDOW_AUDIO_SAMPLES:
            audio_padded = np.zeros(WINDOW_AUDIO_SAMPLES, dtype=np.int16)
            audio_padded[-len(raw_audio):] = raw_audio
            audio_data = audio_padded
        else:
            audio_data = raw_audio[-WINDOW_AUDIO_SAMPLES:]

        t_start = sampled_timestamps[0]
        t_end = sampled_timestamps[-1]

        # Filter overlapping challenges
        # Challenge overlaps if its timestamp falls within [t_start - 1.0, t_end + 1.0]
        overlapping = []
        for ch in self.registered_challenges:
            t0 = ch.get("t0", 0) / 1000.0 if ch.get("t0") else 0
            if t0 == 0 or (t_start - 1.0 <= t0 <= t_end + 1.0):
                overlapping.append(ch)

        self._window_counter += 1
        return AnalysisWindow(
            window_id=self._window_counter,
            t_start=t_start,
            t_end=t_end,
            frames=sampled_frames,
            timestamps=sampled_timestamps,
            audio=audio_data,
            challenges=overlapping,
            is_warmup=False,
        )

    def extract_all_windows(self) -> List[AnalysisWindow]:
        """Extract all sliding windows sequentially across the full decoded buffer for offline/file analysis."""
        with self.decoder.lock:
            all_frames = list(self.decoder.frames_buffer)
            all_audio = np.array(self.decoder.audio_buffer, dtype=np.int16)

        if len(all_frames) < WINDOW_FRAMES_TOTAL:
            return []

        stride_frames = int(round(self.stride_sec * settings.FPS))
        windows = []
        max_start = len(all_frames) - WINDOW_FRAMES_TOTAL

        for start_idx in range(0, max_start + 1, max(1, stride_frames)):
            window_raw_frames = all_frames[start_idx : start_idx + WINDOW_FRAMES_TOTAL]
            audio_start = int(start_idx * (16000 / settings.FPS))
            audio_end = audio_start + WINDOW_AUDIO_SAMPLES

            if audio_end <= len(all_audio):
                audio_slice = all_audio[audio_start:audio_end]
            else:
                audio_padded = np.zeros(WINDOW_AUDIO_SAMPLES, dtype=np.int16)
                if audio_start < len(all_audio):
                    available = all_audio[audio_start:]
                    audio_padded[:len(available)] = available
                audio_slice = audio_padded

            frame_indices = np.linspace(0, len(window_raw_frames) - 1, SAMPLED_FRAMES_COUNT, dtype=int)
            sampled_frames = [window_raw_frames[i][1] for i in frame_indices]
            sampled_timestamps = [window_raw_frames[i][0] for i in frame_indices]

            t_start = sampled_timestamps[0]
            t_end = sampled_timestamps[-1]

            overlapping = []
            for ch in self.registered_challenges:
                t0 = ch.get("t0", 0) / 1000.0 if ch.get("t0") else 0
                if t0 == 0 or (t_start - 1.0 <= t0 <= t_end + 1.0):
                    overlapping.append(ch)

            self._window_counter += 1
            windows.append(
                AnalysisWindow(
                    window_id=self._window_counter,
                    t_start=t_start,
                    t_end=t_end,
                    frames=sampled_frames,
                    timestamps=sampled_timestamps,
                    audio=audio_slice,
                    challenges=overlapping,
                    is_warmup=False,
                )
            )

        return windows

    async def emit_window(self, window: AnalysisWindow):
        """Put window in queue, enforcing latest-wins by dropping oldest if full."""
        if self.queue.full():
            try:
                # Drop oldest
                dropped = self.queue.get_nowait()
                DROPPED_WINDOWS.inc()
                logger.warning("Pipeline behind: Dropped window %d (latest-wins)", dropped.window_id)
            except asyncio.QueueEmpty:
                pass

        await self.queue.put(window)

    async def run_loop(self):
        """Periodic background task that extracts a window every stride_sec."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.stride_sec)
            if not self._running:
                break
            window = self.extract_current_window()
            if window is not None:
                await self.emit_window(window)

    def stop(self):
        """Stop the periodic window extraction."""
        self._running = False
