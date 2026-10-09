"""ChunkDecoder for persistent and clip-mode WebM decoding.
Extracts 15 fps 224x224 RGB video frames and 16 kHz mono PCM audio into ring buffers.
"""
import subprocess
import threading
import time
import logging
import collections
from typing import Optional, List, Tuple
import numpy as np
from app.config import settings
from app.metrics import FFMPEG_RESTARTS

logger = logging.getLogger("trustlens.decoder")

FRAME_WIDTH = 224
FRAME_HEIGHT = 224
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3
AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLE_BYTES = 2  # s16le (16-bit)
BUFFER_DURATION_SEC = 5.0
MAX_FRAMES_IN_BUFFER = int(BUFFER_DURATION_SEC * settings.FPS)  # 75 frames
MAX_AUDIO_SAMPLES_IN_BUFFER = int(BUFFER_DURATION_SEC * AUDIO_SAMPLE_RATE)  # 80000 samples


class ChunkDecoder:
    """Decodes streaming WebM chunks into aligned video and audio ring buffers."""

    def __init__(self, clip_mode: Optional[bool] = None):
        self.clip_mode = clip_mode if clip_mode is not None else settings.CLIP_MODE
        self.frames_buffer: collections.deque = collections.deque(maxlen=MAX_FRAMES_IN_BUFFER)
        self.audio_buffer: collections.deque = collections.deque(maxlen=MAX_AUDIO_SAMPLES_IN_BUFFER)
        self.lock = threading.Lock()

        self._running = False
        self._video_proc: Optional[subprocess.Popen] = None
        self._audio_proc: Optional[subprocess.Popen] = None
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._audio_sample_count = 0
        self.stream_start_time = time.time()

        if not self.clip_mode:
            self._start_persistent_processes()

    def _start_persistent_processes(self):
        """Start persistent ffmpeg subprocesses for video and audio decoding."""
        video_cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "webm",
            "-i", "pipe:0",
            "-vf", f"fps={settings.FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        audio_cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "webm",
            "-i", "pipe:0",
            "-vn",
            "-ac", "1",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            self._video_proc = subprocess.Popen(
                video_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * FRAME_BYTES,
            )
            self._audio_proc = subprocess.Popen(
                audio_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_BYTES,
            )
            self._running = True

            self._video_thread = threading.Thread(target=self._read_video_stdout, daemon=True)
            self._audio_thread = threading.Thread(target=self._read_audio_stdout, daemon=True)
            self._video_thread.start()
            self._audio_thread.start()
            logger.info("Persistent ffmpeg decoder processes started")
        except Exception as e:
            logger.error("Failed to start ffmpeg processes: %s", e)
            self._running = False

    def _read_video_stdout(self):
        """Continuously read raw RGB frames from video process stdout."""
        while self._running and self._video_proc and self._video_proc.stdout:
            try:
                raw_frame = self._video_proc.stdout.read(FRAME_BYTES)
                if not raw_frame or len(raw_frame) < FRAME_BYTES:
                    # Incomplete frame or EOF
                    if not self._running:
                        break
                    continue

                frame_np = np.frombuffer(raw_frame, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                with self.lock:
                    timestamp = (self._frame_count / float(settings.FPS)) + self.stream_start_time
                    self.frames_buffer.append((timestamp, frame_np))
                    self._frame_count += 1
            except Exception as e:
                if self._running:
                    logger.warning("Video read error: %s", e)
                break

    def _read_audio_stdout(self):
        """Continuously read raw 16kHz s16le PCM samples from audio process stdout."""
        chunk_samples = 1600  # 100 ms chunks
        chunk_bytes = chunk_samples * AUDIO_SAMPLE_BYTES
        while self._running and self._audio_proc and self._audio_proc.stdout:
            try:
                raw_audio = self._audio_proc.stdout.read(chunk_bytes)
                if not raw_audio:
                    if not self._running:
                        break
                    continue

                samples = np.frombuffer(raw_audio, dtype=np.int16)
                with self.lock:
                    self.audio_buffer.extend(samples)
                    self._audio_sample_count += len(samples)
            except Exception as e:
                if self._running:
                    logger.warning("Audio read error: %s", e)
                break

    def feed_chunk(self, data: bytes):
        """Feed a WebM fragment to the decoder."""
        if not data:
            return

        if self.clip_mode:
            self._decode_clip_standalone(data)
            return

        if not self._running:
            self._start_persistent_processes()

        # Tee chunk to both processes
        for proc in [self._video_proc, self._audio_proc]:
            if proc and proc.stdin:
                try:
                    proc.stdin.write(data)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    logger.warning("FFmpeg pipe broken (%s). Restarting decoder.", e)
                    FFMPEG_RESTARTS.inc()
                    self._restart()
                    break

    def _decode_clip_standalone(self, data: bytes):
        """Fallback: Decode a standalone WebM clip."""
        try:
            # Video
            v_res = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                    "-vf", f"fps={settings.FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                ],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if v_res.stdout:
                num_frames = len(v_res.stdout) // FRAME_BYTES
                for i in range(num_frames):
                    raw = v_res.stdout[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]
                    frame_np = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                    with self.lock:
                        timestamp = (self._frame_count / float(settings.FPS)) + self.stream_start_time
                        self.frames_buffer.append((timestamp, frame_np))
                        self._frame_count += 1

            # Audio
            a_res = subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
                    "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
                    "-f", "s16le", "pipe:1",
                ],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if a_res.stdout:
                samples = np.frombuffer(a_res.stdout, dtype=np.int16)
                with self.lock:
                    self.audio_buffer.extend(samples)
                    self._audio_sample_count += len(samples)
        except Exception as e:
            logger.error("Standalone clip decode failed: %s", e)

    def _restart(self):
        """Clean up and restart dead ffmpeg processes."""
        self.close()
        self._start_persistent_processes()

    def get_latest_frames(self, count: int) -> List[Tuple[float, np.ndarray]]:
        """Retrieve the latest `count` frames from the ring buffer."""
        with self.lock:
            buf = list(self.frames_buffer)
            return buf[-count:] if len(buf) >= count else buf

    def get_latest_audio(self, num_samples: int) -> np.ndarray:
        """Retrieve the latest `num_samples` from the audio ring buffer."""
        with self.lock:
            buf = list(self.audio_buffer)
            if len(buf) >= num_samples:
                return np.array(buf[-num_samples:], dtype=np.int16)
            return np.array(buf, dtype=np.int16)

    def get_buffer_stats(self) -> dict:
        """Inspect current buffer fill status."""
        with self.lock:
            return {
                "frames_count": len(self.frames_buffer),
                "audio_samples": len(self.audio_buffer),
                "total_frames_decoded": self._frame_count,
                "total_audio_samples_decoded": self._audio_sample_count,
            }

    def close(self):
        """Terminate ffmpeg processes and join threads."""
        self._running = False
        for proc in [self._video_proc, self._audio_proc]:
            if proc:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._video_proc = None
        self._audio_proc = None
