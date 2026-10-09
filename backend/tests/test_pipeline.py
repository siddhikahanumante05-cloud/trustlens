"""Tests for ChunkDecoder, Windower, Preprocessor, and end-to-end streaming."""
import asyncio
import os
import subprocess
import tempfile
import time
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.decoder import ChunkDecoder, FRAME_WIDTH, FRAME_HEIGHT
from app.windower import Windower, AnalysisWindow
from app.preprocess import Preprocessor

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_webm_file():
    """Generate a valid 4-second WebM file with VP8 video and Opus audio using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Generate test video and audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=4",
        "-c:v", "libvpx", "-b:v", "400k",
        "-c:a", "libopus", "-b:a", "64k",
        "-f", "webm",
        tmp_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_chunk_decoder_with_sample_webm(sample_webm_file):
    """Verify ChunkDecoder decodes 15 fps 224x224 RGB frames and 16kHz PCM audio."""
    decoder = ChunkDecoder(clip_mode=False)
    try:
        with open(sample_webm_file, "rb") as f:
            data = f.read()

        # Feed in two 1-second chunks
        chunk_size = len(data) // 4
        for i in range(4):
            decoder.feed_chunk(data[i * chunk_size : (i + 1) * chunk_size])
            time.sleep(0.1)

        # Allow background decoder threads to consume
        time.sleep(1.0)
        stats = decoder.get_buffer_stats()

        # Verify decoded frames and audio
        assert stats["total_frames_decoded"] > 20, f"Expected >20 frames, got {stats['total_frames_decoded']}"
        assert stats["total_audio_samples_decoded"] > 20000, f"Expected >20k audio samples, got {stats['total_audio_samples_decoded']}"

        latest_frames = decoder.get_latest_frames(5)
        assert len(latest_frames) == 5
        ts, frame = latest_frames[-1]
        assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert frame.dtype == np.uint8

        audio_latest = decoder.get_latest_audio(16000)
        assert len(audio_latest) == 16000
        assert audio_latest.dtype == np.int16
    finally:
        decoder.close()


def test_windower_sampling_and_latest_wins():
    """Verify Windower extracts 16 sampled frames, 48k audio, and enforces latest-wins."""
    decoder = ChunkDecoder(clip_mode=False)
    # Populate mock buffer in decoder
    with decoder.lock:
        now = time.time()
        for i in range(50):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), fill_value=i * 5, dtype=np.uint8)
            decoder.frames_buffer.append((now + (i / 15.0), frame))
        audio_samples = np.ones(60000, dtype=np.int16) * 100
        decoder.audio_buffer.extend(audio_samples)

    windower = Windower(decoder=decoder, stride_sec=1.0, max_queue_size=1)
    window = windower.extract_current_window()

    assert window is not None
    assert len(window.frames) == 16, f"Expected 16 frames, got {len(window.frames)}"
    assert len(window.timestamps) == 16
    assert len(window.audio) == 48000
    assert window.frames[0].shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    # Test latest-wins queue behavior
    async def run_queue_test():
        # Put first window
        await windower.emit_window(window)
        assert windower.queue.qsize() == 1

        # Put second window with queue full: should drop oldest and replace
        window_2 = AnalysisWindow(
            window_id=999,
            t_start=10.0,
            t_end=13.0,
            frames=window.frames,
            timestamps=window.timestamps,
            audio=window.audio,
        )
        await windower.emit_window(window_2)
        assert windower.queue.qsize() == 1
        popped = await windower.queue.get()
        assert popped.window_id == 999, "Latest-wins failed: expected window 999"

    asyncio.run(run_queue_test())
    decoder.close()


def test_preprocessor_window():
    """Verify Preprocessor extracts 160x160 face crops, mouth apertures, ROIs, and quality."""
    prep = Preprocessor(crop_size=160)

    # Construct 16 synthetic frames with face-like center colors
    frames = []
    timestamps = []
    for i in range(16):
        img = np.full((224, 224, 3), 100, dtype=np.uint8)
        # Face patch (skin tones ~ [210, 160, 140])
        img[40:180, 50:170] = [210, 160, 140]
        frames.append(img)
        timestamps.append(100.0 + i * (1.0 / 15.0))

    audio = np.zeros(48000, dtype=np.int16)
    out = prep.process_window(frames=frames, timestamps=timestamps, audio=audio)

    assert out.face_detected is True
    assert len(out.face_crops) == 16
    assert out.face_crops[0].shape == (160, 160, 3)
    assert len(out.mouth_apertures) == 16
    assert all(a > 0 for a in out.mouth_apertures)
    assert len(out.landmarks_series) == 16
    assert out.landmarks_series[0].shape[0] == 478

    # Check ROIs
    for roi in ["face", "neck", "background"]:
        assert roi in out.skin_rois
        assert len(out.skin_rois[roi]) == 16
        first_stat = out.skin_rois[roi][0]
        assert "r" in first_stat and "g" in first_stat and "b" in first_stat
        # Check chromaticity bounds
        assert 0.0 <= first_stat["r"] <= 1.0

    # Check Quality
    assert "blur" in out.quality
    assert "brightness" in out.quality
    assert "quality_trust" in out.quality
    assert 0.0 <= out.quality["quality_trust"] <= 1.0


def test_ws_live_binary_chunk_ingest(sample_webm_file):
    """Verify live WS accepts binary WebM chunks and runs pipeline without error."""
    with open(sample_webm_file, "rb") as f:
        data = f.read()

    with client.websocket_connect("/ws/analyze") as ws:
        # Start handshake with caller consent
        ws.send_text('{"type": "start", "client_time": 1000, "consent": true}')

        # Read initial warm-up check messages
        for _ in range(6):
            ws.receive_text()

        # Send binary chunks
        chunk_size = len(data) // 4
        for i in range(4):
            ws.send_bytes(data[i * chunk_size : (i + 1) * chunk_size])

        # Send challenge event
        challenge_json = '{"type": "challenge", "kind": "phrase", "expected": "Banana Pocket Mountain", "t0": 1000}'
        ws.send_text(challenge_json)

        # Connection should remain healthy and responsive
        ws.send_text('{"type": "clock", "client_time": 5555}')
        received_pong = False
        for _ in range(5):
            raw = ws.receive_text()
            if "clock" in raw:
                received_pong = True
                break
        assert received_pong is True


def test_twenty_second_stream_yields_eighteen_windows():
    """Acceptance test: streaming a 20s stream yields ~18 windows with correct timestamps and bounded memory."""
    decoder = ChunkDecoder(clip_mode=False)
    windower = Windower(decoder=decoder, stride_sec=1.0)
    windows_extracted = []

    try:
        start_ts = 1000.0
        # Simulate 20 seconds of stream (20 1-second chunks)
        for sec in range(1, 21):
            # Feed 1 second worth of frames (15 frames)
            with decoder.lock:
                for f_idx in range(15):
                    frame_ts = start_ts + (sec - 1) + (f_idx / 15.0)
                    dummy_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), sec, dtype=np.uint8)
                    decoder.frames_buffer.append((frame_ts, dummy_frame))
                # Feed 1 second worth of audio (16,000 samples)
                dummy_audio = np.ones(16000, dtype=np.int16) * sec
                decoder.audio_buffer.extend(dummy_audio)

            # Extract window if available
            win = windower.extract_current_window()
            if win is not None:
                windows_extracted.append(win)

        # 20 seconds of video with 3s window and 1s stride yields 18 windows (from t=3 to t=20)
        assert 17 <= len(windows_extracted) <= 19, f"Expected ~18 windows, got {len(windows_extracted)}"

        # Verify monotonic timestamps
        for i in range(1, len(windows_extracted)):
            assert windows_extracted[i].t_start > windows_extracted[i - 1].t_start
            assert windows_extracted[i].t_end > windows_extracted[i - 1].t_end

        # Verify memory bounds (ring buffer limits)
        assert len(decoder.frames_buffer) <= 75, f"Frame buffer grew beyond 75: {len(decoder.frames_buffer)}"
        assert len(decoder.audio_buffer) <= 80000, f"Audio buffer grew beyond 80k: {len(decoder.audio_buffer)}"
    finally:
        decoder.close()
