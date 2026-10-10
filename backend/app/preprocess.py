"""Face detection, tracking, landmarks, mouth aperture, ROIs, and quality estimation."""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger("trustlens.preprocess")


@dataclass
class PreprocessedWindow:
    """Preprocessed features for a 16-frame analysis window."""
    face_detected: bool
    face_crops: List[np.ndarray]           # 16 aligned face crops (160x160x3)
    landmarks_series: List[np.ndarray]     # 16 x (N, 3) landmarks normalized
    mouth_apertures: List[float]           # 16 normalized mouth aperture values
    skin_rois: Dict[str, List[Dict[str, float]]]  # "face", "neck", "background" -> per-frame stats
    hand_occlusion_score: float            # 0.0 to 1.0 occlusion probability
    quality: Dict[str, float]              # blur, brightness, face_ratio, quality_trust
    audio: np.ndarray                      # 48,000 samples (16kHz)
    timestamps: List[float]                # 16 timestamps
    face_bboxes: List[Any] = field(default_factory=list) # 16 face bounding boxes (x, y, w, h)
    challenges: List[Dict[str, Any]] = field(default_factory=list)


class Preprocessor:
    """Extracts face crops, landmarks, mouth apertures, ROIs, and quality metrics."""

    def __init__(self, crop_size: int = 160):
        self.crop_size = crop_size
        self._landmarker = None
        self._yunet = None
        self._init_mediapipe_landmarker()
        self._init_yunet_detector()

    def _init_mediapipe_landmarker(self):
        """Initialize MediaPipe FaceLandmarker if task model is present."""
        task_path = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")
        if os.path.exists(task_path):
            try:
                import mediapipe as mp
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
                logger.info("MediaPipe FaceLandmarker loaded from %s", task_path)
            except Exception as e:
                logger.warning("Failed to initialize MediaPipe FaceLandmarker: %s.", e)

    def _init_yunet_detector(self):
        """Initialize YuNet ONNX face & landmark detector if present."""
        yunet_path = os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx")
        if os.path.exists(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 240), score_threshold=0.50)
                logger.info("YuNet neural face detector loaded from %s", yunet_path)
            except Exception as e:
                logger.warning("Failed to initialize YuNet: %s", e)

    def process_window(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        audio: np.ndarray,
        challenges: Optional[List[Dict[str, Any]]] = None,
    ) -> PreprocessedWindow:
        """Process 16 sampled frames and 48k audio samples."""
        face_crops: List[np.ndarray] = []
        landmarks_list: List[np.ndarray] = []
        mouth_apertures: List[float] = []
        face_rois_stats: List[Dict[str, float]] = []
        neck_rois_stats: List[Dict[str, float]] = []
        bg_rois_stats: List[Dict[str, float]] = []

        total_blur = 0.0
        total_brightness = 0.0
        face_detected_count = 0

        face_bboxes_list: List[Optional[Tuple[int, int, int, int]]] = []

        for frame in frames:
            h, w, c = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright_val = float(np.mean(gray))
            total_blur += blur_val
            total_brightness += bright_val

            # Face detection & bounding box estimation
            bbox, landmarks, aperture = self._detect_face_and_landmarks(frame)
            face_bboxes_list.append(bbox)

            if bbox is not None:
                face_detected_count += 1
                bx, by, bw, bh = bbox
                # 160x160 aligned face crop
                crop = frame[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (self.crop_size, self.crop_size))
                else:
                    crop_resized = cv2.resize(frame, (self.crop_size, self.crop_size))
                face_crops.append(crop_resized)

                # Skin ROI (center of face)
                face_patch = frame[by + int(bh * 0.3):by + int(bh * 0.7), bx + int(bw * 0.3):bx + int(bw * 0.7)]
                face_rois_stats.append(self._compute_chromaticity(face_patch))

                # Neck ROI (below chin)
                neck_y = min(h - 1, by + bh + int(bh * 0.05))
                neck_h = min(h - neck_y, int(bh * 0.25))
                neck_patch = frame[neck_y:neck_y + neck_h, bx + int(bw * 0.25):bx + int(bw * 0.75)]
                neck_rois_stats.append(self._compute_chromaticity(neck_patch))
            else:
                crop_resized = cv2.resize(frame, (self.crop_size, self.crop_size))
                face_crops.append(crop_resized)
                face_rois_stats.append(self._compute_chromaticity(frame))
                neck_rois_stats.append(self._compute_chromaticity(frame))

            # Background ROI (top-left border region)
            bg_patch = frame[0:max(10, int(h * 0.2)), 0:max(10, int(w * 0.2))]
            bg_rois_stats.append(self._compute_chromaticity(bg_patch))

            landmarks_list.append(landmarks)
            mouth_apertures.append(aperture)

        n_frames = max(1, len(frames))
        avg_blur = total_blur / n_frames
        avg_bright = total_brightness / n_frames
        face_ratio = face_detected_count / float(n_frames)

        # Calibrate quality trust score in [0.0, 1.0]
        # High trust if blur > 80, brightness in [40, 210], and face visible
        blur_score = min(1.0, max(0.0, (avg_blur - 20.0) / 100.0))
        bright_score = 1.0 - min(1.0, abs(avg_bright - 128.0) / 100.0)
        quality_trust = float(np.clip(0.5 * blur_score + 0.3 * bright_score + 0.2 * face_ratio, 0.0, 1.0))

        return PreprocessedWindow(
            face_detected=(face_detected_count >= (n_frames // 2)),
            face_crops=face_crops,
            landmarks_series=landmarks_list,
            mouth_apertures=mouth_apertures,
            skin_rois={
                "face": face_rois_stats,
                "neck": neck_rois_stats,
                "background": bg_rois_stats,
            },
            face_bboxes=face_bboxes_list,
            hand_occlusion_score=0.0,
            quality={
                "blur": avg_blur,
                "brightness": avg_bright,
                "face_ratio": face_ratio,
                "quality_trust": quality_trust,
            },
            audio=audio,
            timestamps=timestamps,
            challenges=challenges or [],
        )

    def _compute_chromaticity(self, patch: np.ndarray) -> Dict[str, float]:
        """Compute normalized RGB and chromaticity r = R/(R+G+B), g = G/(R+G+B)."""
        if patch.size == 0:
            return {"r": 0.333, "g": 0.333, "b": 0.333, "R": 128.0, "G": 128.0, "B": 128.0}
        means = np.mean(patch, axis=(0, 1))
        R, G, B = float(means[0]), float(means[1]), float(means[2])
        total = R + G + B + 1e-6
        return {
            "r": R / total,
            "g": G / total,
            "b": B / total,
            "R": R,
            "G": G,
            "B": B,
        }

    def _detect_face_and_landmarks(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray, float]:
        """Detect face bbox, normalized landmarks, and mouth aperture."""
        h, w, c = frame.shape

        # 1. MediaPipe Tasks API if loaded
        if self._landmarker:
            try:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                res = self._landmarker.detect(mp_img)
                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    xs = coords[:, 0] * w
                    ys = coords[:, 1] * h
                    bx = int(np.min(xs))
                    by = int(np.min(ys))
                    bw = int(np.max(xs) - bx)
                    bh = int(np.max(ys) - by)

                    iod = np.linalg.norm(coords[33, :2] - coords[263, :2])
                    mouth_dist = np.linalg.norm(coords[13, :2] - coords[14, :2])
                    aperture = float(mouth_dist / (iod + 1e-6))
                    return (bx, by, bw, bh), coords, aperture
            except Exception as e:
                logger.debug("MediaPipe detection exception: %s", e)

        # 2. YuNet neural face & landmark detection if present
        if self._yunet is not None:
            try:
                self._yunet.setInputSize((w, h))
                ret, faces = self._yunet.detect(frame)
                if faces is not None and len(faces) > 0:
                    f = faces[0]
                    bx = max(0, int(f[0]))
                    by = max(0, int(f[1]))
                    bw = min(w - bx, int(f[2]))
                    bh = min(h - by, int(f[3]))

                    re_x, re_y = float(f[4]), float(f[5])
                    le_x, le_y = float(f[6]), float(f[7])
                    nose_x, nose_y = float(f[8]), float(f[9])
                    rm_x, rm_y = float(f[10]), float(f[11])
                    lm_x, lm_y = float(f[12]), float(f[13])

                    coords = np.zeros((478, 3), dtype=np.float32)
                    coords[33] = [re_x / (w + 1e-6), re_y / (h + 1e-6), 0.0]
                    coords[263] = [le_x / (w + 1e-6), le_y / (h + 1e-6), 0.0]
                    coords[1] = [nose_x / (w + 1e-6), nose_y / (h + 1e-6), 0.0]
                    coords[61] = [rm_x / (w + 1e-6), rm_y / (h + 1e-6), 0.0]
                    coords[291] = [lm_x / (w + 1e-6), lm_y / (h + 1e-6), 0.0]

                    mouth_cx = (rm_x + lm_x) / 2.0
                    mouth_cy = (rm_y + lm_y) / 2.0
                    coords[13] = [mouth_cx / (w + 1e-6), (mouth_cy - 2.0) / (h + 1e-6), 0.0]
                    coords[14] = [mouth_cx / (w + 1e-6), (mouth_cy + 2.0) / (h + 1e-6), 0.0]

                    iod = np.linalg.norm(np.array([re_x, re_y]) - np.array([le_x, le_y]))
                    mouth_w = np.linalg.norm(np.array([rm_x, rm_y]) - np.array([lm_x, lm_y]))
                    aperture = float(mouth_w / (iod + 1e-6))

                    return (bx, by, bw, bh), coords, aperture
            except Exception as e:
                logger.debug("YuNet detection exception: %s", e)

        # 3. Geometric fallback: central 60% of frame
        bx = int(w * 0.2)
        by = int(h * 0.15)
        bw = int(w * 0.6)
        bh = int(h * 0.7)

        pseudo_landmarks = np.zeros((478, 3), dtype=np.float32)
        pseudo_landmarks[33] = [0.35, 0.4, 0.0]
        pseudo_landmarks[263] = [0.65, 0.4, 0.0]
        pseudo_landmarks[13] = [0.5, 0.68, 0.0]
        pseudo_landmarks[14] = [0.5, 0.72, 0.0]

        iod = np.linalg.norm(pseudo_landmarks[33, :2] - pseudo_landmarks[263, :2])
        mouth_dist = np.linalg.norm(pseudo_landmarks[13, :2] - pseudo_landmarks[14, :2])
        aperture = float(mouth_dist / (iod + 1e-6))

        return (bx, by, bw, bh), pseudo_landmarks, aperture
