"""E9 Presentation Attack Detection (PAD) & Screen Replay signal module.
Detects physical display re-capture and mobile phone screen replay artifacts:
1. Specular glass reflections & glare hotspots on mobile screen glass
2. Motion decoupling between facial movements in video and screen reflections
3. Planar reflection field uniformity vs 3D facial curvature
4. Veiling glare and dynamic range pedestal from screen glass
5. 2D FFT periodic Moiré subpixel patterns
6. Subpixel chrominance lattice edges (RGB display stripes vs skin subsurface scattering)
7. Rectangular device screen bezels / borders surrounding the face
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from app.signals.thresholds import (
    SCREEN_GLARE_THRESHOLD,
    REFLECTION_DECOUPLING_THRESHOLD,
    PLANAR_REFLECTION_THRESHOLD,
    MOIRE_SPIKE_THRESHOLD,
    CHROMA_LATTICE_RATIO_THRESHOLD,
    DEVICE_BEZEL_THRESHOLD,
    SCREEN_REPLAY_RISK_THRESHOLD,
)


@dataclass
class PresentationResult:
    replay_score: float             # 0.0 (genuine 3D live capture) to 1.0 (screen replay)
    glare_score: float              # Glass specular reflection detection
    decoupling_score: float         # Motion decoupling between facial motion and glass reflection
    planar_score: float             # Planar reflection field fit vs 3D facial curvature
    moire_score: float              # 2D FFT periodic harmonic peakiness
    chroma_lattice_score: float     # Subpixel RGB chrominance edge variance
    bezel_score: float              # Rectangular device screen border / bezel detection
    veiling_glare_score: float = 0.0 # Elevated pedestal / veiling glare
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def compute_fft_moire_score(crop_gray: np.ndarray) -> float:
    """
    Computes 2D Fast Fourier Transform peak-to-average spike ratio
    in the mid-to-high spatial frequency band.
    Periodic screen grids produce isolated harmonic peaks (spike > 4.4).
    Natural human skin exhibits smooth 1/f^gamma decay (spike <= 3.2).
    """
    if crop_gray is None or crop_gray.size == 0:
        return 0.0
    h, w = crop_gray.shape
    if h < 32 or w < 32:
        return 0.0

    f = np.fft.fft2(crop_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)

    # Annular band between 25% and 85% of Nyquist frequency
    mask = (dist > 0.25 * max_r) & (dist < 0.85 * max_r)
    vals = mag[mask]
    if len(vals) < 100:
        return 0.0

    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))
    max_v = float(np.max(vals))

    spike_ratio = (max_v - mean_v) / (std_v + 1e-6)
    return float(spike_ratio)


def compute_chroma_lattice_score(crop_rgb: np.ndarray) -> float:
    """
    Measures ratio of chrominance edge energy to luminance edge energy.
    Human skin has strong subsurface scattering (smooth Cr/Cb gradients, ratio <= 0.08).
    Screens have discrete RGB subpixel stripes producing sharp chrominance edges (ratio >= 0.28).
    """
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 32 or crop_rgb.shape[1] < 32:
        return 0.0

    ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    var_y = float(cv2.Laplacian(y, cv2.CV_64F).var())
    if var_y < 5.0:  # Flat / blurred frame
        return 0.0

    var_cr = float(cv2.Laplacian(cr, cv2.CV_64F).var())
    var_cb = float(cv2.Laplacian(cb, cv2.CV_64F).var())

    ratio = (var_cr + var_cb) / (2.0 * var_y)
    return float(ratio)


def detect_device_bezel(
    frame: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Detects prominent rectangular device borders or screen bezels framing the face.
    Requires a face bounding box to avoid mistaking normal room background edges
    (doors, walls, furniture) for a mobile phone bezel.
    """
    if frame is None or frame.size == 0 or face_bbox is None:
        return 0.0

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame

    bx, by, bw, bh = face_bbox
    if bw < 20 or bh < 20:
        return 0.0

    # Define a localized search margin around the face (1.1x to 2.2x the face size)
    pad_x = int(bw * 0.45)
    pad_y = int(bh * 0.45)
    x0, y0 = max(0, bx - pad_x), max(0, by - pad_y)
    x1, y1 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 30 or roi.shape[1] < 30:
        return 0.0

    edges = cv2.Canny(roi, 60, 180)

    # Zero out interior face region within the ROI
    rel_bx0, rel_bx1 = max(0, bx - x0 + int(bw * 0.15)), min(roi.shape[1], bx - x0 + int(bw * 0.85))
    rel_by0, rel_by1 = max(0, by - y0 + int(bh * 0.15)), min(roi.shape[0], by - y0 + int(bh * 0.85))
    edges[rel_by0:rel_by1, rel_bx0:rel_bx1] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(min(roi.shape[0], roi.shape[1]) * 0.35),
        maxLineGap=10,
    )

    if lines is None:
        return 0.0

    h_lines = 0
    v_lines = 0
    roi_w, roi_h = roi.shape[1], roi.shape[0]

    for line in lines:
        coords = line.ravel()
        if len(coords) < 4:
            continue
        lx1, ly1, lx2, ly2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        dx = abs(lx2 - lx1)
        dy = abs(ly2 - ly1)
        if dy < 6 and dx > int(roi_w * 0.30):
            h_lines += 1
        elif dx < 6 and dy > int(roi_h * 0.30):
            v_lines += 1

    # Phone screen bezel requires both horizontal and vertical edges framing the face
    if h_lines >= 1 and v_lines >= 1:
        return float(min(1.0, 0.40 + 0.15 * (h_lines + v_lines)))
    elif h_lines >= 2 or v_lines >= 2:
        return float(min(0.65, 0.30 + 0.10 * max(h_lines, v_lines)))

    return 0.0


def detect_specular_glass_glare(crop_rgb: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Detects specular glass reflections typical of mobile and monitor screens.
    Distinguishes flat glass reflections from natural skin highlights via:
    1. Saturated/near-saturated luminance patches (Y >= 225, or R,G,B >= 235)
    2. Perimeter edge gradient sharpness (glass reflections have sharp boundaries,
       whereas skin subsurface scattering yields soft, smooth rolloffs)
    3. Local contrast ratio against surrounding non-glare skin
    4. Glare area ratio and boundary gradient magnitude
    """
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 16 or crop_rgb.shape[1] < 16:
        return 0.0, {"glare_ratio": 0.0, "edge_sharpness": 0.0, "contrast_ratio": 0.0}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if len(crop_rgb.shape) == 3 else crop_rgb
    h, w = gray.shape
    total_pixels = float(h * w)

    # 1. Specular highlight mask: bright pixels
    if len(crop_rgb.shape) == 3:
        glare_mask = ((gray >= 225) | (
            (crop_rgb[:, :, 0] >= 235) & 
            (crop_rgb[:, :, 1] >= 235) & 
            (crop_rgb[:, :, 2] >= 235)
        )).astype(np.uint8)
    else:
        glare_mask = (gray >= 225).astype(np.uint8)

    glare_pixel_count = int(np.sum(glare_mask))
    glare_ratio = glare_pixel_count / total_pixels

    # Real glass reflections create a distinct patch of at least 25 pixels
    if glare_pixel_count < 25 or glare_ratio < 0.003:
        return 0.0, {
            "glare_ratio": round(glare_ratio, 4),
            "edge_sharpness": 0.0,
            "contrast_ratio": 0.0,
        }

    # 2. Boundary sharpness: compute gradient magnitude around glare perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(glare_mask, kernel, iterations=1)
    boundary_mask = dilated - glare_mask

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    boundary_pixels = boundary_mask > 0
    if np.sum(boundary_pixels) > 0:
        edge_sharpness = float(np.mean(grad_mag[boundary_pixels]))
    else:
        edge_sharpness = 0.0

    # 3. Local contrast ratio: mean glare brightness vs local surrounding annular ring
    ring_mask = cv2.dilate(glare_mask, kernel, iterations=4) - dilated
    glare_mean = float(np.mean(gray[glare_mask > 0]))
    if np.sum(ring_mask > 0) > 0:
        surround_mean = float(np.mean(gray[ring_mask > 0]))
        contrast_ratio = (glare_mean - surround_mean) / (surround_mean + 1e-3)
    else:
        contrast_ratio = 0.0

    # Real skin highlights have soft transitions (edge_sharpness < 18.0) and lower local contrast (< 0.25).
    # Screen glass reflections have crisp boundaries (edge_sharpness >= 22.0) and high local contrast (>= 0.30).
    if edge_sharpness < 18.0 or contrast_ratio < 0.25:
        return 0.0, {
            "glare_ratio": round(glare_ratio, 4),
            "edge_sharpness": round(edge_sharpness, 2),
            "contrast_ratio": round(contrast_ratio, 3),
        }

    area_score = np.clip((glare_ratio - 0.010) / 0.05, 0.0, 1.0)
    sharpness_score = np.clip((edge_sharpness - 18.0) / 25.0, 0.0, 1.0)
    contrast_score = np.clip((contrast_ratio - 0.25) / 0.35, 0.0, 1.0)

    glare_score = float(np.clip(
        0.45 * sharpness_score + 0.30 * area_score + 0.25 * contrast_score,
        0.0, 1.0
    ))

    # Boost if both sharpness and contrast firmly match glass reflection
    if sharpness_score >= 0.40 and contrast_score >= 0.30:
        glare_score = float(min(1.0, glare_score + 0.25))

    return glare_score, {
        "glare_ratio": round(glare_ratio, 4),
        "edge_sharpness": round(edge_sharpness, 2),
        "contrast_ratio": round(contrast_ratio, 3),
    }


def detect_reflection_motion_decoupling(
    crops_rgb: List[np.ndarray],
    landmarks_series: Optional[List[np.ndarray]] = None,
    glare_present: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates motion decoupling between facial movements and specular reflection glare.
    Only evaluated if an actual specular glass glare patch is present on the screen.
    """
    if not glare_present or not crops_rgb or len(crops_rgb) < 3:
        return 0.0, {"reason": "no_glare_or_insufficient_frames", "decoupling_score": 0.0}

    n = min(len(crops_rgb), 16)
    glare_centroids = []
    face_motions = []

    # Track glare centroid in each frame
    for i in range(n):
        crop = crops_rgb[i]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
        # Specular glass candidate: pixels >= 225
        mask = (gray >= 225).astype(np.uint8)
        
        m = cv2.moments(mask)
        if m["m00"] > 15:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            glare_centroids.append((cx, cy))
        else:
            glare_centroids.append(None)

    # Frame-to-frame glare displacements
    glare_displacements = []
    valid_glare_pairs = 0
    for i in range(1, len(glare_centroids)):
        c0 = glare_centroids[i - 1]
        c1 = glare_centroids[i]
        if c0 is not None and c1 is not None:
            dist = np.sqrt((c1[0] - c0[0])**2 + (c1[1] - c0[1])**2)
            glare_displacements.append(float(dist))
            valid_glare_pairs += 1
        else:
            glare_displacements.append(0.0)

    if valid_glare_pairs < 2:
        return 0.0, {"reason": "glare_not_persistent", "decoupling_score": 0.0}

    # Facial motion across frames
    if landmarks_series and len(landmarks_series) >= n:
        for i in range(1, n):
            lm0 = landmarks_series[i - 1]
            lm1 = landmarks_series[i]
            if lm0 is not None and lm1 is not None and len(lm0) > 0 and len(lm1) > 0:
                diff = np.linalg.norm(lm1[:, :2] - lm0[:, :2], axis=1)
                face_motions.append(float(np.mean(diff) * 100.0))
            else:
                face_motions.append(0.0)
    else:
        for i in range(1, n):
            g0 = cv2.cvtColor(crops_rgb[i - 1], cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(crops_rgb[i], cv2.COLOR_RGB2GRAY)
            diff = float(np.mean(np.abs(g1.astype(np.float32) - g0.astype(np.float32))))
            face_motions.append(diff)

    mean_face_motion = float(np.mean(face_motions)) if face_motions else 0.0
    mean_glare_motion = float(np.mean(glare_displacements)) if glare_displacements else 0.0

    decoupling_score = 0.0
    # Case 1: Active face inside video, static reflection on mobile screen glass
    if mean_face_motion > 0.8 and mean_glare_motion < 0.6:
        decoupling_score = float(np.clip((mean_face_motion - 0.8) / 2.5, 0.40, 0.95))
    elif len(face_motions) >= 4 and np.std(face_motions) > 0.1 and np.std(glare_displacements) > 0.1:
        # Trajectory correlation
        r = float(np.corrcoef(face_motions, glare_displacements)[0, 1])
        if r < 0.20:
            decoupling_score = float(np.clip((0.20 - r) / 0.80, 0.0, 0.85))

    return decoupling_score, {
        "mean_face_motion": round(mean_face_motion, 3),
        "mean_glare_motion": round(mean_glare_motion, 3),
        "decoupling_score": round(decoupling_score, 3),
    }


def detect_planar_reflection_uniformity(
    crop_rgb: np.ndarray,
    glare_present: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Measures how closely the specular reflection field conforms to a 2D plane (Z = 0).
    Only evaluated if actual specular glass glare is present.
    """
    if not glare_present or crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 24 or crop_rgb.shape[1] < 24:
        return 0.0, {"r2": 0.0, "planar_score": 0.0}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if len(crop_rgb.shape) == 3 else crop_rgb
    h, w = gray.shape

    # Focus on the highest luminance pixels in the reflection region
    threshold = max(180, int(np.percentile(gray, 75)))
    y_coords, x_coords = np.where(gray >= threshold)

    if len(x_coords) < 30:
        return 0.0, {"r2": 0.0, "planar_score": 0.0}

    vals = gray[y_coords, x_coords].astype(np.float64)
    var_total = float(np.var(vals))
    if var_total < 5.0:
        return 0.50, {"r2": 1.0, "planar_score": 0.50}

    # Normalize coordinates to [-1, 1]
    norm_x = (x_coords - w / 2.0) / (w / 2.0)
    norm_y = (y_coords - h / 2.0) / (h / 2.0)

    # Design matrix [x, y, 1]
    A = np.column_stack([norm_x, norm_y, np.ones_like(norm_x)])
    coeffs, residuals, rank, s = np.linalg.lstsq(A, vals, rcond=None)
    pred = A @ coeffs
    var_res = float(np.var(vals - pred))

    r2 = max(0.0, 1.0 - (var_res / (var_total + 1e-6)))
    planar_score = float(np.clip((r2 - 0.40) / 0.40, 0.0, 1.0))

    return planar_score, {
        "r2": round(r2, 3),
        "planar_score": round(planar_score, 3),
    }


def detect_veiling_glare(
    crop_rgb: np.ndarray,
    glare_present: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Detects veiling glare and dynamic range compression from light reflecting off display glass.
    Only evaluated if specular glass glare is present.
    """
    if not glare_present or crop_rgb is None or crop_rgb.size == 0:
        return 0.0, {"black_level": 0.0, "veiling_score": 0.0}

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if len(crop_rgb.shape) == 3 else crop_rgb
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))

    elevated_pedestal = float(np.clip((p5 - 35.0) / 40.0, 0.0, 1.0))
    compressed_dynamic_range = float(np.clip((140.0 - (p95 - p5)) / 70.0, 0.0, 1.0))

    veiling_score = float(np.clip(0.6 * elevated_pedestal + 0.4 * compressed_dynamic_range, 0.0, 1.0))

    return veiling_score, {
        "black_level": round(p5, 2),
        "dynamic_range": round(p95 - p5, 2),
        "veiling_score": round(veiling_score, 3),
    }


def analyze_presentation_replay(
    face_crops: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
    landmarks_series: Optional[List[np.ndarray]] = None,
    face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> PresentationResult:
    """
    Comprehensive multi-cue Presentation Attack Detection (PAD).
    Enforces PRIMARY EVIDENCE GATING:
    - To flag a mobile screen replay, there MUST be direct physical evidence on the face:
      1. Saturated specular glass glare with sharp boundary gradient and contrast (glare_score >= 0.35)
      2. Display subpixel artifacts (2D FFT Moiré spike >= 0.40 or chroma lattice ratio >= 0.40)
    - Normal indoor diffuse lighting on real human skin and background room edges (walls, doors)
      will NEVER trigger a replay warning.
    """
    if not face_crops:
        return PresentationResult(
            replay_score=0.0,
            glare_score=0.0,
            decoupling_score=0.0,
            planar_score=0.0,
            moire_score=0.0,
            chroma_lattice_score=0.0,
            bezel_score=0.0,
            veiling_glare_score=0.0,
            flags=[],
            details={"reason": "no_face_crops"},
        )

    # 1. Specular glare, Moiré, and Chrominance analysis across crops
    spikes = []
    chroma_ratios = []
    glares = []

    for crop in face_crops[:16]:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
        spikes.append(compute_fft_moire_score(gray))
        if len(crop.shape) == 3:
            chroma_ratios.append(compute_chroma_lattice_score(crop))
            g_score, _ = detect_specular_glass_glare(crop)
            glares.append(g_score)

    mean_spike = float(np.mean(spikes)) if spikes else 0.0
    mean_chroma = float(np.mean(chroma_ratios)) if chroma_ratios else 0.0
    max_glare = float(np.max(glares)) if glares else 0.0
    mean_glare = float(np.mean(glares)) if glares else 0.0

    glare_is_present = (max_glare >= SCREEN_GLARE_THRESHOLD)

    # 2. Conditional Reflection Cues (only evaluated if glass glare exists)
    planars = []
    veilings = []
    for crop in face_crops[:16]:
        if len(crop.shape) == 3:
            p_score, _ = detect_planar_reflection_uniformity(crop, glare_present=glare_is_present)
            planars.append(p_score)
            v_score, _ = detect_veiling_glare(crop, glare_present=glare_is_present)
            veilings.append(v_score)

    max_planar = float(np.max(planars)) if planars else 0.0
    mean_veiling = float(np.mean(veilings)) if veilings else 0.0

    # Motion Decoupling (only evaluated if glass glare exists)
    decoupling_score, dec_details = detect_reflection_motion_decoupling(
        crops_rgb=face_crops[:16],
        landmarks_series=landmarks_series,
        glare_present=glare_is_present,
    )

    # 3. Bezel evaluation (requires face bounding box to avoid room edges)
    bezel_scores = []
    if frames and face_bboxes:
        n_frames = min(len(frames), 8)
        for i in range(n_frames):
            f = frames[i]
            bbox = face_bboxes[i] if i < len(face_bboxes) else None
            bezel_scores.append(detect_device_bezel(f, bbox))
    max_bezel = float(np.max(bezel_scores)) if bezel_scores else 0.0

    # 4. Normalized score mappings
    moire_norm = float(np.clip((mean_spike - MOIRE_SPIKE_THRESHOLD) / 1.8, 0.0, 1.0))
    chroma_norm = float(np.clip((mean_chroma - CHROMA_LATTICE_RATIO_THRESHOLD) / 0.35, 0.0, 1.0))
    glare_norm = float(np.clip(max_glare, 0.0, 1.0))
    planar_norm = float(np.clip(max_planar, 0.0, 1.0)) if glare_is_present else 0.0
    decoupling_norm = float(np.clip(decoupling_score, 0.0, 1.0)) if glare_is_present else 0.0
    bezel_norm = float(np.clip(max_bezel, 0.0, 1.0)) if (glare_is_present or moire_norm >= 0.35 or chroma_norm >= 0.35) else 0.0

    # 5. Composite Replay Scoring with Corroborated Display Gating
    has_glass_reflection = (glare_norm >= SCREEN_GLARE_THRESHOLD)
    has_subpixel_pattern = (moire_norm >= 0.40 or chroma_norm >= 0.40)
    has_decoupling = (decoupling_norm >= REFLECTION_DECOUPLING_THRESHOLD)
    has_planar = (planar_norm >= PLANAR_REFLECTION_THRESHOLD)
    has_bezel = (bezel_norm >= DEVICE_BEZEL_THRESHOLD)

    # Corroborated Physical Display Gating:
    # A natural specular highlight on human skin (forehead, nose tip) under indoor lighting
    # must NEVER be classified as screen glare or replay unless corroborated by independent physical display cues.
    corroborated_display = bool(has_decoupling or has_planar or has_subpixel_pattern or has_bezel)

    flags = []
    if has_glass_reflection and corroborated_display:
        flags.append("SPECULAR_SCREEN_GLARE")
        confirmed_glare = glare_norm
        composite = float(np.clip(
            0.45 * glare_norm + 0.35 * max(decoupling_norm, planar_norm) + 0.25 * max(moire_norm, chroma_norm, bezel_norm) + 0.15,
            0.65, 1.0
        ))
    elif has_subpixel_pattern and (has_bezel or has_decoupling):
        confirmed_glare = 0.0
        composite = float(np.clip(0.50 * max(moire_norm, chroma_norm) + 0.35 * max(bezel_norm, decoupling_norm) + 0.20, 0.65, 1.0))
    elif has_subpixel_pattern:
        confirmed_glare = 0.0
        composite = float(np.clip(max(moire_norm, chroma_norm) * 0.80, 0.0, 1.0))
    elif has_bezel and has_decoupling:
        confirmed_glare = 0.0
        composite = float(np.clip(0.50 * bezel_norm + 0.50 * decoupling_norm, 0.65, 1.0))
    else:
        # Real 3D human in normal room lighting -> strictly 0.0 (Authentic, NO false positives)
        confirmed_glare = 0.0
        composite = 0.0

    if has_decoupling and corroborated_display:
        flags.append("REFLECTION_MOTION_DECOUPLED")
    if has_planar and corroborated_display:
        flags.append("PLANAR_SCREEN_REFLECTION")
    if moire_norm >= 0.40:
        flags.append("MOIRE_PATTERN_DETECTED")
    if chroma_norm >= 0.40:
        flags.append("SUBPIXEL_LATTICE_DETECTED")
    if bezel_norm >= DEVICE_BEZEL_THRESHOLD:
        flags.append("DEVICE_BEZEL_DETECTED")
    if composite >= SCREEN_REPLAY_RISK_THRESHOLD:
        flags.append("SCREEN_REPLAY_DETECTED")

    return PresentationResult(
        replay_score=composite,
        glare_score=confirmed_glare,
        decoupling_score=decoupling_norm if corroborated_display else 0.0,
        planar_score=planar_norm if corroborated_display else 0.0,
        moire_score=moire_norm,
        chroma_lattice_score=chroma_norm,
        bezel_score=bezel_norm,
        veiling_glare_score=round(mean_veiling, 3) if corroborated_display else 0.0,
        flags=flags,
        details={
            "mean_spike": round(mean_spike, 3),
            "mean_chroma_ratio": round(mean_chroma, 4),
            "max_glare": round(max_glare, 3),
            "mean_glare": round(mean_glare, 3),
            "max_planar": round(max_planar, 3),
            "decoupling_score": round(decoupling_score, 3),
            "max_bezel": round(max_bezel, 3),
            "mean_veiling": round(mean_veiling, 3),
        },
    )


