"""Centralized tunable thresholds and constants for TrustLens signals."""

# ============================================================================
# E1: Provenance / Hardware Checks
# ============================================================================
VIRTUAL_CAMERA_KEYWORDS = [
    "obs",
    "virtual",
    "manycam",
    "droidcam",
    "epoccam",
    "xsplit",
    "snap camera",
    "vysor",
    "iriun",
    "camo",
    "fake",
    "capture device",
]

# Real webcams exhibit natural hardware jitter (typically 3ms - 15ms std)
# Synthetic replays / programmatic streams often exhibit perfect timer intervals (< 1.0ms std)
JITTER_TOO_SMOOTH_STD_MS = 0.30
FRAME_DROP_RATIO_HIGH = 0.35

# ============================================================================
# E2: Light Response Challenge
# ============================================================================
LIGHT_CORR_FACE_THRESHOLD = 0.35  # Below this on face reflection -> suspicious
LIGHT_SNR_MIN = 2.0               # Minimum SNR for reliable decision
LIGHT_MAX_LAG_SEARCH_MS = 600     # Maximum lag to search for sync
LIGHT_LATE_LAG_MS = 120.0         # Latency threshold (evaluated only when TAP=caller)
LIGHT_NECK_FACE_CORR_MIN = 0.30   # Face vs neck correlation minimum
LIGHT_NECK_FACE_RATIO_MIN = 0.20  # Ratio of neck response vs face response

# ============================================================================
# E4: Identity Consistency & Flicker
# ============================================================================
IDENTITY_FLICKER_MIN_SIM = 0.60   # Cosine similarity between frames below this is suspicious
IDENTITY_FLICKER_MAX_STD = 0.15   # Standard deviation of embedding similarities
LANDMARK_JITTER_ENERGY_MAX = 0.08 # High frequency landmark jitter threshold

# ============================================================================
# E6: Lip & Speech Consistency
# ============================================================================
PHRASE_WER_HIGH = 0.40            # Word Error Rate vs expected phrase
BILABIAL_APERTURE_MAX = 0.12      # Maximum allowed mouth aperture during b/p/m phonemes
SYNC_OFFSET_MAX_MS = 200.0        # Audio-video sync offset threshold
SYNC_CONF_MIN = 1.5               # Minimum confidence score for AV sync

# ============================================================================
# E8: Quality Trust
# ============================================================================
QUALITY_BLUR_LAPLACIAN_MIN = 40.0
QUALITY_BRIGHTNESS_MIN = 35.0
QUALITY_BRIGHTNESS_MAX = 225.0
QUALITY_FACE_RATIO_MIN = 0.08
QUALITY_TRUST_MIN = 0.30

# ============================================================================
# Visual & Audio Deepfake Models
# ============================================================================
EDGENET_PROB_THRESHOLD = 0.70
CLIP_PROB_THRESHOLD = 0.70
VOICE_SPOOF_PROB_THRESHOLD = 0.60
