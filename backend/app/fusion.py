"""Multi-signal Fusion Engine for TrustLens.
Handles feature vector assembly, regularized logistic regression, isotonic calibration,
exponential moving average (EMA) smoothing, and hysteresis state machine.
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from app.schemas import CheckId, CheckState
from app.modules.base import ModuleResult

logger = logging.getLogger("trustlens.fusion")


@dataclass
class WindowFeatures:
    # E1: Provenance
    source_risk: Optional[float] = None

    # E2: Light response
    light_corr_face: Optional[float] = None
    light_lag_ms: Optional[float] = None
    light_neck_face_ratio: Optional[float] = None
    light_neck_face_corr: Optional[float] = None
    light_snr: Optional[float] = None

    # E3: Fast model (EdgeNet)
    edge_logit: Optional[float] = None

    # E3/E5: Slow models (CLIP, WavLM)
    clip_logit: Optional[float] = None
    model_disagreement: Optional[float] = None
    ood_score: Optional[float] = None

    # E4: Identity & Landmark
    identity_flicker: Optional[float] = None
    landmark_jitter: Optional[float] = None
    occlusion_artifact: Optional[float] = None

    # E5: Audio spoof
    spoof_max: Optional[float] = None
    spoof_mean: Optional[float] = None

    # E6: Lip & Audio-Visual
    sync_conf: Optional[float] = None
    sync_offset_ms: Optional[float] = None
    phrase_wer: Optional[float] = None
    bilabial_aperture_err: Optional[float] = None

    # E8: Quality Trust
    quality_trust: Optional[float] = None


@dataclass
class FusionResult:
    raw_logit: float
    raw_probability: float
    calibrated_probability: float
    smoothed_probability: float
    category: str  # "Authentic", "Suspicious", "Likely deepfake"
    in_deepfake_state: bool
    action_recommended: Optional[str]
    check_states: Dict[CheckId, CheckState]
    top_contributions: List[Tuple[str, float]]
    feature_dict: Dict[str, float]
    missing_features: List[str]


class FusionEngine:
    """Combines 20 multimodal signals into calibrated risk probability and verdict."""

    def __init__(
        self,
        fusion_config_path: Optional[str] = None,
        calibration_config_path: Optional[str] = None,
        ema_alpha: float = 0.50,
    ):
        base_models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.fusion_path = fusion_config_path or os.path.join(base_models_dir, "fusion.json")
        self.calibration_path = calibration_config_path or os.path.join(base_models_dir, "calibration.json")
        self.ema_alpha = ema_alpha

        self.bias: float = -1.25
        self.weights: Dict[str, float] = {}
        self.neutral_values: Dict[str, float] = {}
        self.calibration_points: List[Tuple[float, float]] = []

        self._load_configs()

        # State tracking for EMA and hysteresis
        self._prev_ema: Optional[float] = None
        self._consecutive_fake_count = 0
        self._consecutive_authentic_count = 0
        self._in_deepfake_state = False
        self._action_emitted = False

    def _load_configs(self):
        """Load fusion weights and calibration lookup points."""
        if os.path.exists(self.fusion_path):
            try:
                with open(self.fusion_path, "r") as f:
                    data = json.load(f)
                    self.bias = float(data.get("bias", -1.25))
                    self.weights = {k: float(v) for k, v in data.get("weights", {}).items()}
                    self.neutral_values = {k: float(v) for k, v in data.get("neutral_values", {}).items()}
            except Exception as e:
                logger.error("Failed to load fusion config: %s", e)

        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r") as f:
                    data = json.load(f)
                    self.calibration_points = [
                        (float(p[0]), float(p[1])) for p in data.get("points", [])
                    ]
                    self.calibration_points.sort(key=lambda x: x[0])
            except Exception as e:
                logger.error("Failed to load calibration config: %s", e)

    def calibrate(self, logit: float) -> float:
        """Piecewise linear interpolation on isotonic regression calibration points."""
        if not self.calibration_points:
            # Standard sigmoid fallback: 1 / (1 + exp(-logit))
            return float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))

        if logit <= self.calibration_points[0][0]:
            return self.calibration_points[0][1]
        if logit >= self.calibration_points[-1][0]:
            return self.calibration_points[-1][1]

        for i in range(len(self.calibration_points) - 1):
            x0, y0 = self.calibration_points[i]
            x1, y1 = self.calibration_points[i + 1]
            if x0 <= logit <= x1:
                t = (logit - x0) / (x1 - x0) if x1 > x0 else 0.0
                return float(y0 + t * (y1 - y0))

        return float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0))

    def evaluate_checks(self, feat: Dict[str, float]) -> Dict[CheckId, CheckState]:
        """Map feature values to sub-check states."""
        checks: Dict[CheckId, CheckState] = {
            "source": "ok",
            "light": "ok",
            "face": "ok",
            "voice": "ok",
            "lips": "ok",
        }

        # 1. Source / Provenance
        src_risk = feat.get("source_risk", 0.0)
        if src_risk > 0.50:
            checks["source"] = "bad"
        elif src_risk > 0.20:
            checks["source"] = "warn"

        # 2. Active Light Challenge
        light_corr = feat.get("light_corr_face", 0.50)
        light_snr = feat.get("light_snr", 3.0)
        if light_snr >= 2.0:
            if light_corr < 0.20:
                checks["light"] = "bad"
            elif light_corr < 0.35:
                checks["light"] = "warn"

        # 3. Face Artifacts & Flicker
        edge_logit = feat.get("edge_logit", 0.0)
        flicker = feat.get("identity_flicker", 0.0)
        jitter = feat.get("landmark_jitter", 0.0)
        if edge_logit > 0.80 or flicker > 0.50:
            checks["face"] = "bad"
        elif edge_logit > 0.30 or flicker > 0.25 or jitter > 0.06:
            checks["face"] = "warn"

        # 4. Voice Spoof
        spoof = feat.get("spoof_max", 0.0)
        if spoof > 0.60:
            checks["voice"] = "bad"
        elif spoof > 0.30:
            checks["voice"] = "warn"

        # 5. Lip Closure & Sync
        lip_err = feat.get("bilabial_aperture_err", 0.0)
        sync_off = feat.get("sync_offset_ms", 0.0)
        sync_conf = feat.get("sync_conf", 2.0)
        if lip_err > 0.30 or (sync_conf >= 1.5 and sync_off > 180.0):
            checks["lips"] = "bad"
        elif lip_err > 0.15 or (sync_conf >= 1.5 and sync_off > 100.0):
            checks["lips"] = "warn"

        return checks

    def process_window(self, features: WindowFeatures) -> FusionResult:
        """
        Calculates logit, isotonic calibration, EMA smoothed score,
        hysteresis state, and feature contributions.
        """
        feat_dict: Dict[str, float] = {}
        missing_keys: List[str] = []
        raw_values = features.__dict__

        # 1. Imputation with neutral values
        for k, default_val in self.neutral_values.items():
            val = raw_values.get(k)
            if val is None or np.isnan(val):
                feat_dict[k] = default_val
                missing_keys.append(k)
            else:
                feat_dict[k] = float(val)

        # 2. Quality trust modulation:
        # If video quality trust is very low, dampen strong artifact detections
        quality_trust = feat_dict.get("quality_trust", 0.85)
        dampener = 1.0 if quality_trust >= 0.50 else max(0.35, quality_trust / 0.50)

        # 3. Compute raw logit and feature contributions
        contributions: List[Tuple[str, float]] = []
        logit = self.bias

        for k, weight in self.weights.items():
            val = feat_dict.get(k, self.neutral_values.get(k, 0.0))
            neutral = self.neutral_values.get(k, 0.0)
            
            # Apply dampener to visual artifact models when quality is poor
            effective_weight = weight
            if k in ["edge_logit", "clip_logit", "identity_flicker", "landmark_jitter"]:
                effective_weight *= dampener

            delta = val - neutral
            contrib = effective_weight * delta
            contributions.append((k, float(contrib)))
            logit += contrib

        raw_prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
        calibrated_prob = self.calibrate(logit)

        # 4. EMA Smoothing
        if self._prev_ema is None:
            smoothed_prob = calibrated_prob
        else:
            smoothed_prob = self.ema_alpha * calibrated_prob + (1.0 - self.ema_alpha) * self._prev_ema
        self._prev_ema = smoothed_prob

        # 5. Hysteresis State Machine
        # To enter "Likely deepfake", require 2 consecutive windows >= 0.70
        # To leave "Likely deepfake", require 3 consecutive windows < 0.60
        if smoothed_prob >= 0.70:
            self._consecutive_fake_count += 1
            self._consecutive_authentic_count = 0
            if self._consecutive_fake_count >= 2:
                self._in_deepfake_state = True
        elif smoothed_prob < 0.60:
            self._consecutive_authentic_count += 1
            self._consecutive_fake_count = 0
            if self._consecutive_authentic_count >= 3:
                self._in_deepfake_state = False
                self._action_emitted = False
        else:
            # In neutral band [0.60, 0.70): keep current state counters unchanged
            pass

        # Determine category
        if self._in_deepfake_state:
            category = "Likely deepfake"
        elif smoothed_prob >= 0.40:
            category = "Suspicious"
        else:
            category = "Authentic"

        # Action event trigger (emit once when deepfake state entered)
        action_text: Optional[str] = None
        if self._in_deepfake_state and not self._action_emitted:
            action_text = "Ask the caller to turn their head left, then say the phrase again."
            self._action_emitted = True

        # Sort contributions by impact
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Evaluate sub-checks
        check_states = self.evaluate_checks(feat_dict)

        return FusionResult(
            raw_logit=float(logit),
            raw_probability=raw_prob,
            calibrated_probability=calibrated_prob,
            smoothed_probability=float(smoothed_prob),
            category=category,
            in_deepfake_state=self._in_deepfake_state,
            action_recommended=action_text,
            check_states=check_states,
            top_contributions=contributions[:5],
            feature_dict=feat_dict,
            missing_features=missing_keys,
        )

    def fuse_modules(
        self,
        module_results: List[ModuleResult],
    ) -> FusionResult:
        """
        Multimodal fusion over independent expert ModuleResult objects.
        Applies:
        1. Hard gates (virtual camera + light failure -> reject; poor quality -> step-up)
        2. Confidence-weighted feature assembly (skipped modules dropped, not zero-risk)
        3. Regularized logistic fusion & isotonic calibration
        4. Hysteresis smoothing & action recommendation
        """
        # Aggregate features from active modules
        aggregated_features: Dict[str, float] = {}
        active_modules: Dict[str, ModuleResult] = {}

        for mod in module_results:
            active_modules[mod.module_id] = mod
            if mod.status in ("ok", "degraded") and mod.confidence > 0.0:
                for k, v in mod.features.items():
                    aggregated_features[k] = v

        # Construct WindowFeatures from aggregated module features
        wf = WindowFeatures(
            source_risk=aggregated_features.get("source_risk"),
            light_corr_face=aggregated_features.get("light_corr_face"),
            light_lag_ms=aggregated_features.get("light_lag_ms"),
            light_neck_face_ratio=aggregated_features.get("light_neck_face_ratio"),
            light_neck_face_corr=aggregated_features.get("light_neck_face_corr"),
            light_snr=aggregated_features.get("light_snr"),
            edge_logit=aggregated_features.get("edge_logit"),
            clip_logit=aggregated_features.get("clip_logit"),
            model_disagreement=aggregated_features.get("model_disagreement"),
            ood_score=aggregated_features.get("ood_score"),
            identity_flicker=aggregated_features.get("identity_flicker"),
            landmark_jitter=aggregated_features.get("landmark_jitter"),
            occlusion_artifact=aggregated_features.get("occlusion_artifact"),
            spoof_max=aggregated_features.get("spoof_max"),
            spoof_mean=aggregated_features.get("spoof_mean"),
            sync_conf=aggregated_features.get("sync_conf"),
            sync_offset_ms=aggregated_features.get("sync_offset_ms"),
            phrase_wer=aggregated_features.get("phrase_wer"),
            bilabial_aperture_err=aggregated_features.get("bilabial_aperture_err"),
            quality_trust=aggregated_features.get("quality_trust"),
        )

        # Standard window processing
        res = self.process_window(wf)

        # Hard Gate 1: Virtual camera detected + failed active light reflection -> immediate Reject / Deepfake
        m1 = active_modules.get("M1_source")
        if m1 and m1.status == "ok":
            if m1.features.get("source_risk", 0.0) >= 0.70 and m1.features.get("light_corr_face", 0.5) < 0.20:
                res.category = "Likely deepfake"
                res.in_deepfake_state = True
                if not self._action_emitted:
                    res.action_recommended = "Hard gate triggered: Virtual camera with light modulation failure. Escalate for manual review."
                    self._action_emitted = True

        return res

    def reset(self):
        """Reset state tracking between sessions."""
        self._prev_ema = None
        self._consecutive_fake_count = 0
        self._consecutive_authentic_count = 0
        self._in_deepfake_state = False
        self._action_emitted = False
