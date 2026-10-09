"""Explainability Engine for TrustLens.
Generates structured findings, manages finding deduplication and severity escalation,
and generates visual evidence overlays (heatmaps) for suspicious windows.
"""
import os
import cv2
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from app.schemas import FindingEvent, Severity, EvidenceEvent

logger = logging.getLogger("trustlens.explain")


class ExplainEngine:
    """Manages explainability artifacts, finding catalog, deduplication, and heatmaps."""

    def __init__(self, cases_dir: Optional[str] = None):
        self.cases_dir = cases_dir or os.path.join(os.path.dirname(__file__), "..", "..", "cases")
        # Track previously emitted findings per check_id: {check_id: (severity_rank, last_emitted_time)}
        self._emitted_findings: Dict[str, Tuple[int, float]] = {}
        self._severity_ranks: Dict[Severity, int] = {"low": 1, "medium": 2, "high": 3}

    def generate_findings(
        self,
        features: Dict[str, float],
        t_sec: float,
    ) -> List[FindingEvent]:
        """
        Evaluates features against catalog rules and returns deduplicated findings.
        Only emits if new or if severity escalated.
        """
        candidates: List[FindingEvent] = []

        # 1. Source / Hardware provenance
        src_risk = features.get("source_risk", 0.0)
        if src_risk >= 0.70:
            candidates.append(
                FindingEvent(
                    id="source",
                    severity="high",
                    title="Virtual Camera Stream Detected",
                    detail="Software capture device or synthetic frame timing signature identified.",
                    t=round(t_sec, 2),
                )
            )
        elif src_risk >= 0.35:
            candidates.append(
                FindingEvent(
                    id="source",
                    severity="medium",
                    title="Suspicious Video Source Timing",
                    detail="Frame delivery jitter deviates from physical webcam hardware behavior.",
                    t=round(t_sec, 2),
                )
            )

        # 2. Light challenge response
        light_corr = features.get("light_corr_face", 0.50)
        light_snr = features.get("light_snr", 3.0)
        if light_snr >= 2.0:
            if light_corr < 0.15:
                candidates.append(
                    FindingEvent(
                        id="light",
                        severity="high",
                        title="Failed Active Light Challenge",
                        detail="Skin luminance did not modulate in sync with the randomized challenge sequence.",
                        t=round(t_sec, 2),
                    )
                )
            elif light_corr < 0.30:
                candidates.append(
                    FindingEvent(
                        id="light",
                        severity="medium",
                        title="Weak Active Light Modulation",
                        detail="Low photometric correlation between challenge sequence and face reflection.",
                        t=round(t_sec, 2),
                    )
                )

        # 3. Facial artifacts & flicker
        edge_logit = features.get("edge_logit", 0.0)
        flicker = features.get("identity_flicker", 0.0)
        jitter = features.get("landmark_jitter", 0.0)

        if edge_logit >= 1.4 or flicker >= 0.60:
            candidates.append(
                FindingEvent(
                    id="face",
                    severity="high",
                    title="Deepfake Boundary Artifacts",
                    detail="High-frequency frequency-domain blending seams and temporal embedding flicker detected.",
                    t=round(t_sec, 2),
                )
            )
        elif edge_logit >= 0.80 or flicker >= 0.35 or jitter >= 0.08:
            candidates.append(
                FindingEvent(
                    id="face",
                    severity="medium",
                    title="Facial Instability Detected",
                    detail="Micro-jitter and spatial blending boundaries observed across consecutive frames.",
                    t=round(t_sec, 2),
                )
            )

        # 4. Voice spoofing
        spoof_max = features.get("spoof_max", 0.0)
        if spoof_max >= 0.65:
            candidates.append(
                FindingEvent(
                    id="voice",
                    severity="high",
                    title="Synthetic Audio Cloned Voice",
                    detail="Acoustic features display non-natural vocoder synthesis patterns.",
                    t=round(t_sec, 2),
                )
            )
        elif spoof_max >= 0.40:
            candidates.append(
                FindingEvent(
                    id="voice",
                    severity="medium",
                    title="Acoustic Spectral Anomaly",
                    detail="Voice harmonics indicate potential voice conversion or re-synthesis.",
                    t=round(t_sec, 2),
                )
            )

        # 5. Lip Closure & AV sync
        lip_err = features.get("bilabial_aperture_err", 0.0)
        sync_off = features.get("sync_offset_ms", 0.0)
        sync_conf = features.get("sync_conf", 2.0)

        if lip_err >= 0.35 or (sync_conf >= 1.5 and sync_off >= 200.0):
            candidates.append(
                FindingEvent(
                    id="lips",
                    severity="high",
                    title="Phoneme-Lip Incoherence",
                    detail="Failure to seal lips on bilabial consonants (B/P/M) with significant AV desync.",
                    t=round(t_sec, 2),
                )
            )
        elif lip_err >= 0.18 or (sync_conf >= 1.5 and sync_off >= 120.0):
            candidates.append(
                FindingEvent(
                    id="lips",
                    severity="medium",
                    title="Mouth Motion Desynchronization",
                    detail="Aperture anomalies during bilabial speech segments.",
                    t=round(t_sec, 2),
                )
            )

        # Deduplicate candidates against emission history
        emitted: List[FindingEvent] = []
        for cand in candidates:
            cand_rank = self._severity_ranks.get(cand.severity, 1)
            prev_info = self._emitted_findings.get(cand.id)

            should_emit = False
            if prev_info is None:
                should_emit = True
            else:
                prev_rank, prev_time = prev_info
                # Emit if severity escalated
                if cand_rank > prev_rank:
                    should_emit = True
                # Or re-emit if 15 seconds have passed
                elif (t_sec - prev_time) >= 15.0:
                    should_emit = True

            if should_emit:
                self._emitted_findings[cand.id] = (cand_rank, t_sec)
                emitted.append(cand)

        return emitted

    def generate_heatmap_evidence(
        self,
        session_id: str,
        window_idx: int,
        face_crops: List[np.ndarray],
        edge_logit: float,
    ) -> Optional[EvidenceEvent]:
        """
        Creates a spatial anomaly heatmap overlay on the most suspicious face crop.
        Saves JPEG artifact into ./cases/<session_id>/evidence/heatmap_<window_idx>.jpg.
        """
        if not face_crops or edge_logit < 0.40:
            return None

        try:
            target_crop = face_crops[len(face_crops) // 2].copy()  # mid-window crop
            h, w = target_crop.shape[:2]

            # High-pass / Laplacian gradient map as anomaly proxy
            gray = cv2.cvtColor(target_crop, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            mag = np.abs(lap)
            mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply colormap (COLORMAP_JET)
            heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Alpha blend: 65% image, 35% heatmap
            blended = cv2.addWeighted(target_crop, 0.65, heatmap_rgb, 0.35, 0)

            # Save to disk
            evidence_dir = os.path.join(self.cases_dir, session_id, "evidence")
            os.makedirs(evidence_dir, exist_ok=True)
            filename = f"heatmap_w{window_idx:03d}.jpg"
            out_path = os.path.join(evidence_dir, filename)

            # Convert to BGR for cv2 write
            cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            url = f"/cases/{session_id}/evidence/{filename}"
            return EvidenceEvent(
                kind="heatmap",
                url=url,
                data={
                    "window_idx": window_idx,
                    "edge_logit": round(edge_logit, 3),
                    "file": filename,
                },
            )
        except Exception as e:
            logger.warning("Failed to generate heatmap evidence: %s", e)
            return None

    def reset(self):
        """Reset deduplication state for a new session."""
        self._emitted_findings.clear()
