"""Case audit persistence and storage for TrustLens.
Saves session findings, metrics, and evidence to ./cases/<session_id>/case.json
with configurable TTL cleanup.
"""
import os
import json
import time
import shutil
import logging
from typing import Dict, Any, Optional, List

from app.config import settings

logger = logging.getLogger("trustlens.store")


class CaseStore:
    """Manages disk persistence of KYC case audits and evidence."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.path.dirname(__file__), "..", "..", "cases")
        os.makedirs(self.base_dir, exist_ok=True)

    def get_case_dir(self, session_id: str) -> str:
        d = os.path.join(self.base_dir, session_id)
        os.makedirs(d, exist_ok=True)
        return d

    def save_case(
        self,
        session_id: str,
        consent_status: str,
        stubs_in_use: bool,
        risk_score: float,
        category: str,
        findings: List[Dict[str, Any]],
        summary: str,
        ai_generated_summary: bool,
        windows: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Persists complete case audit record to cases/<session_id>/case.json."""
        case_dir = self.get_case_dir(session_id)
        evidence_dir = os.path.join(case_dir, "evidence")
        evidence_files = []
        if os.path.exists(evidence_dir):
            evidence_files = sorted(os.listdir(evidence_dir))

        record = {
            "session_id": session_id,
            "created_at": int(time.time()),
            "consent_status": consent_status,
            "stubs_in_use": stubs_in_use,
            "risk_score": round(float(risk_score), 4),
            "category": category,
            "findings": findings,
            "summary": summary,
            "ai_generated_summary": ai_generated_summary,
            "window_count": len(windows) if windows else 0,
            "evidence_files": evidence_files,
            "windows": windows or [],
        }

        file_path = os.path.join(case_dir, "case.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            logger.info("Saved case record for session %s to %s", session_id, file_path)
            return file_path
        except Exception as e:
            logger.error("Failed to write case record for %s: %s", session_id, e)
            return ""

    def load_case(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads case.json for a session if it exists."""
        file_path = os.path.join(self.base_dir, session_id, "case.json")
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read case %s: %s", session_id, e)
            return None

    def cleanup_expired_cases(self, ttl_hours: Optional[int] = None) -> int:
        """Deletes case folders older than TTL."""
        ttl = ttl_hours if ttl_hours is not None else settings.CASE_TTL_HOURS
        max_age_sec = ttl * 3600
        now = time.time()
        removed_count = 0

        if not os.path.exists(self.base_dir):
            return 0

        for entry in os.listdir(self.base_dir):
            entry_path = os.path.join(self.base_dir, entry)
            if os.path.isdir(entry_path):
                try:
                    stat = os.stat(entry_path)
                    age_sec = now - stat.st_mtime
                    if age_sec > max_age_sec:
                        shutil.rmtree(entry_path, ignore_errors=True)
                        removed_count += 1
                        logger.info("Evicted expired case %s (age: %.1f hrs)", entry, age_sec / 3600.0)
                except Exception as e:
                    logger.warning("Error checking case folder %s: %s", entry_path, e)

        return removed_count


case_store = CaseStore()
