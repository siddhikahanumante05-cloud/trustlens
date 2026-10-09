"""Narrator Engine for TrustLens.
Generates concise, factual executive summaries using deterministic templates
with optional local Ollama (qwen2.5:0.5b) enhancement and strict validation.
"""
import logging
import urllib.request
import json
from typing import List, Tuple, Optional

from app.config import settings
from app.schemas import FindingEvent

logger = logging.getLogger("trustlens.narrator")


class NarratorEngine:
    """Produces verified forensic case summaries."""

    def __init__(self):
        self.ollama_url = settings.OLLAMA_URL.rstrip("/")
        self.model_name = settings.OLLAMA_MODEL
        self.mode = settings.NARRATOR_MODE.lower()  # auto | template | llm
        self.timeout = settings.LLM_TIMEOUT_SEC

    def check_ollama_available(self) -> bool:
        """Check if local Ollama daemon is reachable and has the target model."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "").lower() for m in data.get("models", [])]
                    # Check exact or partial match e.g. "qwen2.5:0.5b" in "qwen2.5:0.5b:latest"
                    target = self.model_name.lower().split(":")[0]
                    return any(target in m for m in models)
        except Exception:
            return False
        return False

    def build_template_summary(
        self,
        risk_score: float,
        category: str,
        findings: List[FindingEvent],
    ) -> str:
        """Constructs a deterministic, factual 3-sentence summary."""
        pct = int(round(risk_score * 100))

        # Sentence 1: Risk score & category
        s1 = f"Verification session completed with an assessed deepfake risk of {pct}% ({category})."

        # Sentence 2: Findings breakdown
        if findings:
            high_findings = [f.title for f in findings if f.severity == "high"]
            med_findings = [f.title for f in findings if f.severity == "medium"]
            key_signals = high_findings[:2] or med_findings[:2]
            signals_str = " and ".join(f"'{s}'" for s in key_signals)
            s2 = f"Primary anomalies observed include {signals_str} across active analysis windows."
        else:
            s2 = "All evaluated biometric and physical challenge signals remained within expected authentic baselines."

        # Sentence 3: Operational recommendation
        if category == "Likely deepfake":
            s3 = "Recommendation: Escalate for manual analyst review or issue a secondary physical head-turn challenge."
        elif category == "Suspicious":
            s3 = "Recommendation: Request additional caller verbal verification before proceeding with authentication."
        else:
            s3 = "Recommendation: Biometric and hardware checks confirm legitimate caller presence."

        return f"{s1} {s2} {s3}"

    def rewrite_with_llm(self, template_text: str, risk_score: float) -> Optional[str]:
        """Calls local Ollama to polish the summary with strict hallucination checks."""
        prompt = (
            "You are a forensic KYC compliance assistant. "
            "Rewrite the following forensic assessment into a concise, professional 2-3 sentence executive summary. "
            "Strict constraints: Do NOT invent new facts or signals. You MUST preserve the exact percentage or risk number. "
            f"Forensic assessment:\n\"{template_text}\"\n\nExecutive summary:"
        )

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 120,
            }
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    llm_text = res_data.get("response", "").strip()

                    # Strict validation:
                    # 1. Length check
                    if not (20 <= len(llm_text) <= 500):
                        logger.warning("LLM summary length invalid (%d chars). Discarding.", len(llm_text))
                        return None

                    # 2. Score preservation check: ensure risk percentage appears
                    pct_str = f"{int(round(risk_score * 100))}%"
                    score_str = f"{risk_score:.2f}"
                    if pct_str not in llm_text and score_str not in llm_text:
                        logger.warning("LLM response failed numeric validation (missing %s). Discarding.", pct_str)
                        return None

                    return llm_text
        except Exception as e:
            logger.info("Ollama rewrite unavailable or timed out: %s. Using template fallback.", e)
            return None

        return None

    def generate_summary(
        self,
        risk_score: float,
        category: str,
        findings: List[FindingEvent],
    ) -> Tuple[str, bool]:
        """
        Generates case summary. Returns (text, ai_generated).
        """
        template_text = self.build_template_summary(risk_score, category, findings)

        if self.mode == "template":
            return template_text, False

        # If mode is 'auto' or 'llm', attempt Ollama if available
        if self.mode in ["auto", "llm"]:
            ollama_ready = self.check_ollama_available()
            if ollama_ready:
                llm_text = self.rewrite_with_llm(template_text, risk_score)
                if llm_text:
                    return llm_text, True

            # If mode was strictly 'llm' but it failed, still return template to avoid breaking callers
            if self.mode == "llm":
                logger.info("Narrator mode 'llm' requested but Ollama rewrite failed; falling back to template.")

        return template_text, False
