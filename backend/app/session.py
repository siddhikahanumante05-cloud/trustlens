"""Session service for challenge generation (nonce, phrase, light sequence)."""
import time
import secrets
import hmac
import hashlib
from typing import List, Dict, Optional, Tuple
from app.config import settings
from app.schemas import LightStep, SessionCreateResponse

# Words chosen where each word begins with b/p/m phonemes (bilabial sounds)
B_WORDS = ["Banana", "Bottle", "Blanket", "Bridge", "Button", "Basket"]
P_WORDS = ["Paper", "Pencil", "Pocket", "Pillow", "Purple", "Planet"]
M_WORDS = ["Mountain", "Mango", "Marble", "Mirror", "Monkey", "Morning"]

# Soft saturated palette for light reflection test
# High contrast against dark rooms, safe soft hues
SOFT_COLORS = [
    "#ff4d4d",  # Soft Red
    "#4d79ff",  # Soft Blue
    "#4dff88",  # Soft Green
    "#ffb84d",  # Soft Amber
    "#b84dff",  # Soft Purple
    "#4dffff",  # Soft Cyan
    "#ff4da6",  # Soft Magenta
    "#ffe04d",  # Soft Yellow
]


class SessionService:
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or settings.SECRET_KEY
        self._sessions: Dict[str, Dict] = {}

    def generate_phrase(self, rng: Optional[secrets.SystemRandom] = None) -> str:
        """Generate a 3-word challenge phrase where each word starts with B, P, or M."""
        r = rng or secrets.SystemRandom()
        b_word = r.choice(B_WORDS)
        p_word = r.choice(P_WORDS)
        m_word = r.choice(M_WORDS)
        words = [b_word, p_word, m_word]
        r.shuffle(words)
        return " ".join(words)

    def generate_light_sequence(self, nonce: str, num_steps: int = 7, step_duration_ms: int = 350) -> List[LightStep]:
        """
        Generate deterministic light sequence from HMAC(secret, nonce).
        Rules:
        - 6 to 8 steps
        - <= 3 color changes per second (step_duration_ms >= 333 ms)
        - ALWAYS begins with a white sync pulse ("#ffffff")
        - Soft saturated colors
        """
        assert 6 <= num_steps <= 8, "num_steps must be between 6 and 8"
        assert step_duration_ms >= 333, "color changes must be <= 3 per second (ms >= 333)"

        # HMAC-SHA256 from secret and nonce
        h = hmac.new(self.secret_key.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256).digest()

        steps: List[LightStep] = []
        # 1. First step is always the white sync pulse
        steps.append(LightStep(color="#ffffff", ms=step_duration_ms))

        # 2. Subsequent steps chosen deterministically from HMAC bytes
        for i in range(1, num_steps):
            byte_val = h[i % len(h)]
            color = SOFT_COLORS[byte_val % len(SOFT_COLORS)]
            # Prevent adjacent identical colors
            if color == steps[-1].color:
                color = SOFT_COLORS[(byte_val + 1) % len(SOFT_COLORS)]
            steps.append(LightStep(color=color, ms=step_duration_ms))

        return steps

    def create_session(self, ttl_seconds: int = 300) -> SessionCreateResponse:
        """Create a new challenge session with nonce, b/p/m phrase, and light sequence."""
        session_id = secrets.token_urlsafe(16)
        nonce = secrets.token_hex(16)
        phrase = self.generate_phrase()
        light_sequence = self.generate_light_sequence(nonce=nonce)
        expires_at = int(time.time()) + ttl_seconds

        record = {
            "session_id": session_id,
            "nonce": nonce,
            "phrase": phrase,
            "light_sequence": light_sequence,
            "expires_at": expires_at,
            "used": False,
        }
        self._sessions[session_id] = record

        return SessionCreateResponse(
            session_id=session_id,
            nonce=nonce,
            phrase=phrase,
            light_sequence=light_sequence,
            expires_at=expires_at,
        )

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve an active session if not expired."""
        rec = self._sessions.get(session_id)
        if not rec:
            return None
        if time.time() > rec["expires_at"]:
            self._sessions.pop(session_id, None)
            return None
        return rec

    def invalidate(self, session_id: str) -> None:
        """Invalidate a session after use."""
        self._sessions.pop(session_id, None)


# Global singleton instance
session_service = SessionService()
