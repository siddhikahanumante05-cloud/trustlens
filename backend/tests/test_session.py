"""Unit tests for SessionService and challenge generation."""
import time
from app.session import SessionService, B_WORDS, P_WORDS, M_WORDS


def test_phrase_words_start_with_bpm():
    """Verify generated 3-word phrase consists of words starting with B, P, or M."""
    service = SessionService(secret_key="test-secret")
    for _ in range(50):
        phrase = service.generate_phrase()
        words = phrase.split()
        assert len(words) == 3, f"Expected 3 words, got: {phrase}"
        
        # Check first letters
        first_letters = {w[0].upper() for w in words}
        assert first_letters == {"B", "P", "M"}, f"Expected B, P, M, got: {first_letters}"
        
        # Check each word belongs to the respective word sets
        for w in words:
            assert (w in B_WORDS) or (w in P_WORDS) or (w in M_WORDS)


def test_light_sequence_starts_with_white_sync_pulse():
    """Verify light sequence always starts with #ffffff sync pulse."""
    service = SessionService(secret_key="test-secret")
    for nonce in ["nonce_a", "nonce_b", "1234567890abcdef"]:
        seq = service.generate_light_sequence(nonce=nonce)
        assert len(seq) >= 6 and len(seq) <= 8
        assert seq[0].color == "#ffffff"


def test_light_sequence_rate_limits():
    """Verify <= 3 color changes per second (each step duration >= 333 ms)."""
    service = SessionService(secret_key="test-secret")
    seq = service.generate_light_sequence(nonce="test_nonce")
    for step in seq:
        assert step.ms >= 333, f"Step duration {step.ms} ms allows > 3 changes/sec"


def test_light_sequence_determinism():
    """Verify sequence is strictly deterministic given the same secret and nonce."""
    service_1 = SessionService(secret_key="consistent-secret")
    service_2 = SessionService(secret_key="consistent-secret")
    
    seq_1 = service_1.generate_light_sequence(nonce="exact_same_nonce")
    seq_2 = service_2.generate_light_sequence(nonce="exact_same_nonce")
    
    assert [s.model_dump() for s in seq_1] == [s.model_dump() for s in seq_2]


def test_light_sequence_differs_for_different_nonces():
    """Verify different nonces produce different color sequences."""
    service = SessionService(secret_key="consistent-secret")
    seq_1 = service.generate_light_sequence(nonce="nonce_one")
    seq_2 = service.generate_light_sequence(nonce="nonce_two")
    
    colors_1 = [s.color for s in seq_1]
    colors_2 = [s.color for s in seq_2]
    assert colors_1 != colors_2


def test_session_lifecycle_and_expiry():
    """Verify session creation, retrieval, and expiration."""
    service = SessionService(secret_key="test-secret")
    created = service.create_session(ttl_seconds=300)
    
    assert created.session_id is not None
    assert created.nonce is not None
    assert created.expires_at >= int(time.time()) + 295
    
    # Retrieve active session
    rec = service.get_session(created.session_id)
    assert rec is not None
    assert rec["session_id"] == created.session_id
    
    # Invalidate session
    service.invalidate(created.session_id)
    assert service.get_session(created.session_id) is None
