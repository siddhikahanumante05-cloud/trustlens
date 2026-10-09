"""E6a Phrase Verification signal module: Speech recognition matching and Word Error Rate (WER)."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from app.signals.thresholds import PHRASE_WER_HIGH


@dataclass
class WordTimestamp:
    word: str
    start_sec: float
    end_sec: float


@dataclass
class PhraseResult:
    phrase_wer: float                     # Word error rate (0.0 = perfect match, 1.0 = failure)
    matched: bool
    expected_phrase: str
    transcribed_text: str
    word_timestamps: List[WordTimestamp]  # Timestamps for each spoken word
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def compute_word_error_rate(reference: List[str], hypothesis: List[str]) -> float:
    """Compute standard Levenshtein Word Error Rate (WER = (S + D + I) / N)."""
    r_len = len(reference)
    h_len = len(hypothesis)
    if r_len == 0:
        return 0.0 if h_len == 0 else 1.0

    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i - 1].lower() == hypothesis[j - 1].lower():
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    wer = float(d[r_len][h_len] / r_len)
    return wer


def analyze_phrase_match(
    transcribed_words: List[Dict[str, Any]],
    expected_phrase: str,
) -> PhraseResult:
    """
    Evaluate speech-to-text transcription vs expected challenge phrase.
    Pure function, zero I/O.
    """
    ref_words = [w.strip() for w in expected_phrase.strip().split() if w.strip()]
    hyp_words = [w.get("word", "").strip() for w in transcribed_words if w.get("word")]
    transcribed_text = " ".join(hyp_words)

    wer = compute_word_error_rate(ref_words, hyp_words)
    matched = wer <= PHRASE_WER_HIGH

    word_ts_list = [
        WordTimestamp(
            word=w.get("word", ""),
            start_sec=float(w.get("start", 0.0)),
            end_sec=float(w.get("end", 0.0)),
        )
        for w in transcribed_words
    ]

    flags = []
    if wer > PHRASE_WER_HIGH:
        flags.append("PHRASE_MISMATCH")

    return PhraseResult(
        phrase_wer=wer,
        matched=matched,
        expected_phrase=expected_phrase,
        transcribed_text=transcribed_text,
        word_timestamps=word_ts_list,
        flags=flags,
        details={
            "expected": expected_phrase,
            "heard": transcribed_text,
            "wer": wer,
        },
    )


_whisper_model = None


def get_whisper_model():
    """Retrieve or lazily initialize cached Whisper tiny model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model("tiny")
        except Exception:
            return None
    return _whisper_model


def transcribe_audio_pcm(audio_pcm: Any) -> List[Dict[str, Any]]:
    """Transcribe 16kHz s16le PCM audio using cached Whisper tiny model."""
    import numpy as np
    if audio_pcm is None or len(audio_pcm) < 8000:
        return []
    if float(np.std(audio_pcm)) < 60.0:
        return []
    model = get_whisper_model()
    if model is None:
        return []
    try:
        audio_f32 = audio_pcm.astype(np.float32) / 32768.0
        res = model.transcribe(audio_f32, word_timestamps=True, fp16=False)
        words: List[Dict[str, Any]] = []
        for segment in res.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                })
        return words
    except Exception:
        return []
