"""
Extract tempo (BPM) and musical key from audio files for the enrich step.

Reads from data/audio/ (e.g. MP3); song_id = filename stem.
Returns DataFrame: song_id, tempo_bpm, key (e.g. "C major", "A minor").
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Krumhansl-Schmuckler key profiles (12 pitch classes C, C#, D, ..., B)
# Order matches librosa chroma: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
_MAJOR_RAW = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_MINOR_RAW = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)
# Normalize to unit norm so cosine similarity is in [0, 1]
MAJOR_PROFILE = _MAJOR_RAW / (np.linalg.norm(_MAJOR_RAW) + 1e-12)
MINOR_PROFILE = _MINOR_RAW / (np.linalg.norm(_MINOR_RAW) + 1e-12)
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> str:
    """
    Estimate key from chroma vector (12,) using Krumhansl-Schmuckler profiles.
    Uses cosine similarity (normalized chroma vs rotated key profile).
    Fallback: if no profile wins, use pitch class with max energy as root (major).
    """
    if chroma is None or chroma.size != 12:
        return "unknown"
    chroma = np.asarray(chroma, dtype=float).ravel()[:12]
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(chroma > 0):
        return "unknown"
    norm = np.linalg.norm(chroma)
    if norm < 1e-12:
        norm = np.max(chroma) + 1e-12
    chroma = chroma / norm
    best_score = -np.inf
    best_key = "unknown"
    for shift in range(12):
        major_rot = np.roll(MAJOR_PROFILE, shift)
        minor_rot = np.roll(MINOR_PROFILE, shift)
        score_maj = float(np.dot(chroma, major_rot))
        score_min = float(np.dot(chroma, minor_rot))
        if np.isfinite(score_maj) and score_maj > best_score:
            best_score = score_maj
            best_key = f"{PITCH_NAMES[shift]} major"
        if np.isfinite(score_min) and score_min > best_score:
            best_score = score_min
            best_key = f"{PITCH_NAMES[shift]} minor"
    # Fallback: pick root as pitch class with highest chroma
    if best_key == "unknown":
        root_idx = int(np.argmax(chroma))
        best_key = f"{PITCH_NAMES[root_idx]} major"
    return best_key


def _chroma_vector_for_key(y: np.ndarray, sr: int) -> np.ndarray | None:
    """
    Get a single 12-D chroma vector for key estimation.
    Sum over time. Tries chroma_stft, then chroma_cqt, then chroma_cens.
    Returns None only if all methods fail or return empty/zero chroma.
    """
    import librosa

    hop = 2048
    n_chroma = 12

    def make_vector(chroma: np.ndarray) -> np.ndarray | None:
        if chroma is None or chroma.size == 0:
            return None
        vec = np.sum(chroma, axis=1)
        if vec.size != n_chroma or not np.any(vec > 0):
            return None
        return vec.astype(float)

    # 1) chroma_stft (most portable)
    try:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=hop, n_chroma=n_chroma
        )
        out = make_vector(chroma)
        if out is not None:
            return out
    except Exception:
        pass

    # 2) chroma_cqt (often better for key, can fail on some systems)
    try:
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop, n_chroma=n_chroma
        )
        out = make_vector(chroma)
        if out is not None:
            return out
    except Exception:
        pass

    # 3) chroma_cens (smoother, different backend)
    try:
        chroma = librosa.feature.chroma_cens(
            y=y, sr=sr, hop_length=hop, n_chroma=n_chroma
        )
        out = make_vector(chroma)
        if out is not None:
            return out
    except Exception:
        pass

    return None


def extract_tempo_and_key(
    audio_path: Path, sr: int = 22050, duration: float = 45.0
) -> dict:
    """
    Extract tempo (BPM) and key from one audio file.
    Returns dict with song_id, tempo_bpm, key. Uses first `duration` seconds for speed.
    Key uses chroma_stft (fallback chroma_cqt) and Krumhansl–Schmuckler profiles.
    """
    import librosa

    song_id = audio_path.stem
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
    except Exception:
        return {"song_id": song_id, "tempo_bpm": float("nan"), "key": "unknown"}

    if len(y) == 0:
        return {"song_id": song_id, "tempo_bpm": float("nan"), "key": "unknown"}

    # Tempo (BPM)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        tempo_bpm = float("nan")

    # Key from chroma: use middle segment to avoid silent intros/outros
    n = len(y)
    if n >= sr * 10:
        mid_start = n // 4
        mid_end = (3 * n) // 4
        y_key = y[mid_start:mid_end]
    else:
        y_key = y
    chroma_vec = _chroma_vector_for_key(y_key, sr)
    key = _estimate_key(chroma_vec) if chroma_vec is not None else "unknown"

    return {"song_id": song_id, "tempo_bpm": tempo_bpm, "key": key}


def run_audio_derived_pipeline(
    audio_dir: Path | None = None,
    extensions: tuple[str, ...] = (".mp3", ".wav", ".flac"),
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Scan audio_dir for audio files, extract tempo and key for each.
    Returns DataFrame with columns song_id, tempo_bpm, key.
    limit: if set, process only this many files (for testing).
    """
    from ..config.data_paths import AUDIO_DIR

    audio_dir = audio_dir or AUDIO_DIR
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(f"*{ext}"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No audio files in {audio_dir}")
    if limit is not None:
        files = files[:limit]

    rows = []
    for i, path in enumerate(files):
        row = extract_tempo_and_key(path)
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  audio: {i + 1}/{len(files)}", file=sys.stderr)

    return pd.DataFrame(rows)
