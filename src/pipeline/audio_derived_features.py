"""
Extract tempo (BPM) and musical key from audio files for the enrich step.

Reads from data/audio/ (e.g. MP3); song_id = filename stem.
Returns DataFrame: song_id, tempo_bpm, key (e.g. "C major", "A minor").
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Krumhansl-Schmuckler key profiles (12 pitch classes C, C#, D, ..., B)
# Major and minor; order matches librosa chroma: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> str:
    """Estimate key from chroma vector (12,) using Krumhansl-Schmuckler profiles."""
    if chroma.size != 12 or not np.any(chroma):
        return "unknown"
    chroma = chroma.astype(float)
    chroma = chroma / (np.linalg.norm(chroma) + 1e-8)
    best_corr = -np.inf
    best_key = "unknown"
    for shift in range(12):
        major_rot = np.roll(MAJOR_PROFILE, shift)
        minor_rot = np.roll(MINOR_PROFILE, shift)
        c_maj = np.corrcoef(chroma, major_rot)[0, 1]
        c_min = np.corrcoef(chroma, minor_rot)[0, 1]
        if np.isfinite(c_maj) and c_maj > best_corr:
            best_corr = c_maj
            best_key = f"{PITCH_NAMES[shift]} major"
        if np.isfinite(c_min) and c_min > best_corr:
            best_corr = c_min
            best_key = f"{PITCH_NAMES[shift]} minor"
    return best_key


def extract_tempo_and_key(audio_path: Path, sr: int = 22050, duration: float = 30.0) -> dict:
    """
    Extract tempo (BPM) and key from one audio file.
    Returns dict with song_id, tempo_bpm, key. Uses first `duration` seconds for speed.
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

    # Key from chroma (CQT)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        chroma_mean = np.mean(chroma, axis=1)
        key = _estimate_key(chroma_mean)
    except Exception:
        key = "unknown"

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
            import sys
            print(f"  audio: {i + 1}/{len(files)}", file=sys.stderr)

    return pd.DataFrame(rows)
