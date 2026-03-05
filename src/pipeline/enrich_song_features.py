"""
Enrich step: merge 10 DEAM core features with tempo (BPM), genre, and key from audio.

Produces data/processed/song_features.csv with columns:
  song_id, spectral_centroid, energy, mfcc_mean, chroma_variance, spectral_rolloff50,
  zcr, spectral_flux, spectral_variance, spectral_entropy, spectral_harmonicity,
  tempo_bpm, genre, key.

Genre is set to "unknown" (DEAM does not provide genre labels).
"""

from pathlib import Path
import pandas as pd

from .deam_feature_loader import run_feature_pipeline, RICH_DEAM_COLUMNS
from .audio_derived_features import run_audio_derived_pipeline


def run_enrich_pipeline(
    features_dir: Path | None = None,
    audio_dir: Path | None = None,
    output_path: Path | None = None,
    audio_limit: int | None = None,
) -> pd.DataFrame:
    """
    Build rich song_features: 10 DEAM features + tempo_bpm (from audio) + genre + key.

    Creates output_path parent dir if needed. Songs without matching audio get NaN tempo_bpm
    and key "unknown". audio_limit: if set, only process this many audio files (for testing).
    """
    from ..config.data_paths import FEATURES_CSV_DIR, AUDIO_DIR, SONG_FEATURES_PATH

    features_dir = features_dir or FEATURES_CSV_DIR
    audio_dir = audio_dir or AUDIO_DIR
    output_path = output_path or SONG_FEATURES_PATH

    # 1) DEAM: 10 core features per song (no write yet)
    deam_df = run_feature_pipeline(
        features_dir=features_dir,
        output_path=output_path,  # ignored when write=False
        rich=True,
        write=False,
    )

    # 2) Audio-derived: tempo_bpm, key
    audio_df = run_audio_derived_pipeline(audio_dir=audio_dir, limit=audio_limit)
    audio_df["genre"] = "unknown"

    # 3) Merge on song_id (left join: keep all DEAM songs)
    # song_id in DEAM may be string (from stem); in audio we use stem. Align dtypes.
    deam_df["song_id"] = deam_df["song_id"].astype(str)
    audio_df["song_id"] = audio_df["song_id"].astype(str)
    merged = deam_df.merge(
        audio_df[["song_id", "tempo_bpm", "genre", "key"]],
        on="song_id",
        how="left",
    )
    # Fill missing audio-derived rows
    merged["tempo_bpm"] = merged["tempo_bpm"].fillna(float("nan"))
    merged["genre"] = merged["genre"].fillna("unknown")
    merged["key"] = merged["key"].fillna("unknown")

    # Column order: song_id, 10 DEAM, tempo_bpm, genre, key
    final_cols = ["song_id"] + RICH_DEAM_COLUMNS + ["tempo_bpm", "genre", "key"]
    merged = merged[final_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return merged
