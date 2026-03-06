"""
Phase 2 Step 2: Join song_features.csv with deam_labels.csv on song_id → modeling_dataset.csv.

Inner join so only songs with both features and labels are included.
Output: data/processed/modeling_dataset.csv (all song_features columns + arousal, valence).
"""

from pathlib import Path
import pandas as pd


def run_build_modeling_dataset(
    song_features_path: Path | None = None,
    deam_labels_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load song_features and deam_labels, inner-join on song_id, write modeling_dataset.csv.

    Creates output_path parent dir if needed.
    """
    from ..config.data_paths import (
        SONG_FEATURES_PATH,
        DEAM_LABELS_PATH,
        MODELING_DATASET_PATH,
    )

    song_features_path = song_features_path or SONG_FEATURES_PATH
    deam_labels_path = deam_labels_path or DEAM_LABELS_PATH
    output_path = output_path or MODELING_DATASET_PATH

    features = pd.read_csv(song_features_path)
    labels = pd.read_csv(deam_labels_path)
    features["song_id"] = features["song_id"].astype(str)
    labels["song_id"] = labels["song_id"].astype(str)
    merged = features.merge(
        labels[["song_id", "arousal", "valence"]], on="song_id", how="inner"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
