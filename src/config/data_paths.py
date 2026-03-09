"""Project data paths. All relative to project root."""

import re
from pathlib import Path

# Project root (directory containing src/, data/, docs/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data layout (git-ignored content lives here)
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
FEATURES_CSV_DIR = DATA_DIR / "deam_csvs" / "features"
ANNOTATIONS_DIR = DATA_DIR / "deam_csvs" / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Pipeline outputs
SONG_FEATURES_PATH = PROCESSED_DIR / "song_features.csv"
DEAM_LABELS_PATH = PROCESSED_DIR / "deam_labels.csv"
MODELING_DATASET_PATH = PROCESSED_DIR / "modeling_dataset.csv"
EMOTION_PREDICTIONS_PATH = PROCESSED_DIR / "emotion_predictions.csv"
PIPELINE_RUN_PATH = PROCESSED_DIR / "pipeline_run.json"
PIPELINE_CHECKPOINT_DIR = PROCESSED_DIR / ".checkpoints"

# Versioned model runs: models/<run_id>/ (run_id = YYYYMMDD_HHMMSS)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def get_latest_models_dir() -> Path:
    """
    Return the latest versioned models directory (models/<run_id>/).
    If no run_id subdirs exist, return MODELS_DIR for backward compatibility.
    """
    if not MODELS_DIR.exists():
        return MODELS_DIR
    subdirs = [
        d.name
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and RUN_ID_PATTERN.match(d.name)
    ]
    if not subdirs:
        return MODELS_DIR
    # run_id is YYYYMMDD_HHMMSS so lexicographic max = latest run
    return MODELS_DIR / max(subdirs)
