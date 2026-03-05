"""Project data paths. All relative to project root."""

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
