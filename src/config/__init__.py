# Package initializer: makes "from src.config import SONG_FEATURES_PATH" work (Python convention).
from .data_paths import (
    DATA_DIR,
    AUDIO_DIR,
    FEATURES_CSV_DIR,
    PROCESSED_DIR,
    SONG_FEATURES_PATH,
)

__all__ = [
    "DATA_DIR",
    "AUDIO_DIR",
    "FEATURES_CSV_DIR",
    "PROCESSED_DIR",
    "SONG_FEATURES_PATH",
]
