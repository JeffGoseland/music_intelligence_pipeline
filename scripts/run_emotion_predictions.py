#!/usr/bin/env python3
"""Generate emotion_predictions.csv from song_features using trained XGBoost models. Run from project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.generate_emotion_predictions import run_emotion_predictions

if __name__ == "__main__":
    run_emotion_predictions()
    print("Wrote data/processed/emotion_predictions.csv. Run: python3 scripts/validate_song_features.py", file=sys.stderr)
