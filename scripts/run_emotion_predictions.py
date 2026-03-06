#!/usr/bin/env python3
"""Generate emotion_predictions.csv from song_features using trained XGBoost models. Run from project root."""

import sys

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.pipeline.generate_emotion_predictions import run_emotion_predictions

if __name__ == "__main__":
    run_emotion_predictions()
    print(
        "Wrote data/processed/emotion_predictions.csv. Run: python3 scripts/validate_song_features.py",
        file=sys.stderr,
    )
