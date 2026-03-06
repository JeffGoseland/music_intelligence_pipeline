#!/usr/bin/env python3
"""Run Phase 1: DEAM CSVs → song_features.csv (minimal schema). Run from project root."""

import sys

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.config.data_paths import SONG_FEATURES_PATH
from src.pipeline.deam_feature_loader import run_feature_pipeline

if __name__ == "__main__":
    df = run_feature_pipeline()
    print(f"Wrote {len(df)} rows to {SONG_FEATURES_PATH}", file=sys.stderr)
    print(df.head())
