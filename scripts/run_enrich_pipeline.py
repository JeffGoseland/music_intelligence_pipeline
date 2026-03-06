#!/usr/bin/env python3
"""Run enrich: 10 DEAM features + tempo, genre, key → song_features.csv. Run from project root."""

import sys

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.config.data_paths import SONG_FEATURES_PATH
from src.pipeline.enrich_song_features import run_enrich_pipeline

if __name__ == "__main__":
    print("Building DEAM 10-feature table...", file=sys.stderr)
    df = run_enrich_pipeline()
    print(f"Wrote {len(df)} rows to {SONG_FEATURES_PATH}", file=sys.stderr)
    print(df.head())
