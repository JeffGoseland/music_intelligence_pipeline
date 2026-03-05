#!/usr/bin/env python3
"""
Run Phase 1 feature pipeline: DEAM CSVs → song_features.csv.

Run from project root:
  python scripts/run_feature_pipeline.py
  or: python -m src.pipeline.deam_feature_loader
"""
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.deam_feature_loader import run_feature_pipeline
from src.config.data_paths import SONG_FEATURES_PATH

if __name__ == "__main__":
    df = run_feature_pipeline()
    print(f"Wrote {len(df)} rows to {SONG_FEATURES_PATH}", file=sys.stderr)
    print(df.head())
