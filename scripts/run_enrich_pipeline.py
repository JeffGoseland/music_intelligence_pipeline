#!/usr/bin/env python3
"""
Run the enrich step: 10 DEAM features + tempo (BPM), genre, key from audio → song_features.csv.

Run from project root:
  python scripts/run_enrich_pipeline.py

Requires data/audio/ (MP3s) and data/deam_csvs/features/ (DEAM CSVs).
Output: data/processed/song_features.csv with columns:
  song_id, [10 DEAM cols], tempo_bpm, genre, key.
Genre is "unknown" (DEAM has no genre labels). Processing 1800+ files may take several minutes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.enrich_song_features import run_enrich_pipeline
from src.config.data_paths import SONG_FEATURES_PATH

if __name__ == "__main__":
    print("Building DEAM 10-feature table...", file=sys.stderr)
    df = run_enrich_pipeline()
    print(f"Wrote {len(df)} rows to {SONG_FEATURES_PATH}", file=sys.stderr)
    print(df.head())
