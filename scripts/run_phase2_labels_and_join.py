#!/usr/bin/env python3
"""Run Phase 2 steps 1–2: DEAM labels → deam_labels.csv, then join → modeling_dataset.csv. Run from project root."""

import sys

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.pipeline.deam_labels_loader import run_deam_labels_pipeline
from src.pipeline.build_modeling_dataset import run_build_modeling_dataset

if __name__ == "__main__":
    print("Step 1: Building deam_labels.csv from DEAM annotations...", file=sys.stderr)
    run_deam_labels_pipeline()
    print(
        "Step 2: Joining song_features + deam_labels → modeling_dataset.csv...",
        file=sys.stderr,
    )
    run_build_modeling_dataset()
    print("Done. Run: python3 scripts/validate_song_features.py", file=sys.stderr)
