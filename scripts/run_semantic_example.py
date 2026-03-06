#!/usr/bin/env python3
"""Example: Phase 3 semantic layer — list tag counts and filter songs. Run from project root."""

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

import pandas as pd

from src.config.data_paths import EMOTION_PREDICTIONS_PATH
from src.semantic import filter_songs_by_tag, list_tags


def main() -> int:
    df = pd.read_csv(EMOTION_PREDICTIONS_PATH)
    print("Tag counts (songs with at least that tag):")
    for tag in list_tags():
        subset = filter_songs_by_tag(df, tag)
        print(f"  {tag}: {len(subset)}")
    print("\nExample: first 5 'Calm Focus' song_ids:")
    calm = filter_songs_by_tag(df, "Calm Focus")
    if len(calm) > 0:
        print(calm["song_id"].head().tolist())
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
