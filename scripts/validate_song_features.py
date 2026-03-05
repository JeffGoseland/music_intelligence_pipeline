#!/usr/bin/env python3
"""Validate song_features.csv (schema + 90% tempo/key coverage). Run from project root."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.validate_song_features import main

if __name__ == "__main__":
    sys.exit(main())
