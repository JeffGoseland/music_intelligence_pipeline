#!/usr/bin/env python3
"""Validate song_features.csv (schema + 90% tempo/key coverage). Run from project root."""

import sys

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.pipeline.validate_song_features import main

if __name__ == "__main__":
    sys.exit(main())
