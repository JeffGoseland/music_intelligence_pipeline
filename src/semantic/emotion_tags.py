"""
Semantic layer: map predicted arousal/valence to mood tags.
Scale ~1–9; tags are derived from configurable thresholds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# --- Configurable thresholds (change in one place) ---
AROUSAL_HIGH = 6.0
AROUSAL_LOW = 4.0
VALENCE_POSITIVE = 6.0
VALENCE_NEGATIVE = 4.0
VALENCE_CALM_FOCUS_MIN = 5.0  # valence >= this for Calm Focus

_ALL_TAGS = [
    "High Energy",
    "Low Energy",
    "Positive",
    "Negative",
    "Calm Focus",
    "Dramatic",
    "Neutral",
]


def _is_finite(x: float) -> bool:
    """Return False if x is NaN or inf."""
    return math.isfinite(x)


def get_tags_for_song(arousal: float, valence: float) -> list[str]:
    """
    Return list of tag names for a single (arousal, valence) point.
    Returns [] if either value is NaN or non-finite.
    """
    if not _is_finite(arousal) or not _is_finite(valence):
        return []
    tags: list[str] = []
    if arousal >= AROUSAL_HIGH:
        tags.append("High Energy")
    if arousal <= AROUSAL_LOW:
        tags.append("Low Energy")
    if valence >= VALENCE_POSITIVE:
        tags.append("Positive")
    if valence <= VALENCE_NEGATIVE:
        tags.append("Negative")
    if arousal <= AROUSAL_LOW and valence >= VALENCE_CALM_FOCUS_MIN:
        tags.append("Calm Focus")
    if arousal >= AROUSAL_HIGH and valence <= VALENCE_NEGATIVE:
        tags.append("Dramatic")
    if (
        AROUSAL_LOW < arousal < AROUSAL_HIGH
        and VALENCE_NEGATIVE < valence < VALENCE_POSITIVE
    ):
        tags.append("Neutral")
    return tags


def list_tags() -> list[str]:
    """Return all tag names in canonical order."""
    return list(_ALL_TAGS)


def filter_songs_by_tag(predictions_df: "pd.DataFrame", tag: str) -> "pd.DataFrame":
    """
    Return rows of predictions_df that have the given tag.
    predictions_df must have columns predicted_arousal, predicted_valence;
    song_id is optional. Rows with NaN arousal/valence are skipped (no tags).
    """
    import pandas as pd

    if tag not in _ALL_TAGS:
        return pd.DataFrame(columns=predictions_df.columns)
    rows = []
    for _, row in predictions_df.iterrows():
        a = row.get("predicted_arousal")
        v = row.get("predicted_valence")
        try:
            a_f, v_f = float(a), float(v)
        except (TypeError, ValueError):
            continue
        if not _is_finite(a_f) or not _is_finite(v_f):
            continue
        if tag in get_tags_for_song(a_f, v_f):
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=predictions_df.columns)
    return pd.DataFrame(rows).reset_index(drop=True)
