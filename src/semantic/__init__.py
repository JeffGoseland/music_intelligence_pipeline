"""
Semantic layer: tag songs by arousal/valence and filter by tag.
"""

from .emotion_tags import (
    AROUSAL_HIGH,
    AROUSAL_LOW,
    VALENCE_CALM_FOCUS_MIN,
    VALENCE_NEGATIVE,
    VALENCE_POSITIVE,
    filter_songs_by_tag,
    get_tags_for_song,
    list_tags,
)

__all__ = [
    "AROUSAL_HIGH",
    "AROUSAL_LOW",
    "VALENCE_POSITIVE",
    "VALENCE_NEGATIVE",
    "VALENCE_CALM_FOCUS_MIN",
    "list_tags",
    "get_tags_for_song",
    "filter_songs_by_tag",
]
