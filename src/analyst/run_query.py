"""
Run analyst: user query -> LLM or fallback -> semantic tags -> filter songs.
Returns ranked list + explanation.
"""

from __future__ import annotations

from pathlib import Path

from ..config.data_paths import EMOTION_PREDICTIONS_PATH
from ..semantic import filter_songs_by_tag
from .query_llm import query_to_intent


def run_analyst(
    user_query: str,
    predictions_path: Path | None = None,
    max_songs: int = 20,
) -> tuple[list[str], str]:
    """
    Interpret user_query (Grok or fallback), filter by tags, return (song_ids, explanation).
    song_ids ordered by number of matching tags (most first), then stable order.
    """
    import pandas as pd

    path = predictions_path or EMOTION_PREDICTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}. Run the pipeline first."
        )
    df = pd.read_csv(path)
    intent = query_to_intent(user_query)
    tags = intent["tags"]
    explanation = intent["explanation"]

    # Union of song_ids matching any tag; count matches for ranking
    seen: set[str | float] = set()
    match_count: dict[str | float, int] = {}
    for tag in tags:
        subset = filter_songs_by_tag(df, tag)
        for sid in subset["song_id"]:
            key = sid
            seen.add(key)
            match_count[key] = match_count.get(key, 0) + 1

    # Rank by match count (desc), then stable order
    song_ids = sorted(seen, key=lambda s: (-match_count.get(s, 0), str(s)))
    song_ids = [str(s) for s in song_ids[:max_songs]]
    return song_ids, explanation
