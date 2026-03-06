# Semantic layer: emotion tags (flags)

Mood tags are derived from **predicted arousal** and **predicted valence** (scale ~1–9). Each song can have **multiple tags**. Rules are applied in order; overlap is intentional (e.g. a song can be both High Energy and Positive).

---

## Tag definitions

| Tag | Rule | Meaning |
|-----|------|---------|
| **High Energy** | arousal ≥ 6 | Energetic, intense |
| **Low Energy** | arousal ≤ 4 | Calm, subdued |
| **Positive** | valence ≥ 6 | Happy, uplifting |
| **Negative** | valence ≤ 4 | Sad, tense, dark |
| **Calm Focus** | arousal ≤ 4 **and** valence ≥ 5 | Calm + positive (focus, light) |
| **Dramatic** | arousal ≥ 6 **and** valence ≤ 4 | Intense + negative (tense, angry) |
| **Neutral** | 4 < arousal < 6 **and** 4 < valence < 6 | Middle of the space (optional catch-all) |

---

## Thresholds (configurable)

- **Arousal:** high = 6, low = 4 (mid ≈ 5).  
- **Valence:** positive = 6, negative = 4 (mid ≈ 5).  

These values are defined as constants in the semantic module (`src/semantic/emotion_tags.py`) so they can be changed in one place.

---

## Usage

- **Tag a single point:** `get_tags_for_song(arousal, valence)` → list of tag strings.  
- **Filter catalog:** `filter_songs_by_tag(predictions_df, tag)` → rows that have that tag.  
- **List all tags:** `list_tags()` → `["High Energy", "Low Energy", "Positive", "Negative", "Calm Focus", "Dramatic", "Neutral"]`.  

The semantic layer does **not** perform natural-language understanding; that is the responsibility of the agent (Phase 4), which maps user queries to these tags or to similarity in (arousal, valence) space.

---

## Implementation

- **Module location:** `src/semantic/emotion_tags.py` (or package `src.semantic`).
- **Use from code:** e.g. `from src.semantic import filter_songs_by_tag; filtered = filter_songs_by_tag(predictions_df, "Calm Focus")`. The DataFrame must have columns `predicted_arousal` and `predicted_valence` (e.g. from `emotion_predictions.csv`).
- **Example script:** `python3 scripts/run_semantic_example.py` prints tag counts and sample song_ids for "Calm Focus".
