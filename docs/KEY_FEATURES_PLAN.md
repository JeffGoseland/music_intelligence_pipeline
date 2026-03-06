# Plan: Key-derived columns (note, mode, accidental)

**✅ Implemented** (key_parsing.py, enrich_song_features, validate_song_features, Shiny Colour by).

Add **key_note**, **key_mode**, **key_signature**, and **is_major** derived from the existing `key` column (e.g. `"C major"`, `"A minor"`, `"F# major"`). Single source of truth remains `key`; new columns are parsed once and propagated everywhere.

---

## 1. Schema (new columns)

| Column          | Type    | Description |
|-----------------|---------|-------------|
| **key_note**    | string  | Letter name: A, B, C, D, E, F, G (no accidental). |
| **key_mode**    | string  | `"major"` or `"minor"`. |
| **key_signature** | string | `"sharp"`, `"flat"`, or `"natural"` (accidental). |
| **is_major**    | bool    | `True` if key_mode == "major", else `False`. Convenience for filters/Shiny. |

**Note:** Current pipeline emits only sharps (e.g. `F# major`). Flats are supported in parsing if we ever add `Bb`/`Eb` style output; otherwise key_signature will be `"natural"` or `"sharp"` only.

---

## 2. Where to implement

- **Parsing:** One small helper that takes `key` (e.g. `"F# minor"`) → `(key_note, key_mode, key_signature)`. Place in `src/pipeline/audio_derived_features.py` (next to key estimation) or in a new `src/pipeline/key_parsing.py`. Calling it from enrich keeps key estimation unchanged.
- **Enrich:** In `enrich_song_features.py`, after merging DEAM + audio (so we have `key`), run the parser on each `key` and add columns **key_note**, **key_mode**, **key_signature**, **is_major** to the merged DataFrame before writing `song_features.csv`. Append these to `final_cols`.
- **Validation:** In `validate_song_features.py`, add the four columns to `REQUIRED_COLUMNS` (or an optional “key-derived” set) and document in DATA_DICTIONARY. Optional: allow key_note in A–G, key_mode in [major, minor], key_signature in [sharp, flat, natural].
- **Data dictionary:** Document the four columns in `docs/DATA_DICTIONARY.md` under song_features.csv.
- **Shiny:** In `_load()`, include the new columns in the merge from song_features (already merging on song_id). Add **Colour by** options: Key note, Key mode (major/minor), Key signature. Add key_note, key_mode, key_signature, is_major to the song table display columns.

---

## 3. Propagation (no change to training)

- **song_features.csv** — gains key_note, key_mode, key_signature, is_major.
- **modeling_dataset.csv** — built by joining song_features + labels; will automatically get the new columns (they are not used as model features unless we add them later).
- **emotion_predictions.csv** — unchanged (song_id, predicted_arousal, predicted_valence). Shiny already joins with song_features for key/tempo/genre; after this it will also get the new key columns from that join.
- **Training** — no change; no new model inputs.

---

## 4. Parsing rules (from current `key` format)

- Split on space: `root_part`, `mode_part`. Mode: `"minor"` if `"minor"` in mode_part else `"major"`.
- Root: if last character of root_part is `#` → key_signature = `"sharp"`, key_note = root_part[:-1]. If last char is `b` (flat) → key_signature = `"flat"`, key_note = root_part[:-1]. Else → key_signature = `"natural"`, key_note = root_part.
- Normalize key_note to one of A–G (strip any remaining accidental). Handle `"unknown"` → key_note = "Unknown", key_mode = "major", key_signature = "natural", is_major = True (or leave as missing if you prefer).

---

## 5. Opinion

- **One parser, one place:** Derive everything from the existing `key` string in the enrich step. No change to key estimation; just parse its output. That keeps one source of truth and avoids duplication.
- **key_mode over is_major/is_minor:** Prefer a single **key_mode** (`"major"` / `"minor"`) plus an **is_major** boolean for convenience in Shiny and filters. Two booleans (is_major, is_minor) are redundant.
- **key_note as letter only:** Keep **key_note** as the note letter (A–G). Accidental is in **key_signature**. That gives clean “Colour by note” (7 buckets) and “Colour by sharp/flat/natural” without mixing.
- **Shiny:** Adding “Key note”, “Key mode”, and “Key signature” to **Colour by** and to the table makes the new fields immediately useful. No need to change how data is loaded beyond including the new columns in the merge.

Implement in this order: (1) parsing helper + enrich, (2) DATA_DICTIONARY + validation, (3) Shiny merge + Colour by + table columns.
