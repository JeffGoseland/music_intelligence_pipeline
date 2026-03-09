# Design and coding decisions

Short notes on **why** certain choices were made. Use this when changing behaviour or debugging.

---

## Data and IDs

- **`song_id` as string everywhere**  
  CSVs are read with `dtype={"song_id": str}` (e.g. in `validate_song_features`, `build_modeling_dataset`) so that:
  - Joins and merges don’t fail due to int vs string (e.g. `2` vs `"2"`).
  - Validation doesn’t report “song_id is not string/object” when pandas infers numeric IDs.  
  DEAM and audio stems can be numeric; we normalize to string at read time.

- **Key-derived columns (key_note, key_mode, key_signature, is_major)**  
  Added so the emotion map and filters can use key without re-parsing. Validation requires them so downstream code can assume they exist. See `src/pipeline/key_parsing.py` and `docs/KEY_FEATURES_PLAN.md`.

---

## Pipeline and checkpoints

- **Resume via `.done` files**  
  `run_full_pipeline.py` skips a step if `data/processed/.checkpoints/<step>.done` exists. Use `--force` to run all steps anyway. Checkpoints are step-level only (no per-file granularity).

- **Versioned model dirs: `models/<run_id>/`**  
  Each training run writes to `models/YYYYMMDD_HHMMSS/` so we don’t overwrite previous runs. `get_latest_models_dir()` in `src/config/data_paths.py` returns the latest such dir (lexicographic max = newest timestamp). Prediction step uses that dir for XGBoost joblibs.

---

## Modeling

- **Same train/validation split for arousal and valence**  
  In `_load_and_split`, we call `train_test_split` twice (once for arousal, once for valence) with the **same** `X`, `test_size`, and `random_state`. That yields the same train/val indices for both targets, so we compare models on the same holdout set and avoid one target having a harder/easier split.

- **Feature NaN imputation before split**  
  Rows with NaN in `FEATURE_COLUMNS` are not dropped; we impute with column median (or 0 if the column is all NaN, e.g. `tempo_bpm` when no audio was available). Otherwise all-NaN columns would remove every row. Targets (arousal/valence) are still required to be non-NaN; those rows are dropped.

- **XGBoost as production model**  
  `generate_emotion_predictions` loads only `arousal_xgboost.joblib` and `valence_xgboost.joblib`. RF, Ridge, and ElasticNet are trained for comparison and metrics; the deployed pipeline uses XGBoost. See `docs/MLOPS.md` and `src/pipeline/generate_emotion_predictions.py`.

- **Fast path (`--fast` / `N_ITER_FAST = 5`)**  
  Reduces RandomizedSearchCV iterations per model for quick iteration (e.g. dev or CI). Full runs use the larger `n_iter` (e.g. 48–72) for better tuning.

---

## DEAM and annotations

- **CSV separator: try `;` then `,`**  
  DEAM annotation files sometimes use semicolon (European locale). `deam_labels_loader` tries `sep=";"` first, then falls back to default/`,` if the result looks wrong (e.g. single column). See comment in `_load_static_song_level` and similar in `_load_dynamic_aggregated`.

- **Duplicate song_id: keep first**  
  When merging multiple static annotation files, we `drop_duplicates(subset=["song_id"], keep="first")` so the first occurrence wins. DEAM layout can have the same song in more than one file.

---

## Analyst and semantic layer

- **Fallback when no XAI key**  
  If `XAI_API_KEY` is not set, `query_to_intent` uses a rule-based fallback (keyword → tags) so the analyst script and demos work without an API key (e.g. CI). See `query_llm.py` and `docs/PHASE4_ANALYST.md`.

- **Tag set must match**  
  `query_llm.VALID_TAGS` must match `src.semantic.list_tags()`. LLM output is filtered to these tags; unknown tags are dropped and may default to `Neutral`.

---

## Emotion map (Shiny)

- **Quadrant order**  
  Quadrants are (valence, arousal): high arousal + low valence = tense; high both = happy; low both = sad; low arousal + high valence = calm. Order in code matches this (left/right = valence, bottom/top = arousal).

- **Colour-by options**  
  Colour by key, tempo_bucket, key_note, key_mode, or key_signature when those columns exist (from song_features merge). If not present, a single default colour is used.
