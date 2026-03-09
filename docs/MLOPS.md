# MLOps pipeline: feature engineering and model generation

Single repeatable pipeline from raw data to validated features and trained models. All steps are run from the **project root**.

---

## Pipeline stages

| Stage | Description | Inputs | Outputs |
|-------|-------------|--------|---------|
| **1. Feature engineering (enrich)** | DEAM feature CSVs + audio → one row per song with 11 numeric features + tempo, key, genre | `data/deam_csvs/features/`, `data/audio/` | `data/processed/song_features.csv` |
| **2. Labels + join** | DEAM annotations → per-song labels; inner join with song_features | `data/deam_csvs/annotations/`, `song_features.csv` | `data/processed/deam_labels.csv`, `data/processed/modeling_dataset.csv` |
| **3. Model training** | Train RF, Ridge, ElasticNet, XGBoost with CV tuning; XGBoost is production | `modeling_dataset.csv` | `models/arousal_*.joblib`, `models/valence_*.joblib` |
| **4. Batch prediction** | Apply XGBoost to all songs in song_features | `song_features.csv`, `models/*xgboost*.joblib` | `data/processed/emotion_predictions.csv` |
| **5. Validation** | Schema, row counts, data-quality checks on all CSVs | All of the above | Exit 0/1; console report |

---

## Entry point

**Single command (recommended):**

```bash
python3 scripts/run_full_pipeline.py
```

Runs stages 1–5 in order. Exits 0 only if validation passes. Writes a run manifest to `data/processed/pipeline_run.json` (timestamp, steps, status) for traceability.

**Options:**

- `--force` — Run all steps even if a step’s checkpoint exists (no resume skip).
- `--fast` — Fast training: 5 RandomizedSearchCV iterations per model instead of 48–72 (for quick iteration).

**Run individual steps** (for debugging or partial rebuilds):

| Step | Command |
|------|---------|
| 1. Enrich only | `python3 scripts/run_enrich_pipeline.py` |
| 2. Labels + join only | `python3 scripts/run_phase2_labels_and_join.py` |
| 3. Train only | `python3 scripts/train_emotion_models.py` (add `--fast` for 5 iterations per model) |
| 4. Predict only | `python3 scripts/run_emotion_predictions.py` |
| 5. Validate only | `python3 scripts/validate_song_features.py` (or `python3 -m src.pipeline.validate_song_features`) |

---

## Artifacts and paths

- **Features:** `data/processed/song_features.csv` (18 columns: song_id, 10 DEAM features, tempo_bpm, genre, key, key_note, key_mode, key_signature, is_major).  
- **Labels:** `data/processed/deam_labels.csv`.  
- **Modeling dataset:** `data/processed/modeling_dataset.csv` (features + labels, inner join).  
- **Models:** `models/arousal_xgboost.joblib`, `models/valence_xgboost.joblib` (git-ignored).  
- **Predictions:** `data/processed/emotion_predictions.csv`.  
- **Run manifest:** `data/processed/pipeline_run.json` (written by `run_full_pipeline.py`).

Paths are centralized in `src/config/data_paths.py`. For design rationale (e.g. song_id as string, same train/val split, XGBoost as production), see [docs/DECISIONS.md](DECISIONS.md).

---

## Validation gates

Validation (stage 5) checks:

- **song_features.csv:** 18 required columns (including key_note, key_mode, key_signature, is_major), row count, no duplicate `song_id`, numeric dtypes, ≥90% non-NaN tempo and non-unknown key.  
- **deam_labels.csv:** `song_id`, `arousal`, `valence`; no duplicates; numeric.  
- **modeling_dataset.csv:** all feature columns + arousal, valence; no duplicates.  
- **emotion_predictions.csv:** `song_id`, `predicted_arousal`, `predicted_valence`; row count matches song_features; no NaN in predictions.

If any check fails, the pipeline exits 1. Fix inputs or re-run the failing stage and re-validate.
