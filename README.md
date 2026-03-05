# Music Emotion Intelligence Pipeline + AI Analyst

An agentic ML pipeline demo that turns **raw audio → structured features → emotion predictions → semantic music concepts → AI-powered analysis and recommendations**. Built to showcase ML engineering, semantic layers, and agentic AI on a music-emotion theme.

**Target build:** 4–6 hours · **Demo narrative:** pipeline → feature store → model → semantic layer → AI analyst

---

## Purpose — Why use this tool?

**For someone using it:** The tool lets you **explore and recommend music by how it feels**, not just by genre or metadata. You can:

- **Ask in plain language** — e.g. *“Find calm music for studying”*, *“Recommend energetic, positive tracks”*, *“What’s the emotional profile of this song?”*
- **Get back** — ranked song lists, emotional labels (e.g. calm, joyful, tense), and short explanations.
- **See your library on an emotion map** — valence vs arousal, with clusters so you can browse by mood.

So the purpose is: **turn a music collection into emotion-aware, queryable data and interact with it through an AI analyst** instead of manual tagging or guesswork.

**For someone evaluating the project (e.g. interview):** It demonstrates ML pipelines, feature stores, semantic layers, and agentic AI in one coherent flow.

---

## System architecture (high-level)

```
Raw Audio
    → Feature Engineering Pipeline
    → ML Feature Store (song_features)
    → Emotion Prediction Model
    → Emotion Predictions Table
    → Semantic Layer (music meaning)
    → AI Music Analyst Agent
```

**Components to highlight:**

| Layer | Responsibility |
|-------|-----------------|
| **Pipeline** | Load audio, extract features, normalize, store feature table |
| **ML modeling** | Train arousal/valence model, evaluate, store predictions |
| **Semantic knowledge** | Define business logic, standardize music attributes, AI-safe access |
| **Agent interaction** | Interpret queries, map to semantics, return analysis & recommendations |

---

## Phase 1 — Data & feature pipeline

**Goal:** Raw audio → structured features suitable for ML.

- **Inputs:** Small set of audio tracks or pre-existing music dataset.
- **Pipeline:** Load audio → extract acoustic features → normalize & structure → store in feature table.

**Features to extract:** tempo, spectral brightness (centroid), energy, MFCC summary stats, chroma features, zero-crossing rate.

**Output — Song Feature Table (CSV):**

- **One row per song.** Each row = **song identifier + all acoustic statistics** for that song (no audio, no raw waveforms).
- **Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `song_id` | string | Track identifier (e.g. filename stem or custom ID) |
| `tempo` | float | Beats per minute |
| `spectral_centroid` | float | Spectral brightness (Hz) |
| `energy` | float | Perceptual energy (e.g. RMS or spectral rolloff-derived) |
| `mfcc_coef1` | float | MFCC coefficient 1 mean over time (single coefficient) |
| `auditory_band_variance` | float | Variance across 26 Rfilt filter-band means (timbre; not chroma) |

So the CSV is **song + statistics only** — identifiers and numeric features ready for ML; no embedded audio or long time-series.

**Enrich step (optional):** Run `python3 scripts/run_enrich_pipeline.py` to produce a **richer** table: **10 DEAM core features** (spectral_centroid, energy, mfcc_coef1, auditory_band_variance, spectral_rolloff50, zcr, spectral_flux, spectral_variance, spectral_entropy, spectral_harmonicity) plus **tempo (BPM)** and **musical key** extracted from audio, and a **genre** column (set to `unknown`; DEAM has no genre labels). Output: `data/processed/song_features.csv` with 14 columns.

**Data dictionary:** Column names, types, sources, and definitions for all pipeline outputs are in **[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)**. Consult it before Phase 2 and for downstream use.

**Validation:** After building or re-running enrich, run `python3 scripts/validate_song_features.py` to check schema, row count, and tempo/key coverage.

**Run from start to finish:** To rebuild all pipeline outputs and validate in one go (enrich → labels + join → train models → emotion_predictions → validate), run `python3 scripts/run_full_pipeline.py` from the project root. Requires `data/audio/`, `data/deam_csvs/` to be populated; takes several minutes (audio processing + model training).

**Key idea:** This table is the **ML feature layer**, reusable across models.

---

## Phase 2 — Emotion prediction model

**Goal:** Predict emotional characteristics from acoustic features.

- **Emotion representation:** **Arousal** (energy/intensity), **Valence** (positive vs negative).
- **Process:** Join feature table with labeled data (DEAM or EmoMusic) → train regression model → evaluate → store predictions.

**Output — Emotion Prediction Table:**

| Column | Description |
|--------|-------------|
| `song_id` | Track identifier |
| `predicted_arousal` | Model output |
| `predicted_valence` | Model output |
| `emotion_cluster` | (Optional) e.g. calm / energetic / tense / joyful |

### Production model: XGBoost

Phase 2 uses **XGBoost** (one regressor per target) as the **production** emotion model. We also train RandomForest, Ridge, and ElasticNet for comparison; XGBoost consistently gives the best holdout performance.

- **Inputs:** 11 numeric features from `modeling_dataset.csv`:  
  `spectral_centroid`, `energy`, `mfcc_coef1`, `auditory_band_variance`, `spectral_rolloff50`, `zcr`, `spectral_flux`, `spectral_variance`, `spectral_entropy`, `spectral_harmonicity`, `tempo_bpm`.
- **Targets:** DEAM **arousal** and **valence** labels (per-song averages).
- **Training:** 80/20 train/holdout split; **5-fold RandomizedSearchCV** on the training set for hyperparameters; best estimator refit on full training data. Saved models: `models/arousal_xgboost.joblib`, `models/valence_xgboost.joblib`.

**Holdout results (XGBoost, ~1,800 songs):**

| Target  | RMSE   | R²     | Pearson r |
|---------|--------|--------|-----------|
| Arousal | ~0.98  | ~0.43  | ~0.66     |
| Valence | ~0.93  | ~0.41  | ~0.64     |

**Findings from model comparison:**

- **XGBoost** slightly outperforms RandomForest (lower RMSE, higher R² and Pearson r). Both are suitable for ranking and mood buckets.
- **Ridge** and **ElasticNet** are clearly worse (~0.34–0.37 R²); non-linearity matters for this task.
- Predictions are **good for relative use** (rank songs by mood, build playlists, emotion maps) and **not for exact 1–9 scores** — typical error is about one point on the scale; ~40% of variance is explained, the rest is label noise and missing signal.

**How to train and reproduce metrics:**  
Run `python3 scripts/train_emotion_models.py` from the project root. The script trains all four model types (RandomForest, Ridge, ElasticNet, XGBoost) with CV tuning and prints a metrics table per type. Trained models are written under `models/` (git-ignored).

**Key idea:** The model **enriches the feature store** with semantic predictions for downstream systems (semantic layer, analyst agent, emotion map).

---

## Phase 3 — Semantic layer

**Goal:** Translate ML outputs into interpretable musical concepts.

Define attributes and rules, for example:

- **High Energy** — arousal above threshold.
- **Calm Focus** — low arousal + positive valence.
- **Dramatic Music** — high arousal + low valence.
- **Joyful Music** — high valence + moderate arousal.

**Responsibilities:** Define business logic, standardize attributes, act as **translation layer** for AI systems.

**Key idea:** Semantic layer prevents agents from touching raw data and ensures **consistent interpretation** of metrics (enterprise AI governance pattern).

---

## Phase 4 — AI Music Analyst Agent

**Goal:** Interface for exploring the music dataset via natural language.

**Agent flow:** Interpret user query → map to semantic layer definitions → generate structured queries → return results and explanations.

**Example questions:**

- *Which songs are energetic and positive?*
- *Find calm music suitable for studying.*
- *What is the emotional profile of this track?*
- *Recommend songs with similar emotional characteristics.*

**Output format:** Ranked song list, emotional interpretation, short explanation.

**Key idea:** Agent acts as an **AI analyst** on top of structured ML outputs.

---

## Phase 5 — Visualization layer

**Goal:** Quick visual understanding of results.

- **Emotion map:** 2D plot — **X:** valence, **Y:** arousal.
- **Clusters:** Calm, Joyful, Tense, Excited (and optionally others).
- **Optional:** Label example songs, highlight recommended songs.

**Key idea:** Visually intuitive explanation of the ML system.