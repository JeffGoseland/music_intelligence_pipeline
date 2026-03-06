# Music Emotion Intelligence Pipeline + AI Analyst

This project turns a small music collection into an **emotion-aware playground**.  
You start with raw audio and DEAM features; the pipeline builds **song-level features**, learns an **emotion model**, adds a **semantic layer** of music tags, and exposes it all through an **AI analyst** and an **emotion map UI**.

Think of it as a compact, interview-ready demo of:

- **Feature engineering** on real data (DEAM + audio)
- **Emotion modeling** (arousal / valence)
- **Semantic layer** for music concepts (High Energy, Calm Focus, Dramatic, …)
- **AI analyst** on top of structured outputs (Grok or rule-based)
- **Visualization** of the emotion space (Shiny map)

The whole build is sized for **4–6 hours** of focused work and is designed to be easy to explain on a whiteboard.

---

## At a glance

- **Run everything:** `python3 scripts/run_full_pipeline.py --fast`  
  (resume without `--force`, see [docs/MLOPS.md](docs/MLOPS.md) for stages and checkpoints).
- **Explore tags:** `python3 scripts/run_semantic_example.py`
- **Ask the analyst:**  
  - With Grok: set `XAI_API_KEY` (see [docs/PHASE4_ANALYST.md](docs/PHASE4_ANALYST.md)), then  
    `python3 scripts/run_analyst.py "Find calm music for studying"`
  - Without a key: same command, using the rule-based fallback.
- **See the map:** `shiny run src/app_emotion_map.py`

### Phase status

| Phase | Status | Notes |
|-------|--------|--------|
| **1. Data & feature pipeline** | Done | Enrich → `song_features.csv`, validation, full pipeline with checkpointing |
| **2. Emotion prediction** | Done | Train (RF, Ridge, ElasticNet, XGBoost), versioned models `models/<run_id>/`, predictions → `emotion_predictions.csv`, MLOps (metrics, run_info, resume) |
| **3. Semantic layer** | Done | Code module in `src/semantic` (emotion_tags); tag set and rules in [docs/SEMANTIC_LAYER.md](docs/SEMANTIC_LAYER.md). |
| **4. AI Music Analyst** | Done | Grok (x.ai) + fallback; CLI: `python3 scripts/run_analyst.py "your question"`. Set `XAI_API_KEY` (see [docs/PHASE4_ANALYST.md](docs/PHASE4_ANALYST.md)). |
| **5. Visualization** | Done | Shiny emotion map (`src/app_emotion_map.py`); Colour by key/tempo/cluster |

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

**Enrich step (optional):** Run `python3 scripts/run_enrich_pipeline.py` to produce a **richer** table: **10 DEAM core features** (spectral_centroid, energy, mfcc_coef1, auditory_band_variance, spectral_rolloff50, zcr, spectral_flux, spectral_variance, spectral_entropy, spectral_harmonicity) plus **tempo (BPM)**, **musical key**, and **key-derived columns** (key_note, key_mode, key_signature, is_major) extracted from audio, and a **genre** column (set to `unknown`; DEAM has no genre labels). Output: `data/processed/song_features.csv` with 18 columns.

**Data dictionary:** Column names, types, sources, and definitions for all pipeline outputs are in **[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)**. Consult it before Phase 2 and for downstream use.

**Validation:** After building or re-running enrich, run `python3 scripts/validate_song_features.py` to check schema, row count, and tempo/key coverage.

**Run from start to finish:** From the project root, run `python3 scripts/run_full_pipeline.py` (resume from last run, skips completed steps) or `python3 scripts/run_full_pipeline.py --force` (run all steps). Add `--fast` to use 5 CV iterations per model for quick training. Requires `data/audio/`, `data/deam_csvs/`; takes several minutes. Writes run manifest to `data/processed/pipeline_run.json` and checkpoints to `data/processed/.checkpoints/`.

**MLOps:** Stages, versioned model paths (`models/<run_id>/`), validation gates, and per-step commands are in **[docs/MLOPS.md](docs/MLOPS.md)**.

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
- **Training:** 80/20 train/holdout split; **5-fold RandomizedSearchCV** on the training set for hyperparameters (default 48–72 iterations per model type); best estimator refit on full training data. Use the **fast path** (`--fast`) for 5 iterations per model when iterating quickly. Models are saved under **versioned** dirs: `models/<run_id>/` (e.g. `arousal_xgboost.joblib`, `valence_xgboost.joblib`, plus `training_metrics.txt` and `run_info.json`).

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
Run `python3 scripts/train_emotion_models.py` from the project root. The script trains all four model types (RandomForest, Ridge, ElasticNet, XGBoost) with CV tuning, prints a metrics table, and writes models plus `training_metrics.txt` and `run_info.json` to `models/<run_id>/` (git-ignored). Use `--fast` for 5 RandomizedSearchCV iterations per model (quick iteration). Standalone predict uses the latest run; the full pipeline uses the run it just trained.

**Key idea:** The model **enriches the feature store** with semantic predictions for downstream systems (semantic layer, analyst agent, emotion map).

---

## Phase 3 — Semantic layer

**Goal:** Translate ML outputs into interpretable musical concepts.

**Tag set and rules:** The semantic layer uses **6 tags + Neutral** derived from predicted arousal and valence (scale ~1–9). Tag definitions, thresholds (e.g. High Energy: arousal ≥ 6; Calm Focus: arousal ≤ 4 and valence ≥ 5), and usage (filter by tag, list tags) are in **[docs/SEMANTIC_LAYER.md](docs/SEMANTIC_LAYER.md)**. Songs can have multiple tags.

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