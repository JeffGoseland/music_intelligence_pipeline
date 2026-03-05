# Full plan and next steps

## Restatement of the entire plan

### Purpose

**Music Emotion Intelligence Pipeline + AI Analyst** — A demo that turns **raw audio → structured features → emotion predictions → semantic music concepts → AI-powered analysis and recommendations**. Target build ~4–6 hours.

- **For users:** Explore and recommend music by *how it feels* (plain-language questions → ranked songs + emotional labels + short explanations; optional emotion map).
- **For evaluation (e.g. interview):** Demonstrates ML pipelines, feature stores, semantic layers, and agentic AI in one flow.

**One-line narrative:** *“The ML pipeline converts raw audio into structured features, the model predicts emotional attributes, the semantic layer translates those into interpretable music concepts, and the AI agent uses that structure to analyze and recommend music.”*

---

### System architecture

```
Raw Audio (data/audio/)
    → Feature Engineering Pipeline
    → ML Feature Store (song_features)
    → Emotion Prediction Model
    → Emotion Predictions Table
    → Semantic Layer (music meaning)
    → AI Music Analyst Agent
```

| Layer | Responsibility |
|-------|----------------|
| **Pipeline** | Load audio / DEAM CSVs, extract or aggregate features, store feature table |
| **ML modeling** | Train arousal/valence model, evaluate, store predictions |
| **Semantic knowledge** | Define music attributes (High Energy, Calm Focus, etc.), AI-safe access |
| **Agent interaction** | Interpret queries, map to semantics, return analysis & recommendations |

---

### Data layout (done)

All project data lives under `data/`. These folders are **git-ignored** (not uploaded):

| Folder | Contents |
|--------|----------|
| `data/audio/` | Music files (1,802 DEAM MP3s). |
| `data/deam_csvs/features/` | DEAM feature CSVs — one per song (1,802 files). |
| `data/deam_csvs/annotations/` | DEAM annotations (arousal/valence, etc.) — same structure as DEAM. |
| `data/processed/` | Pipeline outputs: `song_features.csv`, `emotion_predictions.csv`, etc. |

Documentation for this layout: `data/README.md` (tracked). Data was copied from `/Volumes/Dockcase/Music Research /DEAM`; no code or scripts were added for the move.

---

### Phase 1 — Data & feature pipeline

**Goal:** Produce the **Song Feature Table** (one row per song = song + acoustic statistics).

- **Inputs:** DEAM feature CSVs in `data/deam_csvs/features/` (and optionally raw audio in `data/audio/` if we add librosa extraction later).
- **Process:** Read each feature CSV, aggregate per song (e.g. mean over time), map DEAM columns to our schema.
- **Output:** `data/processed/song_features.csv` with columns: `song_id`, `tempo`, `spectral_centroid`, `energy`, `mfcc_mean`, `chroma_variance`.
- **Done when:** One row per song; CSV is the ML feature layer for Phase 2.

---

### Phase 2 — Emotion prediction model

**Goal:** Predict arousal and valence from the feature table; optionally use DEAM annotations for training.

- **Inputs:** `song_features.csv` + DEAM annotations from `data/deam_csvs/annotations/`.
- **Process:** Join features with labels, train regression model(s), evaluate, write predictions.
- **Output:** `data/processed/emotion_predictions.csv` — `song_id`, `predicted_arousal`, `predicted_valence`, optionally `emotion_cluster`.
- **Done when:** Every song has predictions; simple eval (e.g. correlation/MSE) if labels exist.

---

### Phase 3 — Semantic layer

**Goal:** Translate ML outputs into interpretable music concepts.

- **Define:** High Energy, Calm Focus, Dramatic, Joyful, etc. (rules on arousal/valence/features).
- **Output:** Semantic layer module (code + config/doc) — e.g. “tag song” or “filter by concept.”
- **Done when:** We can programmatically ask “which songs are High Energy?” and get consistent results.

---

### Phase 4 — AI Music Analyst Agent

**Goal:** Natural-language interface for exploring the dataset.

- **Process:** User question → map to semantic definitions → structured query → ranked songs + short explanation.
- **Output:** Working flow (CLI, minimal UI, or notebook); e.g. “Recommend energetic but positive music” → list + explanation.
- **Done when:** 3–5 example questions work end-to-end.

---

### Phase 5 — Visualization

**Goal:** Emotion map (valence × arousal, points = songs, optional clusters/labels).

- **Output:** Script or notebook that produces the plot; optional export to image.
- **Can start:** After Phase 2 (predictions exist). Can run in parallel with Phase 3/4.

---

### Phase 6 — Polish (optional)

**Goal:** Architecture diagram, README/runbook, “how to run,” interview talking points; optional embedding-space twist.

---

### Dependencies

```
Phase 1 (features)  →  Phase 2 (model)  →  Phase 3 (semantic)  →  Phase 4 (analyst)
                              ↓
                        Phase 5 (viz)   [after Phase 2]
                              ↓
                        Phase 6 (polish) [after 1–5]
```

---

## What’s done

- [x] Project and docs (README, BUILD_PLAN, data layout).
- [x] Data and file structure: `data/audio/`, `data/deam_csvs/features/`, `data/deam_csvs/annotations/`, `data/processed/`.
- [x] DEAM data copied into those folders (audio + CSVs); originals unchanged.
- [x] `.gitignore` updated so `data/audio/`, `data/deam_csvs/`, `data/processed/` are not committed.
- [x] `data/README.md` describing the layout (tracked).
- [x] **Phase 1:** Path config (`src/config/data_paths.py`), DEAM feature loader (`src/pipeline/deam_feature_loader.py`), `data/processed/song_features.csv` (1,802 rows). Run: `python scripts/run_feature_pipeline.py` or `python -m src.pipeline.deam_feature_loader`.
- [x] **Enrich step:** 10 DEAM core features + tempo (BPM), genre, key. `src/pipeline/deam_feature_loader.py` (rich=True), `src/pipeline/audio_derived_features.py` (tempo + key from audio), `src/pipeline/enrich_song_features.py` (merge + genre). Run: `python scripts/run_enrich_pipeline.py`. Columns: song_id, spectral_centroid, energy, mfcc_mean, chroma_variance, spectral_rolloff50, zcr, spectral_flux, spectral_variance, spectral_entropy, spectral_harmonicity, tempo_bpm, genre, key. Genre is "unknown" (DEAM has no genre).
- [x] **Data dictionary:** [docs/DATA_DICTIONARY.md](DATA_DICTIONARY.md) documents all columns in `song_features.csv` (and placeholder for `emotion_predictions.csv`). Required reading before Phase 2.

---

## Next steps

1. ~~**Phase 1 — Feature pipeline**~~ **Done.**

2. **Phase 2 — Emotion model**  
   - Load `song_features.csv` and DEAM annotations from `data/deam_csvs/annotations/`.  
   - Train arousal/valence model; evaluate; write `emotion_predictions.csv`.  
   - **Checkpoint:** Confirm predictions table; then Phase 3 (and optionally Phase 5 in parallel).

3. **Phases 3 → 4 → 5 → 6**  
   - Follow BUILD_PLAN: semantic layer → AI analyst agent → visualization → polish.  
   - Checkpoint after each phase.

**Recommended next action:** Start Phase 2 — emotion model (train arousal/valence from `song_features.csv` + DEAM annotations → `emotion_predictions.csv`).
