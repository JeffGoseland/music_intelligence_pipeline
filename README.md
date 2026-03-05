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

## Confirmation: We're on the same page

We are building:

1. **Feature pipeline** — Ingest audio, extract acoustic features, produce a **Song Feature Table** (ML feature layer).
2. **Emotion model** — Train on features (e.g. DEAM/EmoMusic), predict **arousal** and **valence**; output **Emotion Prediction Table**.
3. **Semantic layer** — Define interpretable music attributes (High Energy, Calm Focus, Dramatic, Joyful, etc.) from ML outputs; act as the **translation layer** for the AI agent.
4. **AI Music Analyst Agent** — Parse user questions, map to semantic definitions, query structured data, return **ranked songs + emotional interpretation + short explanation**.
5. **Visualization** — **Emotion map** (valence × arousal) with clusters and optional song labels.

**Core ideas demonstrated:** ML data pipelines · feature engineering · applied modeling · semantic layer design · agentic AI interface.

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
| `mfcc_mean` | float | Summary of MFCCs (e.g. mean of first coefficient across frames) |
| `chroma_variance` | float | Spread of chroma features (pitch-class distribution) |

So the CSV is **song + statistics only** — identifiers and numeric features ready for ML; no embedded audio or long time-series.

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

**Key idea:** The model **enriches the feature store** with semantic predictions for downstream systems.

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

---

## Agent roles (for implementation)

| Agent | Responsibilities | Output |
|-------|------------------|--------|
| **Data Pipeline Agent** | Ingest audio, extract acoustic features, create feature table, validate schema | `song_features` dataset |
| **ML Modeling Agent** | Train emotion model, evaluate, generate predictions | `emotion_predictions` dataset |
| **Semantic Layer Agent** | Define music attributes, map ML outputs to semantic concepts, maintain metadata | Music semantic layer |
| **AI Analyst Agent** | Parse queries, map to semantic definitions, query data, produce analysis | Recommendations and insights |
| **Visualization Agent** (optional) | Arousal–valence plot, clusters, annotate songs | Emotion map visualization |

---

## Deliverables (interview-ready)

1. **Architecture diagram** — Pipeline, ML model, semantic layer, AI agent (systems thinking).
2. **Small interactive demo** — e.g. *“Recommend energetic but positive music”* → agent returns songs.
3. **Simple visualization** — Emotion map with clusters.

---

## One-line narrative (interview)

> *“The ML pipeline converts raw audio into structured features, the model predicts emotional attributes, the semantic layer translates those predictions into interpretable music concepts, and the AI agent uses that structure to analyze and recommend music.”*

**Concepts to emphasize:** ML pipelines · feature engineering layer · semantic abstraction · AI-safe data access · agentic analytics.

---

## Optional enhancement (senior-engineer twist)

**Music emotion embedding space** — Dimensionality reduction (e.g. PCA/UMAP) on features + predictions, with **explainable feature influence** on emotional perception (e.g. which acoustic features drive which region of the space). Adds a clear “why” to the emotion map.

---

## Why this demo works

- **ML engineering** — Pipelines, feature store, model training, evaluation.
- **Analytics engineering** — Semantic layer, consistent definitions, governed access.
- **AI integration** — Agent on structured data, not raw tables.
- **Agent-based interaction** — Natural language → semantic layer → structured queries → insights.

Connects cleanly to **music cognition / emotion** as an application domain.

---

## Build plan (work spread and agents)

**Are we using agents to build this?** The *product* includes an **AI Music Analyst Agent** (what users talk to). The *build* can be you + one assistant, or we can assign phases to separate builder agents to parallelize. See **[docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)** for:

- Clarification: product agents vs. build-time agents
- Phased work breakdown (scope, output, dependency, rough effort)
- Dependency diagram and how to run with or without builder agents

**Suggested order:** Phase 1 (features) → 2 (model) → 3 (semantic) → 4 (analyst); Phase 5 (viz) can start after Phase 2. Checkpoint after each phase before moving on.

If you want to adjust any phase, agent role, or deliverable, say what to change and we’ll update this README and the plan accordingly.
