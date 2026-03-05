# Build plan — work spread and agent usage

## Are we using agents to build this?

**Short answer:** The *product* includes an **AI Music Analyst Agent** (the thing users talk to). How we *build* the project is up to you.

| Kind | What | Where |
|------|------|--------|
| **Product agents** | AI Music Analyst (and optionally Semantic Layer as an “agent” that maps concepts) | In the demo: user asks questions → agent returns recommendations. |
| **Build process** | Right now it’s **you + one assistant** (me). We can optionally use **builder agents** (e.g. Cursor subagents / MCP tasks) to own whole phases in parallel. | Not required; useful if you want to parallelize or role-play “Data Pipeline Agent” vs “ML Modeling Agent” during build. |

**If you want agents involved in the build:** We can assign each phase below to a dedicated builder agent (e.g. “you implement Phase 1”, “subagent implements Phase 2”, etc.) so work is spread and each chunk has a clear owner. Otherwise we proceed phase-by-phase with a single build flow.

---

## Work spread — phased plan

Work is split so we don’t do too much at once. Each phase has a **scope**, **output**, **dependency**, and **optional agent owner**.

---

### Phase 1 — Data & feature pipeline  
**Owner (optional):** Data Pipeline Agent / you+assistant

| Item | Detail |
|------|--------|
| **Scope** | Load audio from a folder, extract acoustic features, write one feature table. No model, no UI. |
| **Output** | `data/processed/song_features.csv`: **one row per song** = song identifier + acoustic statistics (tempo, spectral_centroid, energy, mfcc_mean, chroma_variance). No audio in the CSV, only IDs and numbers. |
| **Depends on** | Nothing (start here). |
| **Rough effort** | ~1–1.5 h. |
| **Done when** | You can point the script at `data/raw/` (or your samples path) and get a valid feature table. |

---

### Phase 2 — Emotion prediction model  
**Owner (optional):** ML Modeling Agent

| Item | Detail |
|------|--------|
| **Scope** | Read feature table, (optionally) join with DEAM/EmoMusic labels, train arousal/valence model, write predictions. No semantic layer, no agent UI. |
| **Output** | `data/processed/emotion_predictions.csv` with `song_id`, `predicted_arousal`, `predicted_valence`, optionally `emotion_cluster`. |
| **Depends on** | Phase 1 (feature table exists). |
| **Rough effort** | ~1–1.5 h. |
| **Done when** | Every song in the feature table has arousal/valence (and optionally cluster); we have a simple eval (e.g. correlation or MSE if labels exist). |

---

### Phase 3 — Semantic layer  
**Owner (optional):** Semantic Layer Agent

| Item | Detail |
|------|--------|
| **Scope** | Define music attributes (High Energy, Calm Focus, Dramatic, Joyful, etc.) as rules on top of arousal/valence (and optionally features). Expose as a small API or module (e.g. “tag song” or “filter by concept”). No natural-language agent yet. |
| **Output** | Semantic layer module + definitions (code + short doc or config). |
| **Depends on** | Phase 2 (predictions table). |
| **Rough effort** | ~30–45 min. |
| **Done when** | We can programmatically ask “which songs are High Energy?” or “Calm Focus?” and get consistent results. |

---

### Phase 4 — AI Music Analyst Agent  
**Owner (optional):** AI Analyst Agent (builder)

| Item | Detail |
|------|--------|
| **Scope** | Simple interface: user question in → map to semantic layer + structured query → return ranked songs + short explanation. Can be CLI, minimal web UI, or notebook. |
| **Output** | Working flow: e.g. “Recommend energetic but positive music” → list of songs + explanation. |
| **Depends on** | Phase 3 (semantic layer). |
| **Rough effort** | ~1–1.5 h. |
| **Done when** | Demo-able: 3–5 example questions work end-to-end. |

---

### Phase 5 — Visualization  
**Owner (optional):** Visualization Agent

| Item | Detail |
|------|--------|
| **Scope** | Single emotion map: valence (x) × arousal (y), points = songs, optional cluster labels / highlighted recommendations. |
| **Output** | Script or notebook that produces the plot; optional export to image. |
| **Depends on** | Phase 2 (predictions); can be built in parallel with Phase 3/4 if desired. |
| **Rough effort** | ~30–45 min. |
| **Done when** | One clear plot that shows emotion space and (optionally) example songs. |

---

### Phase 6 — Polish (optional)  
**Owner:** You + assistant

| Item | Detail |
|------|--------|
| **Scope** | Architecture diagram, README/runbook, optional embedding-space twist. |
| **Output** | Diagram (e.g. Mermaid or image), README updated with “how to run” and interview talking points. |
| **Depends on** | Phases 1–5. |
| **Rough effort** | ~30–45 min. |

---

## Dependency overview

```
Phase 1 (features)     →  Phase 2 (model)  →  Phase 3 (semantic)  →  Phase 4 (analyst agent)
                              ↓
                        Phase 5 (viz)      [can start after Phase 2]
                              ↓
                        Phase 6 (polish)   [after 1–5]
```

---

## How to use this plan

1. **No builder agents:** Work through Phase 1 → 2 → 3 → 4 → 5 → 6 in order (or do 5 in parallel with 3/4).
2. **With builder agents:** Assign each phase to a separate agent (or human) so Phase 1 and Phase 2 prep can be done in parallel where possible; Phase 3 waits for Phase 2, Phase 4 for Phase 3; Phase 5 can run as soon as Phase 2 exists.
3. **Checkpoint after each phase:** Confirm the output artifact (e.g. `song_features.csv`, then `emotion_predictions.csv`) before starting the next phase so we don’t overbuild or drift.

If you tell me whether you want to use builder agents or not, we can either assign phases to agents or proceed step-by-step from Phase 1 with you and your music samples.
