# Data dictionary

Defines the schema and meaning of pipeline output files. Use this before Phase 2 (and for downstream consumers) to interpret `song_features.csv` and, later, `emotion_predictions.csv`.

---

## File: `data/processed/song_features.csv`

**Purpose:** One row per song; ML-ready feature layer plus tempo, genre, and key. Produced by the Phase 1 feature pipeline and enrich step.

**Row count:** One per DEAM feature CSV (e.g. 1,802).  
**Primary key:** `song_id` (string; matches DEAM file stem and audio filename stem).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| **song_id** | string | DEAM / audio | Track identifier. From DEAM feature filename stem (e.g. `2.csv` → `2`) and audio filename stem (e.g. `746.mp3` → `746`). Used to join DEAM features with audio-derived fields. |
| **spectral_centroid** | float | DEAM | Spectral brightness (Hz). Mean over time of `pcm_fftMag_spectralCentroid_sma_amean`. Higher = brighter tone. |
| **energy** | float | DEAM | Perceptual energy (RMS). Mean over time of `pcm_RMSenergy_sma_amean`. Typical range ~0–1; higher = louder. |
| **mfcc_coef1** | float | DEAM | MFCC coefficient 1 mean over time (`pcm_fftMag_mfcc_sma[1]_amean`). Single coefficient, not an aggregate across all MFCCs; captures coarse spectral shape. |
| **auditory_band_variance** | float | DEAM | Variance across 26 OpenSMILE Rfilt (auditory filter-band) means. Timbre/spectral distribution, not chroma (pitch-class). |
| **spectral_rolloff50** | float | DEAM | Frequency below which 50% of spectral energy lies (Hz). Mean over time of `pcm_fftMag_spectralRollOff50.0_sma_amean`. |
| **zcr** | float | DEAM | Zero-crossing rate. Mean over time of `pcm_zcr_sma_amean`. Higher often indicates noisiness or percussive content. |
| **spectral_flux** | float | DEAM | Rate of change of spectrum over time. Mean of `pcm_fftMag_spectralFlux_sma_amean`. |
| **spectral_variance** | float | DEAM | Variance of the spectrum. Mean of `pcm_fftMag_spectralVariance_sma_amean`. |
| **spectral_entropy** | float | DEAM | Spectral entropy. Mean of `pcm_fftMag_spectralEntropy_sma_amean`. |
| **spectral_harmonicity** | float | DEAM | Harmonic-to-noise ratio. Mean of `pcm_fftMag_spectralHarmonicity_sma_amean`. |
| **tempo_bpm** | float | Audio | Beats per minute from audio (librosa beat tracking). May be NaN if extraction failed or no audio file matched. Typical range ~40–250. |
| **genre** | string | Placeholder | Always `"unknown"` in current pipeline; DEAM does not provide genre. Reserved for future tag or classifier. |
| **key** | string | Audio | Estimated musical key (e.g. `C major`, `A minor`). Chroma (STFT, then CQT fallback) + Krumhansl–Schmuckler profiles; cosine similarity. `"unknown"` if extraction failed, no audio matched, or chroma was unusable. |

**Missing values:**
- DEAM-derived columns: may be NaN if the source DEAM CSV lacked that column or had invalid values.
- `tempo_bpm`: NaN when no matching audio file or when beat tracking failed.
- `key`: `"unknown"` when no matching audio or when key estimation failed.
- `genre`: always `"unknown"` until an external source or classifier is added.

**Source pipeline steps:**
1. DEAM feature CSVs (`data/deam_csvs/features/*.csv`) → per-song mean (and auditory_band_variance from Rfilt) → 10 DEAM columns.
2. Audio files (`data/audio/*.mp3`) → tempo (librosa), key (chroma + key profiles) → tempo_bpm, key.
3. Merge on song_id (left join); add genre = "unknown".

---

## File: `data/processed/deam_labels.csv` (Phase 2)

**Purpose:** One row per song; DEAM ground-truth arousal and valence for training the emotion model. Produced by Phase 2 step 1 (DEAM annotations loader).

**Row count:** One per song present in DEAM annotations (song-level static or aggregated from dynamic).  
**Primary key:** `song_id` (string; matches DEAM annotation song IDs).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| **song_id** | string | DEAM | Track identifier; matches song_features.csv and DEAM annotation files. |
| **arousal** | float | DEAM | Averaged arousal (intensity/activation). Typical range 1–9 in DEAM. |
| **valence** | float | DEAM | Averaged valence (positive vs negative). Typical range 1–9 in DEAM. |

**Source:** `data/deam_csvs/annotations/` — song-level static CSVs (`static_annotations_averaged_songs_*.csv`) preferred; else dynamic per-second arousal.csv/valence.csv aggregated to mean per song.

---

## File: `data/processed/modeling_dataset.csv` (Phase 2)

**Purpose:** One row per song with both features and labels; used to train the arousal/valence model. Inner join of song_features.csv and deam_labels.csv.

**Row count:** Intersection of song_features and deam_labels (songs that have both).  
**Primary key:** `song_id`.

**Columns:** All columns from `song_features.csv` (14 columns) plus **arousal** and **valence** from `deam_labels.csv` (16 columns total).

**Source:** Phase 2 step 2: `run_build_modeling_dataset()` joins song_features + deam_labels on song_id (inner).

---

## File: `data/processed/emotion_predictions.csv` (Phase 2)

**Purpose:** One row per song; predicted arousal and valence from the production XGBoost model. Produced by the Phase 2 prediction step (`run_emotion_predictions`).

**Row count:** One per row in `song_features.csv` (same set of songs).  
**Primary key:** `song_id`.

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| **song_id** | string | song_features | Track identifier; matches song_features.csv. |
| **predicted_arousal** | float | XGBoost model | Predicted arousal (energy/intensity). Scale comparable to DEAM 1–9. |
| **predicted_valence** | float | XGBoost model | Predicted valence (positive vs negative). Scale comparable to DEAM 1–9. |

**Source:** `scripts/run_emotion_predictions.py` loads `song_features.csv`, runs `models/arousal_xgboost.joblib` and `models/valence_xgboost.joblib` on the 11 feature columns, and writes this file. Train models first with `python3 scripts/train_emotion_models.py`.

---

## Reference: DEAM source columns

The 10 DEAM-derived columns in `song_features.csv` are aggregated from the following OpenSMILE / Geneva minimal feature set columns (one CSV per song, `;` separator, frame-level):

- `pcm_fftMag_spectralCentroid_sma_amean`
- `pcm_RMSenergy_sma_amean`
- `pcm_fftMag_mfcc_sma[1]_amean`
- `audSpec_Rfilt_sma[0]_amean` … `audSpec_Rfilt_sma[25]_amean` (auditory_band_variance = variance of their means)
- `pcm_fftMag_spectralRollOff50.0_sma_amean`
- `pcm_zcr_sma_amean`
- `pcm_fftMag_spectralFlux_sma_amean`
- `pcm_fftMag_spectralVariance_sma_amean`
- `pcm_fftMag_spectralEntropy_sma_amean`
- `pcm_fftMag_spectralHarmonicity_sma_amean`

Aggregation: **mean over time** (all frames in the song’s DEAM feature CSV), except auditory_band_variance (variance across the 26 Rfilt band means).
