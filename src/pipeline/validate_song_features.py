"""
Validate data/processed/song_features.csv, deam_labels.csv, modeling_dataset.csv, emotion_predictions.csv.

Song features: required columns, dtypes, row count, duplicate song_id, NaN counts,
tempo/key coverage. DEAM labels: song_id, arousal, valence; no duplicates; numeric.
Modeling dataset: song_features columns + arousal, valence. Emotion predictions:
song_id, predicted_arousal, predicted_valence; row count matches song_features; no NaN.
"""

from pathlib import Path
import sys

import pandas as pd

# Expected schema (docs/DATA_DICTIONARY.md)
REQUIRED_COLUMNS = [
    "song_id",
    "spectral_centroid",
    "energy",
    "mfcc_coef1",
    "auditory_band_variance",
    "spectral_rolloff50",
    "zcr",
    "spectral_flux",
    "spectral_variance",
    "spectral_entropy",
    "spectral_harmonicity",
    "tempo_bpm",
    "genre",
    "key",
    "key_note",
    "key_mode",
    "key_signature",
    "is_major",
]
NUMERIC_FEATURE_COLUMNS = [
    "spectral_centroid",
    "energy",
    "mfcc_coef1",
    "auditory_band_variance",
    "spectral_rolloff50",
    "zcr",
    "spectral_flux",
    "spectral_variance",
    "spectral_entropy",
    "spectral_harmonicity",
]
TEMPO_MIN, TEMPO_MAX = 20.0, 300.0  # plausible BPM range for reporting
# Require at least this % of rows with audio-derived data; else enrich step likely failed
MIN_TEMPO_COVERAGE_PCT = 90.0
MIN_KEY_COVERAGE_PCT = 90.0


def _default_csv_path() -> Path:
    """Resolve song_features.csv path so the module works when run as script or as package."""
    try:
        from ..config.data_paths import SONG_FEATURES_PATH

        return SONG_FEATURES_PATH
    except ImportError:
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / "data" / "processed" / "song_features.csv"


def validate_song_features(
    csv_path: Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """
    Validate song_features.csv. Returns (all_passed, errors, info_messages).
    """
    csv_path = csv_path or _default_csv_path()
    errors: list[str] = []
    info: list[str] = []

    if not csv_path.exists():
        errors.append(f"File not found: {csv_path}")
        return False, errors, []

    try:
        df = pd.read_csv(csv_path, dtype={"song_id": str})
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors, []

    # Schema: required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        info.append(f"Unexpected columns: {extra}")

    # Row count
    if len(df) == 0:
        errors.append("Table is empty")
    else:
        info.append(f"Row count: {len(df)}")

    # Primary key: no duplicates
    if "song_id" in df.columns:
        dup = df["song_id"].duplicated()
        if dup.any():
            n = dup.sum()
            errors.append(f"Duplicate song_id: {n} rows")

    # Dtypes: song_id and genre, key can be object/string; rest numeric
    if (
        "song_id" in df.columns
        and not pd.api.types.is_string_dtype(df["song_id"])
        and not pd.api.types.is_object_dtype(df["song_id"])
    ):
        info.append("song_id is not string/object (may be numeric IDs)")
    for col in NUMERIC_FEATURE_COLUMNS + ["tempo_bpm"]:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")

    # NaN counts for DEAM columns
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            info.append(f"{col}: {n} NaN ({100 * n / len(df):.1f}%)")

    # Tempo: report coverage and out-of-range
    tempo_pct = 0.0
    if "tempo_bpm" in df.columns:
        valid_tempo = df["tempo_bpm"].dropna()
        n_valid = len(valid_tempo)
        tempo_pct = 100 * n_valid / len(df) if len(df) else 0
        info.append(f"tempo_bpm: {n_valid} non-NaN ({tempo_pct:.1f}%)")
        if n_valid > 0:
            out = ((valid_tempo < TEMPO_MIN) | (valid_tempo > TEMPO_MAX)).sum()
            if out > 0:
                info.append(
                    f"tempo_bpm: {out} values outside {TEMPO_MIN}-{TEMPO_MAX} BPM"
                )

    # Key: report coverage
    key_pct = 0.0
    if "key" in df.columns:
        known = (df["key"].astype(str).str.strip() != "") & (
            df["key"].astype(str).str.lower() != "unknown"
        )
        n_known = known.sum()
        key_pct = 100 * n_known / len(df) if len(df) else 0
        info.append(f"key: {n_known} non-unknown ({key_pct:.1f}%)")

    # Data quality: require some audio-derived data (enrich step must have run successfully)
    if len(df) > 0 and (
        tempo_pct < MIN_TEMPO_COVERAGE_PCT or key_pct < MIN_KEY_COVERAGE_PCT
    ):
        errors.append(
            "Audio-derived data missing or failed: tempo_bpm and/or key are all empty or unknown. "
            "Re-run: python3 scripts/run_enrich_pipeline.py (ensure data/audio/ has MP3s and librosa can load them)."
        )

    all_passed = len(errors) == 0
    return all_passed, errors, info


# DEAM labels: required columns
DEAM_LABELS_COLUMNS = ["song_id", "arousal", "valence"]


def _default_deam_labels_path() -> Path:
    try:
        from ..config.data_paths import DEAM_LABELS_PATH

        return DEAM_LABELS_PATH
    except ImportError:
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / "data" / "processed" / "deam_labels.csv"


def validate_deam_labels(
    csv_path: Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate deam_labels.csv. Returns (all_passed, errors, info_messages)."""
    csv_path = csv_path or _default_deam_labels_path()
    errors: list[str] = []
    info: list[str] = []

    if not csv_path.exists():
        errors.append(f"File not found: {csv_path}")
        return False, errors, []

    try:
        df = pd.read_csv(csv_path, dtype={"song_id": str})
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors, []

    missing = [c for c in DEAM_LABELS_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if len(df) == 0:
        errors.append("Table is empty")
    else:
        info.append(f"Row count: {len(df)}")
    if "song_id" in df.columns and df["song_id"].duplicated().any():
        errors.append(f"Duplicate song_id: {df['song_id'].duplicated().sum()} rows")
    for col in ("arousal", "valence"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
    return len(errors) == 0, errors, info


def _default_modeling_dataset_path() -> Path:
    try:
        from ..config.data_paths import MODELING_DATASET_PATH

        return MODELING_DATASET_PATH
    except ImportError:
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / "data" / "processed" / "modeling_dataset.csv"


def validate_modeling_dataset(
    csv_path: Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate modeling_dataset.csv (song_features + arousal, valence). Returns (all_passed, errors, info)."""
    csv_path = csv_path or _default_modeling_dataset_path()
    errors: list[str] = []
    info: list[str] = []

    if not csv_path.exists():
        errors.append(f"File not found: {csv_path}")
        return False, errors, []

    try:
        df = pd.read_csv(csv_path, dtype={"song_id": str})
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors, []

    required = REQUIRED_COLUMNS + ["arousal", "valence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if len(df) == 0:
        errors.append("Table is empty")
    else:
        info.append(f"Row count: {len(df)}")
    if "song_id" in df.columns and df["song_id"].duplicated().any():
        errors.append(f"Duplicate song_id: {df['song_id'].duplicated().sum()} rows")
    for col in ("arousal", "valence"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
    return len(errors) == 0, errors, info


EMOTION_PREDICTIONS_REQUIRED = ["song_id", "predicted_arousal", "predicted_valence"]


def _default_emotion_predictions_path() -> Path:
    try:
        from ..config.data_paths import EMOTION_PREDICTIONS_PATH

        return EMOTION_PREDICTIONS_PATH
    except ImportError:
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / "data" / "processed" / "emotion_predictions.csv"


def validate_emotion_predictions(
    csv_path: Path | None = None,
    song_features_path: Path | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate emotion_predictions.csv. Optionally check row count matches song_features."""
    csv_path = csv_path or _default_emotion_predictions_path()
    errors: list[str] = []
    info: list[str] = []

    if not csv_path.exists():
        errors.append(f"File not found: {csv_path}")
        return False, errors, []

    try:
        df = pd.read_csv(csv_path, dtype={"song_id": str})
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors, []

    missing = [c for c in EMOTION_PREDICTIONS_REQUIRED if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if len(df) == 0:
        errors.append("Table is empty")
    else:
        info.append(f"Row count: {len(df)}")
    if "song_id" in df.columns and df["song_id"].duplicated().any():
        errors.append(f"Duplicate song_id: {df['song_id'].duplicated().sum()} rows")
    for col in ("predicted_arousal", "predicted_valence"):
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
            elif df[col].isna().any():
                errors.append(f"Column '{col}' has {df[col].isna().sum()} NaN values")
    if song_features_path and song_features_path.exists() and "song_id" in df.columns:
        try:
            sf = pd.read_csv(song_features_path, dtype={"song_id": str})
            if len(sf) != len(df):
                errors.append(
                    f"Row count {len(df)} does not match song_features.csv ({len(sf)})"
                )
        except Exception:
            pass
    return len(errors) == 0, errors, info


def main() -> int:
    """Run validation for song_features, then deam_labels and modeling_dataset if present. Return 0 if all passed, 1 otherwise."""
    all_passed = True
    sf_path = _default_csv_path()
    dl_path = _default_deam_labels_path()
    md_path = _default_modeling_dataset_path()

    # 1) song_features (required for pipeline)
    print("--- song_features.csv ---")
    passed, errors, info = validate_song_features(sf_path)
    if errors:
        for m in errors:
            print("  Error:", m)
        all_passed = False
    if info:
        for m in info:
            print("  ", m)
    print("song_features: PASSED" if passed else "song_features: FAILED")

    # 2) deam_labels (Phase 2)
    if dl_path.exists():
        print("\n--- deam_labels.csv ---")
        passed, errors, info = validate_deam_labels(dl_path)
        if errors:
            for m in errors:
                print("  Error:", m)
            all_passed = False
        if info:
            for m in info:
                print("  ", m)
        print("deam_labels: PASSED" if passed else "deam_labels: FAILED")

    # 3) modeling_dataset (Phase 2)
    if md_path.exists():
        print("\n--- modeling_dataset.csv ---")
        passed, errors, info = validate_modeling_dataset(md_path)
        if errors:
            for m in errors:
                print("  Error:", m)
            all_passed = False
        if info:
            for m in info:
                print("  ", m)
        print("modeling_dataset: PASSED" if passed else "modeling_dataset: FAILED")

    # 4) emotion_predictions (Phase 2)
    ep_path = _default_emotion_predictions_path()
    if ep_path.exists():
        print("\n--- emotion_predictions.csv ---")
        passed, errors, info = validate_emotion_predictions(
            ep_path, song_features_path=sf_path
        )
        if errors:
            for m in errors:
                print("  Error:", m)
            all_passed = False
        if info:
            for m in info:
                print("  ", m)
        print(
            "emotion_predictions: PASSED" if passed else "emotion_predictions: FAILED"
        )

    print("\nOverall:", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
