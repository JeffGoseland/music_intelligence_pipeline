"""
Validate data/processed/song_features.csv against the data dictionary schema.

Checks: required columns, dtypes, row count, duplicate song_id, NaN counts,
optional tempo range and key coverage. Exits 0 if all required checks pass, 1 otherwise.
"""

from pathlib import Path
import sys

import pandas as pd

# Expected schema (docs/DATA_DICTIONARY.md)
REQUIRED_COLUMNS = [
    "song_id",
    "spectral_centroid",
    "energy",
    "mfcc_mean",
    "chroma_variance",
    "spectral_rolloff50",
    "zcr",
    "spectral_flux",
    "spectral_variance",
    "spectral_entropy",
    "spectral_harmonicity",
    "tempo_bpm",
    "genre",
    "key",
]
NUMERIC_FEATURE_COLUMNS = REQUIRED_COLUMNS[1:11]  # 10 DEAM columns
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


def validate_song_features(csv_path: Path | None = None) -> tuple[bool, list[str], list[str]]:
    """
    Validate song_features.csv. Returns (all_passed, errors, info_messages).
    """
    csv_path = csv_path or _default_csv_path()
    errors: list[str] = []
    info: list[str] = []

    if not csv_path.exists():
        errors.append(f"File not found: {csv_path}")
        return False, errors

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors

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
    if "song_id" in df.columns and not pd.api.types.is_string_dtype(df["song_id"]) and not pd.api.types.is_object_dtype(df["song_id"]):
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
                info.append(f"tempo_bpm: {out} values outside {TEMPO_MIN}-{TEMPO_MAX} BPM")

    # Key: report coverage
    key_pct = 0.0
    if "key" in df.columns:
        known = (df["key"].astype(str).str.strip() != "") & (df["key"].astype(str).str.lower() != "unknown")
        n_known = known.sum()
        key_pct = 100 * n_known / len(df) if len(df) else 0
        info.append(f"key: {n_known} non-unknown ({key_pct:.1f}%)")

    # Data quality: require some audio-derived data (enrich step must have run successfully)
    if len(df) > 0 and (tempo_pct < MIN_TEMPO_COVERAGE_PCT or key_pct < MIN_KEY_COVERAGE_PCT):
        errors.append(
            "Audio-derived data missing or failed: tempo_bpm and/or key are all empty or unknown. "
            "Re-run: python scripts/run_enrich_pipeline.py (ensure data/audio/ has MP3s and librosa can load them)."
        )

    all_passed = len(errors) == 0
    return all_passed, errors, info


def main() -> int:
    """Run validation and print results. Return 0 if passed, 1 otherwise."""
    passed, errors, info = validate_song_features()
    if errors:
        print("Errors:")
        for m in errors:
            print("  ", m)
    if info:
        print("Info:")
        for m in info:
            print("  ", m)
    if passed:
        print("Validation: PASSED")
    else:
        print("Validation: FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
