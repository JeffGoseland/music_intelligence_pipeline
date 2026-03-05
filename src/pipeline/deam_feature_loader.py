"""
Phase 1: Load DEAM feature CSVs, aggregate per song, write Song Feature Table.

Reads from data/deam_csvs/features/ (one CSV per song, ';' separator).
Supports minimal (5 + tempo placeholder) or rich (10 DEAM core features) schema.

Rich schema (10 DEAM features, per-song mean over time):
  spectral_centroid, energy, mfcc_mean, chroma_variance, spectral_rolloff50,
  zcr, spectral_flux, spectral_variance, spectral_entropy, spectral_harmonicity.
Tempo, genre, key are added in the enrich step from audio.
"""

from pathlib import Path
import pandas as pd

# DEAM CSV column names (OpenSMILE / Geneva minimal feature set)
COL_SPECTRAL_CENTROID = "pcm_fftMag_spectralCentroid_sma_amean"
COL_ENERGY = "pcm_RMSenergy_sma_amean"
COL_MFCC = "pcm_fftMag_mfcc_sma[1]_amean"
COL_RFILT_PREFIX = "audSpec_Rfilt_sma["
COL_RFILT_SUFFIX = "_amean"
# Extra columns for rich (10-feature) schema
COL_SPECTRAL_ROLLOFF50 = "pcm_fftMag_spectralRollOff50.0_sma_amean"
COL_ZCR = "pcm_zcr_sma_amean"
COL_SPECTRAL_FLUX = "pcm_fftMag_spectralFlux_sma_amean"
COL_SPECTRAL_VARIANCE = "pcm_fftMag_spectralVariance_sma_amean"
COL_SPECTRAL_ENTROPY = "pcm_fftMag_spectralEntropy_sma_amean"
COL_SPECTRAL_HARMONICITY = "pcm_fftMag_spectralHarmonicity_sma_amean"

# Column order for rich schema (10 DEAM features only)
RICH_DEAM_COLUMNS = [
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
]


def _rfilt_amean_columns(columns: list[str]) -> list[str]:
    """Column names for audSpec_Rfilt_sma[i]_amean, i in 0..25."""
    out = []
    for c in columns:
        if c.startswith(COL_RFILT_PREFIX) and c.endswith(COL_RFILT_SUFFIX):
            try:
                idx = c[len(COL_RFILT_PREFIX) : c.rindex("]")]
                if idx.isdigit() and 0 <= int(idx) <= 25:
                    out.append(c)
            except (ValueError, TypeError):
                pass
    return sorted(out, key=lambda x: int(x.split("[")[1].split("]")[0]))


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col in df.columns:
        return float(df[col].mean())
    return float("nan")


def aggregate_one_song(csv_path: Path, rich: bool = False) -> dict | None:
    """
    Read one DEAM feature CSV and aggregate to one row.

    If rich=False: song_id, tempo (NaN), spectral_centroid, energy, mfcc_mean, chroma_variance.
    If rich=True: song_id + 10 DEAM features (no tempo; added in enrich step).
    """
    song_id = csv_path.stem
    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception:
        return None

    if df.empty:
        row = {"song_id": song_id}
        for c in RICH_DEAM_COLUMNS:
            row[c] = float("nan")
        if not rich:
            row["tempo"] = float("nan")
        return row

    row = {"song_id": song_id}

    row["spectral_centroid"] = _safe_mean(df, COL_SPECTRAL_CENTROID)
    row["energy"] = _safe_mean(df, COL_ENERGY)
    row["mfcc_mean"] = _safe_mean(df, COL_MFCC)

    rfilt_cols = _rfilt_amean_columns(list(df.columns))
    if rfilt_cols:
        row["chroma_variance"] = float(df[rfilt_cols].mean().var())
    else:
        row["chroma_variance"] = float("nan")

    if rich:
        row["spectral_rolloff50"] = _safe_mean(df, COL_SPECTRAL_ROLLOFF50)
        row["zcr"] = _safe_mean(df, COL_ZCR)
        row["spectral_flux"] = _safe_mean(df, COL_SPECTRAL_FLUX)
        row["spectral_variance"] = _safe_mean(df, COL_SPECTRAL_VARIANCE)
        row["spectral_entropy"] = _safe_mean(df, COL_SPECTRAL_ENTROPY)
        row["spectral_harmonicity"] = _safe_mean(df, COL_SPECTRAL_HARMONICITY)
    else:
        row["tempo"] = float("nan")

    return row


def run_feature_pipeline(
    features_dir: Path | None = None,
    output_path: Path | None = None,
    rich: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """
    Load all DEAM feature CSVs in features_dir, aggregate per song, optionally write CSV.

    rich=False: song_id, tempo (NaN), spectral_centroid, energy, mfcc_mean, chroma_variance.
    rich=True: song_id + 10 DEAM core features only (for use with enrich step).
    write=False: return DataFrame without writing to disk.
    """
    from ..config.data_paths import FEATURES_CSV_DIR, SONG_FEATURES_PATH

    features_dir = features_dir or FEATURES_CSV_DIR
    output_path = output_path or SONG_FEATURES_PATH

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    csv_files = sorted(features_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {features_dir}")

    rows = []
    for path in csv_files:
        r = aggregate_one_song(path, rich=rich)
        if r is not None:
            rows.append(r)

    if rich:
        df = pd.DataFrame(rows)[["song_id"] + RICH_DEAM_COLUMNS]
    else:
        df = pd.DataFrame(rows)[
            ["song_id", "tempo", "spectral_centroid", "energy", "mfcc_mean", "chroma_variance"]
        ]

    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    import sys
    from ..config.data_paths import SONG_FEATURES_PATH
    out_path = SONG_FEATURES_PATH
    df = run_feature_pipeline(output_path=out_path)
    print(f"Wrote {len(df)} rows to {out_path}", file=sys.stderr)
    print(df.head())
