"""
Phase 1: Load DEAM feature CSVs, aggregate per song, write Song Feature Table.

Reads from data/deam_csvs/features/ (one CSV per song, ';' separator).
Outputs data/processed/song_features.csv with columns:
  song_id, tempo, spectral_centroid, energy, mfcc_mean, chroma_variance.

DEAM column mapping (per-song mean over time unless noted):
  - tempo: not in DEAM OpenSMILE features → NaN
  - spectral_centroid: pcm_fftMag_spectralCentroid_sma_amean
  - energy: pcm_RMSenergy_sma_amean
  - mfcc_mean: pcm_fftMag_mfcc_sma[1]_amean
  - chroma_variance: variance across 26 audSpec_Rfilt_sma[i]_amean band means (proxy for spectral spread)
"""

from pathlib import Path
import pandas as pd

# DEAM CSV column names (OpenSMILE / Geneva minimal feature set)
COL_SPECTRAL_CENTROID = "pcm_fftMag_spectralCentroid_sma_amean"
COL_ENERGY = "pcm_RMSenergy_sma_amean"
COL_MFCC = "pcm_fftMag_mfcc_sma[1]_amean"
COL_RFILT_PREFIX = "audSpec_Rfilt_sma["
COL_RFILT_SUFFIX = "_amean"


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


def aggregate_one_song(csv_path: Path) -> dict | None:
    """
    Read one DEAM feature CSV and aggregate to one row (our schema).
    Returns dict with song_id, tempo, spectral_centroid, energy, mfcc_mean, chroma_variance.
    """
    song_id = csv_path.stem
    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception:
        return None

    if df.empty:
        return {
            "song_id": song_id,
            "tempo": float("nan"),
            "spectral_centroid": float("nan"),
            "energy": float("nan"),
            "mfcc_mean": float("nan"),
            "chroma_variance": float("nan"),
        }

    row = {"song_id": song_id}

    # Tempo: DEAM doesn't provide it in this feature set
    row["tempo"] = float("nan")

    # Spectral centroid (mean over frames)
    if COL_SPECTRAL_CENTROID in df.columns:
        row["spectral_centroid"] = df[COL_SPECTRAL_CENTROID].mean()
    else:
        row["spectral_centroid"] = float("nan")

    # Energy (mean over frames)
    if COL_ENERGY in df.columns:
        row["energy"] = df[COL_ENERGY].mean()
    else:
        row["energy"] = float("nan")

    # MFCC summary (mean of first coefficient over frames)
    if COL_MFCC in df.columns:
        row["mfcc_mean"] = df[COL_MFCC].mean()
    else:
        row["mfcc_mean"] = float("nan")

    # Chroma variance proxy: variance across Rfilt band means (per song)
    rfilt_cols = _rfilt_amean_columns(list(df.columns))
    if rfilt_cols:
        band_means = df[rfilt_cols].mean()
        row["chroma_variance"] = float(band_means.var())
    else:
        row["chroma_variance"] = float("nan")

    return row


def run_feature_pipeline(
    features_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load all DEAM feature CSVs in features_dir, aggregate per song, write Song Feature Table.

    Returns the feature DataFrame. Creates output_path parent dir if needed.
    """
    from ..config.paths import FEATURES_CSV_DIR, SONG_FEATURES_PATH

    features_dir = features_dir or FEATURES_CSV_DIR
    output_path = output_path or SONG_FEATURES_PATH

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    csv_files = sorted(features_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {features_dir}")

    rows = []
    for path in csv_files:
        r = aggregate_one_song(path)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    df = df[
        ["song_id", "tempo", "spectral_centroid", "energy", "mfcc_mean", "chroma_variance"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    import sys
    from ..config.paths import SONG_FEATURES_PATH
    out_path = SONG_FEATURES_PATH
    df = run_feature_pipeline(output_path=out_path)
    print(f"Wrote {len(df)} rows to {out_path}", file=sys.stderr)
    print(df.head())
