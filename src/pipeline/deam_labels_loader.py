"""
Phase 2 Step 1: Load DEAM annotations and write per-song labels (arousal, valence).

Reads from data/deam_csvs/annotations/:
  - Prefer: song_level static CSVs (static_annotations_averaged_songs_*.csv)
  - Fallback: dynamic (per-second) arousal.csv / valence.csv, aggregated to mean per song.

Output: data/processed/deam_labels.csv with columns song_id, arousal, valence.
"""

from pathlib import Path
import pandas as pd


# Subpaths under ANNOTATIONS_DIR (DEAM layout)
SONG_LEVEL_SUBDIR = Path("annotations averaged per song") / "song_level"
DYNAMIC_SUBDIR = Path("annotations averaged per song") / "dynamic (per second annotations)"
STATIC_GLOB = "static_annotations_averaged_songs_*.csv"


def _find_song_id_column(df: pd.DataFrame) -> str | None:
    for cand in ("song_id", "song", "id", "songId"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "song" in c.lower() or c.lower() == "id":
            return c
    return None


def _find_arousal_column(df: pd.DataFrame) -> str | None:
    for cand in ("arousal", "AM", "mean_arousal", "arousal_mean"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "arousal" in c.lower():
            return c
    return None


def _find_valence_column(df: pd.DataFrame) -> str | None:
    for cand in ("valence", "VM", "mean_valence", "valence_mean"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "valence" in c.lower():
            return c
    return None


def _load_static_song_level(annotations_dir: Path) -> pd.DataFrame | None:
    song_level_dir = annotations_dir / SONG_LEVEL_SUBDIR
    if not song_level_dir.exists():
        return None
    files = sorted(song_level_dir.glob(STATIC_GLOB))
    if not files:
        return None
    dfs = []
    for f in files:
        # Some DEAM distributions use comma as separator here; start with ';'
        # then fall back to ',' if we clearly didn't split columns.
        try:
            df = pd.read_csv(f, sep=";", dtype=str)
        except Exception:
            df = pd.read_csv(f, dtype=str)
        if df.shape[1] == 1 and "," in str(df.columns[0]):
            df = pd.read_csv(f, sep=",", dtype=str)
        if df.empty:
            continue
        sid = _find_song_id_column(df)
        ar = _find_arousal_column(df)
        va = _find_valence_column(df)
        if sid and ar and va:
            out = df[[sid, ar, va]].copy()
            out.columns = ["song_id", "arousal", "valence"]
            out["song_id"] = out["song_id"].astype(str).str.strip()
            out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
            out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
            out = out.dropna(subset=["arousal", "valence"])
            dfs.append(out)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["song_id"], keep="first")
    return combined


def _load_dynamic_aggregated(annotations_dir: Path) -> pd.DataFrame | None:
    dynamic_dir = annotations_dir / DYNAMIC_SUBDIR
    arousal_path = dynamic_dir / "arousal.csv"
    valence_path = dynamic_dir / "valence.csv"
    if not arousal_path.exists() or not valence_path.exists():
        return None
    try:
        ar_df = pd.read_csv(arousal_path, sep=";")
    except Exception:
        ar_df = pd.read_csv(arousal_path)
    try:
        va_df = pd.read_csv(valence_path, sep=";")
    except Exception:
        va_df = pd.read_csv(valence_path)
    # Common layout: first column time or segment, rest are song IDs as column headers
    # Or: song_id in first column, then time steps as columns
    if ar_df.shape[1] < 2 or va_df.shape[1] < 2:
        return None
    # Assume columns are [time_or_id, song1, song2, ...] or [song_id, t1, t2, ...]
    first = ar_df.columns[0].lower()
    if "time" in first or "segment" in first or first in ("0", "1"):
        song_ids = [str(s).strip() for s in ar_df.columns[1:]]
        arousal_means = ar_df.iloc[:, 1:].mean(axis=0)
        # Align valence by same column names; fallback to same column order if names match
        if all(c in va_df.columns for c in ar_df.columns[1:]):
            valence_means = va_df[ar_df.columns[1:]].mean(axis=0)
        else:
            valence_means = va_df.iloc[:, 1:].mean(axis=0)
        out = pd.DataFrame({
            "song_id": song_ids,
            "arousal": arousal_means.values,
            "valence": valence_means.values,
        })
    else:
        # Rows = songs, first column = song_id
        ar_df = ar_df.set_index(ar_df.columns[0])
        va_df = va_df.set_index(va_df.columns[0])
        common = ar_df.index.intersection(va_df.index)
        out = pd.DataFrame({
            "song_id": [str(s).strip() for s in common],
            "arousal": ar_df.loc[common].mean(axis=1).values,
            "valence": va_df.loc[common].mean(axis=1).values,
        })
    return out.dropna(subset=["arousal", "valence"])


def run_deam_labels_pipeline(
    annotations_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load DEAM annotations (song_level static preferred, else dynamic aggregated),
    normalize to song_id, arousal, valence, and write CSV.

    Creates output_path parent dir if needed.
    """
    from ..config.data_paths import ANNOTATIONS_DIR, DEAM_LABELS_PATH

    annotations_dir = annotations_dir or ANNOTATIONS_DIR
    output_path = output_path or DEAM_LABELS_PATH

    labels = _load_static_song_level(annotations_dir)
    if labels is None or labels.empty:
        labels = _load_dynamic_aggregated(annotations_dir)
    if labels is None or labels.empty:
        raise FileNotFoundError(
            f"No DEAM annotation files found under {annotations_dir}. "
            "Expected song_level static_annotations_averaged_songs_*.csv or dynamic arousal.csv/valence.csv."
        )
    labels = labels[["song_id", "arousal", "valence"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)
    return labels
