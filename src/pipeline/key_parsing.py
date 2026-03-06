"""
Parse the pipeline 'key' string (e.g. 'C major', 'F# minor') into note, mode, and accidental.

Used in the enrich step to add key_note, key_mode, key_signature, is_major to song_features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

VALID_NOTES = frozenset("ABCDEFG")


def parse_key(key: str) -> tuple[str, str, str]:
    """
    Parse key string into (key_note, key_mode, key_signature).

    - key_note: letter A–G (or "Unknown" if unparseable).
    - key_mode: "major" or "minor".
    - key_signature: "sharp", "flat", or "natural".

    Examples:
        "C major"   -> ("C", "major", "natural")
        "F# minor"  -> ("F", "minor", "sharp")
        "Bb major"  -> ("B", "major", "flat")
        "unknown"   -> ("Unknown", "major", "natural")
    """
    if not key or not isinstance(key, str):
        return ("Unknown", "major", "natural")
    key = key.strip()
    if key.lower() == "unknown":
        return ("Unknown", "major", "natural")

    parts = key.split(maxsplit=1)
    root = (parts[0] if parts else "").strip()
    mode_part = (parts[1] if len(parts) > 1 else "").lower()
    key_mode = "minor" if "minor" in mode_part else "major"

    if not root:
        return ("Unknown", key_mode, "natural")

    if root.endswith("#"):
        key_signature = "sharp"
        key_note = root[:-1].strip().upper()
    elif root.endswith("b"):
        key_signature = "flat"
        key_note = root[:-1].strip().upper()
    else:
        key_signature = "natural"
        key_note = root.upper()

    if key_note not in VALID_NOTES:
        key_note = "Unknown"
    return (key_note, key_mode, key_signature)


def add_key_derived_columns(df: "pd.DataFrame", key_col: str = "key") -> "pd.DataFrame":
    """
    Add key_note, key_mode, key_signature, is_major to DataFrame in place (and return it).

    Requires key_col to exist. Uses parse_key() on each value.
    """
    if key_col not in df.columns:
        return df
    parsed = df[key_col].astype(str).map(lambda k: parse_key(k))
    df["key_note"] = [p[0] for p in parsed]
    df["key_mode"] = [p[1] for p in parsed]
    df["key_signature"] = [p[2] for p in parsed]
    df["is_major"] = df["key_mode"] == "major"
    return df
