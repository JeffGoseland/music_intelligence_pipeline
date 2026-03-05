"""
Phase 2: Generate emotion_predictions.csv using trained XGBoost models.

Loads song_features.csv and the production models (arousal_xgboost.joblib,
valence_xgboost.joblib), predicts arousal and valence for every song, writes
data/processed/emotion_predictions.csv (song_id, predicted_arousal, predicted_valence).
"""

from pathlib import Path

import pandas as pd
from joblib import load

from ..config.data_paths import (
    EMOTION_PREDICTIONS_PATH,
    MODELS_DIR,
    SONG_FEATURES_PATH,
)
from ..modeling.train_emotion_models import FEATURE_COLUMNS


def run_emotion_predictions(
    song_features_path: Path | None = None,
    models_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load song_features, run XGBoost arousal/valence models, write emotion_predictions.csv.

    Uses the same 11 feature columns as training. Missing feature values (e.g. NaN tempo)
    are left as-is; XGBoost handles them. Creates output_path parent dir if needed.
    """
    song_features_path = song_features_path or SONG_FEATURES_PATH
    models_dir = models_dir or MODELS_DIR
    output_path = output_path or EMOTION_PREDICTIONS_PATH

    df = pd.read_csv(song_features_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in song_features: {missing}")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)

    arousal_path = models_dir / "arousal_xgboost.joblib"
    valence_path = models_dir / "valence_xgboost.joblib"
    if not arousal_path.exists() or not valence_path.exists():
        raise FileNotFoundError(
            f"XGBoost models not found. Run: python3 scripts/train_emotion_models.py "
            f"(expects {arousal_path}, {valence_path})"
        )

    arousal_model = load(arousal_path)
    valence_model = load(valence_path)

    pred_arousal = arousal_model.predict(X)
    pred_valence = valence_model.predict(X)

    out = pd.DataFrame({
        "song_id": df["song_id"].astype(str),
        "predicted_arousal": pred_arousal,
        "predicted_valence": pred_valence,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out
