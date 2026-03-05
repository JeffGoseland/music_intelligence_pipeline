"""
Train RandomForest models for arousal and valence from modeling_dataset.csv.

Reads data/processed/modeling_dataset.csv and fits one RandomForestRegressor
for each target (arousal, valence). Optional CV-based hyperparameter tuning
(RandomizedSearchCV). Saves models under models/ and prints metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from ..config.data_paths import MODELING_DATASET_PATH, MODELS_DIR


FeatureName = Literal[
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
]


FEATURE_COLUMNS: Tuple[FeatureName, ...] = (
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
)

# RandomizedSearchCV param space for RandomForestRegressor
RF_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}


@dataclass
class ModelMetrics:
    rmse: float
    r2: float
    pearson_r: float
    n_train: int
    n_val: int
    model_path: Path
    cv_rmse_mean: float | None = None  # mean CV RMSE when tuning was used
    best_params: Dict[str, Any] | None = field(default_factory=dict)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_path: Path,
    n_train: int,
    n_val: int,
    cv_rmse_mean: float | None = None,
    best_params: Dict[str, Any] | None = None,
) -> ModelMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")
    return ModelMetrics(
        rmse=rmse,
        r2=r2,
        pearson_r=pearson_r,
        n_train=n_train,
        n_val=n_val,
        model_path=model_path,
        cv_rmse_mean=cv_rmse_mean,
        best_params=best_params or {},
    )


def _fit_single_target(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: Path,
    random_state: int,
    tune_hyperparams: bool,
    cv: int,
    n_iter: int,
    param_distributions: Dict[str, Any],
) -> ModelMetrics:
    if tune_hyperparams:
        base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            refit=True,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        cv_rmse_mean = float(-search.best_score_)
        best_params = dict(search.best_params_)
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        cv_rmse_mean = None
        best_params = {}

    y_pred = model.predict(X_val)
    dump(model, model_path)
    return _compute_metrics(
        y_true=y_val,
        y_pred=y_pred,
        model_path=model_path,
        n_train=len(y_train),
        n_val=len(y_val),
        cv_rmse_mean=cv_rmse_mean,
        best_params=best_params if best_params else None,
    )


def train_random_forest_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 24,
    param_distributions: Dict[str, Any] | None = None,
) -> Dict[str, ModelMetrics]:
    """
    Train RandomForest models for arousal and valence.

    If tune_hyperparams=True (default), uses RandomizedSearchCV with cv folds
    to select hyperparameters; the best estimator is refit on the full training
    set and evaluated on the holdout. Otherwise fits a single default RF.

    Returns a dict with keys "arousal" and "valence" mapping to ModelMetrics.
    """
    dataset_path = dataset_path or MODELING_DATASET_PATH
    models_dir = models_dir or MODELS_DIR
    param_distributions = param_distributions or RF_PARAM_DISTRIBUTIONS

    df = pd.read_csv(dataset_path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in modeling dataset: {missing}")
    for target in ("arousal", "valence"):
        if target not in df.columns:
            raise ValueError(f"Missing target column '{target}' in modeling dataset")

    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_arousal = df["arousal"].to_numpy(dtype=float)
    y_valence = df["valence"].to_numpy(dtype=float)

    # Single split: same indices for both targets so train/val are aligned
    X_train, X_val, y_ar_train, y_ar_val = train_test_split(
        X,
        y_arousal,
        test_size=test_size,
        random_state=random_state,
    )
    _, _, y_val_train, y_val_val = train_test_split(
        X,
        y_valence,
        test_size=test_size,
        random_state=random_state,
    )

    models_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, ModelMetrics] = {}

    metrics["arousal"] = _fit_single_target(
        X_train,
        y_ar_train,
        X_val,
        y_ar_val,
        models_dir / "arousal_random_forest.joblib",
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
        param_distributions=param_distributions,
    )

    metrics["valence"] = _fit_single_target(
        X_train,
        y_val_train,
        X_val,
        y_val_val,
        models_dir / "valence_random_forest.joblib",
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
        param_distributions=param_distributions,
    )

    return metrics


def format_metrics_table(metrics: Dict[str, ModelMetrics]) -> str:
    """Return a small text table summarizing metrics (holdout + optional CV/tuning info)."""
    lines = [
        "target,rmse,r2,pearson_r,n_train,n_val,cv_rmse_mean,model_path",
    ]
    for target, m in metrics.items():
        cv_str = f"{m.cv_rmse_mean:.4f}" if m.cv_rmse_mean is not None else ""
        lines.append(
            f"{target},{m.rmse:.4f},{m.r2:.4f},{m.pearson_r:.4f},{m.n_train},{m.n_val},{cv_str},{m.model_path}"
        )
    # Append best_params as readable lines
    for target, m in metrics.items():
        if m.best_params:
            lines.append(f"  {target} best_params: {m.best_params}")
    return "\n".join(lines)

