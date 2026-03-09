"""
Train emotion models (RandomForest, Ridge, ElasticNet, XGBoost) for arousal and valence.

Reads data/processed/modeling_dataset.csv. Each model type is trained with
5-fold RandomizedSearchCV for hyperparameter tuning. Saves models under models/
and returns metrics for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config.data_paths import MODELING_DATASET_PATH, MODELS_DIR

try:
    from xgboost import XGBRegressor  # type: ignore[import-untyped]
except ImportError:
    XGBRegressor = None  # type: ignore[misc, assignment]


FEATURE_COLUMNS: Tuple[str, ...] = (
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
)

# RandomizedSearchCV param spaces (expanded for broader search)
RF_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_estimators": [100, 200, 300, 500, 700, 1000],
    "max_depth": [5, 10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 8],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap": [True, False],
    "min_impurity_decrease": [0.0, 1e-5, 1e-4, 1e-3],
}

RIDGE_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "ridge__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    "ridge__solver": ["auto", "svd", "cholesky", "lsqr"],
}

ELASTICNET_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "elasticnet__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    "elasticnet__l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "elasticnet__max_iter": [1000, 2000, 5000],
}

XGB_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [2, 3, 4, 5, 6, 7, 9],
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "gamma": [0.0, 0.01, 0.1, 0.5, 1.0],
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
    """Compute RMSE, R², Pearson r and wrap in ModelMetrics for one target."""
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


def _fit_single_target_generic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: Path,
    base_estimator: Any,
    random_state: int,
    tune_hyperparams: bool,
    cv: int,
    n_iter: int,
    param_distributions: Dict[str, Any] | None,
) -> ModelMetrics:
    """Fit one model (with optional RandomizedSearchCV), save to model_path, return metrics."""
    if tune_hyperparams and param_distributions:
        search = RandomizedSearchCV(
            base_estimator,
            param_distributions=param_distributions,
            n_iter=min(n_iter, _n_combinations(param_distributions)),
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
        import sklearn.base

        model = sklearn.base.clone(base_estimator)
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


def _n_combinations(param_distributions: Dict[str, Any]) -> int:
    """Return approximate number of parameter combinations for n_iter capping."""
    n = 1
    for v in param_distributions.values():
        n *= len(v) if hasattr(v, "__len__") else 10
    return n


def train_random_forest_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 64,
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

    X_train, X_val, y_ar_train, y_ar_val, y_val_train, y_val_val = _load_and_split(
        dataset_path, models_dir, test_size, random_state
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, ModelMetrics] = {}

    base_rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    metrics["arousal"] = _fit_single_target_generic(
        X_train,
        y_ar_train,
        X_val,
        y_ar_val,
        models_dir / "arousal_random_forest.joblib",
        base_rf,
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
        param_distributions=param_distributions,
    )
    metrics["valence"] = _fit_single_target_generic(
        X_train,
        y_val_train,
        X_val,
        y_val_val,
        models_dir / "valence_random_forest.joblib",
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
        param_distributions=param_distributions,
    )

    return metrics


def train_ridge_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 48,
    param_distributions: Dict[str, Any] | None = None,
) -> Dict[str, ModelMetrics]:
    """Train Ridge (L2) regression with StandardScaler; one model per target."""
    dataset_path = dataset_path or MODELING_DATASET_PATH
    models_dir = models_dir or MODELS_DIR
    param_distributions = param_distributions or RIDGE_PARAM_DISTRIBUTIONS
    X_train, X_val, y_ar_train, y_ar_val, y_val_train, y_val_val = _load_and_split(
        dataset_path, models_dir, test_size, random_state
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=random_state)),
        ]
    )
    return {
        "arousal": _fit_single_target_generic(
            X_train,
            y_ar_train,
            X_val,
            y_ar_val,
            models_dir / "arousal_ridge.joblib",
            base,
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
        "valence": _fit_single_target_generic(
            X_train,
            y_val_train,
            X_val,
            y_val_val,
            models_dir / "valence_ridge.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(random_state=random_state)),
                ]
            ),
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
    }


def train_elasticnet_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 64,
    param_distributions: Dict[str, Any] | None = None,
) -> Dict[str, ModelMetrics]:
    """Train ElasticNet regression with StandardScaler; one model per target."""
    dataset_path = dataset_path or MODELING_DATASET_PATH
    models_dir = models_dir or MODELS_DIR
    param_distributions = param_distributions or ELASTICNET_PARAM_DISTRIBUTIONS
    X_train, X_val, y_ar_train, y_ar_val, y_val_train, y_val_val = _load_and_split(
        dataset_path, models_dir, test_size, random_state
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(random_state=random_state)),
        ]
    )
    return {
        "arousal": _fit_single_target_generic(
            X_train,
            y_ar_train,
            X_val,
            y_ar_val,
            models_dir / "arousal_elasticnet.joblib",
            base,
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
        "valence": _fit_single_target_generic(
            X_train,
            y_val_train,
            X_val,
            y_val_val,
            models_dir / "valence_elasticnet.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("elasticnet", ElasticNet(random_state=random_state)),
                ]
            ),
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
    }


def train_xgboost_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 72,
    param_distributions: Dict[str, Any] | None = None,
) -> Dict[str, ModelMetrics]:
    """Train XGBoost regressors; one model per target. Requires xgboost."""
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is required for train_xgboost_models. Install with: pip install xgboost"
        )
    dataset_path = dataset_path or MODELING_DATASET_PATH
    models_dir = models_dir or MODELS_DIR
    param_distributions = param_distributions or XGB_PARAM_DISTRIBUTIONS
    X_train, X_val, y_ar_train, y_ar_val, y_val_train, y_val_val = _load_and_split(
        dataset_path, models_dir, test_size, random_state
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    base = XGBRegressor(random_state=random_state, n_jobs=-1)
    return {
        "arousal": _fit_single_target_generic(
            X_train,
            y_ar_train,
            X_val,
            y_ar_val,
            models_dir / "arousal_xgboost.joblib",
            base,
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
        "valence": _fit_single_target_generic(
            X_train,
            y_val_train,
            X_val,
            y_val_val,
            models_dir / "valence_xgboost.joblib",
            XGBRegressor(random_state=random_state, n_jobs=-1),
            random_state,
            tune_hyperparams,
            cv,
            n_iter,
            param_distributions,
        ),
    }


def _load_and_split(
    dataset_path: Path,
    models_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load modeling_dataset, drop rows with NaN in features, return train/val arrays (same split for both targets)."""
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
    # Impute feature NaNs with column median so all-NaN columns (e.g. tempo_bpm) don't drop all rows
    for j in range(X.shape[1]):
        col = X[:, j].copy()
        if np.isnan(col).any():
            med = np.nanmedian(col)
            if np.isnan(med):
                med = 0.0  # all-NaN column: use 0
            col = np.where(np.isnan(col), med, col)
            X[:, j] = col
    valid = ~np.isnan(y_arousal) & ~np.isnan(y_valence)
    if not valid.all():
        X = X[valid]
        y_arousal = y_arousal[valid]
        y_valence = y_valence[valid]
    # Same X, test_size, random_state so train/val indices match for both targets (fair comparison).
    X_train, X_val, y_ar_train, y_ar_val = train_test_split(
        X, y_arousal, test_size=test_size, random_state=random_state
    )
    _, _, y_val_train, y_val_val = train_test_split(
        X, y_valence, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_ar_train, y_ar_val, y_val_train, y_val_val


# Fast path: fewer RandomizedSearchCV iterations for quick iteration (e.g. dev/testing).
N_ITER_FAST = 5


def train_all_models(
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    run_id: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparams: bool = True,
    cv: int = 5,
    n_iter: int = 64,
    fast: bool = False,
    include_xgboost: bool = True,
) -> Tuple[Dict[str, Dict[str, ModelMetrics]], str, Path]:
    """
    Train RandomForest, Ridge, ElasticNet, and (optionally) XGBoost with CV tuning.
    Saves models under models/<run_id>/ to avoid silent overwrites.
    If fast=True, uses N_ITER_FAST (e.g. 5) iterations per model instead of n_iter.
    Returns (all_metrics, run_id, versioned_models_dir).
    """
    if fast:
        n_iter = N_ITER_FAST
    base_dir = models_dir or MODELS_DIR
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_dir = base_dir / run_id
    versioned_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Dict[str, ModelMetrics]] = {}
    out["random_forest"] = train_random_forest_models(
        dataset_path=dataset_path,
        models_dir=versioned_dir,
        test_size=test_size,
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
    )
    out["ridge"] = train_ridge_models(
        dataset_path=dataset_path,
        models_dir=versioned_dir,
        test_size=test_size,
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
    )
    out["elasticnet"] = train_elasticnet_models(
        dataset_path=dataset_path,
        models_dir=versioned_dir,
        test_size=test_size,
        random_state=random_state,
        tune_hyperparams=tune_hyperparams,
        cv=cv,
        n_iter=n_iter,
    )
    if include_xgboost and XGBRegressor is not None:
        out["xgboost"] = train_xgboost_models(
            dataset_path=dataset_path,
            models_dir=versioned_dir,
            test_size=test_size,
            random_state=random_state,
            tune_hyperparams=tune_hyperparams,
            cv=cv,
            n_iter=n_iter,
        )
    return out, run_id, versioned_dir


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


def format_all_metrics_table(all_metrics: Dict[str, Dict[str, ModelMetrics]]) -> str:
    """Format metrics for all model types (RandomForest, Ridge, ElasticNet, XGBoost)."""
    sections = []
    for model_name, metrics in all_metrics.items():
        sections.append(f"--- {model_name} ---")
        sections.append(format_metrics_table(metrics))
        sections.append("")
    return "\n".join(sections).strip()
