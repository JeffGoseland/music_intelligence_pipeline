#!/usr/bin/env python3
"""Train all emotion models (RandomForest, Ridge, ElasticNet, XGBoost) with CV tuning. Run from project root. Use --fast for 5 iterations per model (quick iteration)."""

import argparse
import json
import subprocess
from datetime import datetime, timezone

import _bootstrap  # noqa: E402, F401 (side effect: sys.path)

from src.config.data_paths import MODELS_DIR
from src.modeling.train_emotion_models import format_all_metrics_table, train_all_models


def _get_git_hash() -> str:
    """Return short git HEAD hash for run_info, or 'unknown' if not a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=_bootstrap.PROJECT_ROOT,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    """Parse --fast, train all models, write metrics and run_info to versioned dir; exit 0."""
    parser = argparse.ArgumentParser(
        description="Train emotion models (RF, Ridge, ElasticNet, XGBoost)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast path: 5 RandomizedSearchCV iterations per model instead of 48–72.",
    )
    args = parser.parse_args()

    print(
        "Training RandomForest, Ridge, ElasticNet, and XGBoost (arousal + valence each)..."
    )
    if args.fast:
        print("(Fast path: 5 iterations per model)")
    all_metrics, run_id, versioned_dir = train_all_models(fast=args.fast)
    print()
    table = format_all_metrics_table(all_metrics)
    print(table)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = versioned_dir / "training_metrics.txt"
    metrics_path.write_text(table, encoding="utf-8")
    print(f"\nMetrics saved to {metrics_path}")

    run_info = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5,
    }
    run_info_path = versioned_dir / "run_info.json"
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Run info saved to {run_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
