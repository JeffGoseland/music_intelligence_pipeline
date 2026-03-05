#!/usr/bin/env python3
"""Train all emotion models (RandomForest, Ridge, ElasticNet, XGBoost) with CV tuning. Run from project root."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.data_paths import MODELS_DIR
from src.modeling.train_emotion_models import format_all_metrics_table, train_all_models


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=PROJECT_ROOT,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    print("Training RandomForest, Ridge, ElasticNet, and XGBoost (arousal + valence each)...")
    all_metrics, run_id, versioned_dir = train_all_models()
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
