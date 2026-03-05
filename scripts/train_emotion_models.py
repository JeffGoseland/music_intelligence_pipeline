#!/usr/bin/env python3
"""Train all emotion models (RandomForest, Ridge, ElasticNet, XGBoost) with CV tuning. Run from project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train_emotion_models import format_all_metrics_table, train_all_models


def main() -> int:
    print("Training RandomForest, Ridge, ElasticNet, and XGBoost (arousal + valence each)...")
    all_metrics = train_all_models()
    print()
    print(format_all_metrics_table(all_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

