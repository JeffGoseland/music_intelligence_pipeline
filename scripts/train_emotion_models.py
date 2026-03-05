#!/usr/bin/env python3
"""Train RandomForest emotion models (arousal and valence). Run from project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train_emotion_models import format_metrics_table, train_random_forest_models


def main() -> int:
    metrics = train_random_forest_models()
    print("Trained RandomForest models for arousal and valence.")
    print()
    print(format_metrics_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

