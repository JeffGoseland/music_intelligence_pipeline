#!/usr/bin/env python3
"""
Run the full pipeline from start to finish (before committing or moving on).

Order:
  1. Enrich: DEAM features + audio → song_features.csv
  2. Phase 2 labels + join: DEAM annotations → deam_labels.csv; join → modeling_dataset.csv
  3. Train: RandomForest, Ridge, ElasticNet, XGBoost (CV tuning) → models/*.joblib
  4. Predict: XGBoost models + song_features → emotion_predictions.csv
  5. Validate: schema, row counts, and data-quality checks on all outputs

Run from project root: python3 scripts/run_full_pipeline.py
Exit 0 if all steps and validation pass; 1 otherwise.
Writes run manifest to data/processed/pipeline_run.json (MLOps traceability).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    from src.config.data_paths import PIPELINE_RUN_PATH, SONG_FEATURES_PATH

    started_at = datetime.now(timezone.utc).isoformat()
    steps: list[dict] = []

    def step(name: str, status: str = "ok", detail: str | None = None) -> None:
        steps.append({"name": name, "status": status, **({"detail": detail} if detail else {})})

    try:
        print("=== 1. Enrich: song_features.csv ===", flush=True)
        from src.pipeline.enrich_song_features import run_enrich_pipeline
        df = run_enrich_pipeline()
        print(f"  Wrote {len(df)} rows to {SONG_FEATURES_PATH}\n", flush=True)
        step("enrich", "ok", f"{len(df)} rows")

        print("=== 2. Phase 2 labels + join: deam_labels.csv, modeling_dataset.csv ===", flush=True)
        from src.pipeline.deam_labels_loader import run_deam_labels_pipeline
        from src.pipeline.build_modeling_dataset import run_build_modeling_dataset
        run_deam_labels_pipeline()
        run_build_modeling_dataset()
        print("  Done.\n", flush=True)
        step("labels_and_join", "ok")

        print("=== 3. Train emotion models (RF, Ridge, ElasticNet, XGBoost) ===", flush=True)
        from src.modeling.train_emotion_models import format_all_metrics_table, train_all_models
        all_metrics = train_all_models()
        print(format_all_metrics_table(all_metrics))
        print("  Done.\n", flush=True)
        step("train", "ok")

        print("=== 4. Generate emotion_predictions.csv ===", flush=True)
        from src.pipeline.generate_emotion_predictions import run_emotion_predictions
        out = run_emotion_predictions()
        print(f"  Wrote {len(out)} rows to emotion_predictions.csv\n", flush=True)
        step("predict", "ok", f"{len(out)} rows")

        print("=== 5. Validate all outputs ===", flush=True)
        from src.pipeline.validate_song_features import main as validate_main
        exit_code = validate_main()
        step("validate", "ok" if exit_code == 0 else "fail")

        manifest = {
            "pipeline": "feature_engineering_and_model_generation",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "steps": steps,
            "overall": "ok" if exit_code == 0 else "fail",
        }
        _write_manifest(manifest, PIPELINE_RUN_PATH)
        print(f"  Run manifest: {PIPELINE_RUN_PATH}\n", flush=True)
        return exit_code
    except Exception as e:
        step("pipeline", "fail", str(e))
        _write_manifest(
            {
                "pipeline": "feature_engineering_and_model_generation",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "steps": steps,
                "overall": "fail",
                "error": str(e),
            },
            PIPELINE_RUN_PATH,
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
