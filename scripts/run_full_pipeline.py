#!/usr/bin/env python3
"""
Run the full pipeline from start to finish (before committing or moving on).

Order:
  1. Enrich: DEAM features + audio → song_features.csv
  2. Phase 2 labels + join: DEAM annotations → deam_labels.csv; join → modeling_dataset.csv
  3. Train: RandomForest, Ridge, ElasticNet, XGBoost (CV tuning) → models/<run_id>/*.joblib
  4. Predict: XGBoost models + song_features → emotion_predictions.csv
  5. Validate: schema, row counts, and data-quality checks on all outputs

Run from project root: python3 scripts/run_full_pipeline.py [--force]
  --force  Run all steps even if a step's checkpoint exists (resume is disabled).
Exit 0 if all steps and validation pass; 1 otherwise.
Writes run manifest to data/processed/pipeline_run.json (MLOps traceability).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=PROJECT_ROOT,
        ).strip()
    except Exception:
        return "unknown"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_step(name: str, fn, *args, **kwargs):
    """Run a step; on exception print clean message and exit 1."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"{name}: FAILED — {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    from src.config.data_paths import (
        PIPELINE_CHECKPOINT_DIR,
        PIPELINE_RUN_PATH,
        SONG_FEATURES_PATH,
    )

    parser = argparse.ArgumentParser(description="Run full feature + model pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run all steps even if checkpoint exists (no resume skip).",
    )
    args = parser.parse_args()
    force: bool = args.force

    def step_done(name: str) -> bool:
        return (PIPELINE_CHECKPOINT_DIR / f"{name}.done").exists()

    def mark_done(name: str) -> None:
        PIPELINE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (PIPELINE_CHECKPOINT_DIR / f"{name}.done").touch()

    started_at = datetime.now(timezone.utc).isoformat()
    steps: list[dict] = []

    def step(name: str, status: str = "ok", detail: str | None = None) -> None:
        steps.append(
            {"name": name, "status": status, **({"detail": detail} if detail else {})}
        )

    versioned_models_dir: Path | None = None

    try:
        # 1. Enrich
        print("=== 1. Enrich: song_features.csv ===", flush=True)
        if force or not step_done("enrich"):
            from src.pipeline.enrich_song_features import run_enrich_pipeline

            df = run_step("enrich", run_enrich_pipeline)
            print(f"  Wrote {len(df)} rows to {SONG_FEATURES_PATH}\n", flush=True)
            mark_done("enrich")
            step("enrich", "ok", f"{len(df)} rows")
        else:
            print("  Skipping (already done). Use --force to re-run.\n", flush=True)
            step("enrich", "skipped")

        # 2. Labels + join
        print(
            "=== 2. Phase 2 labels + join: deam_labels.csv, modeling_dataset.csv ===",
            flush=True,
        )
        if force or not step_done("labels_and_join"):
            from src.pipeline.deam_labels_loader import run_deam_labels_pipeline
            from src.pipeline.build_modeling_dataset import run_build_modeling_dataset

            run_step("labels_and_join (deam)", run_deam_labels_pipeline)
            run_step("labels_and_join (build)", run_build_modeling_dataset)
            print("  Done.\n", flush=True)
            mark_done("labels_and_join")
            step("labels_and_join", "ok")
        else:
            print("  Skipping (already done). Use --force to re-run.\n", flush=True)
            step("labels_and_join", "skipped")

        # 3. Train
        print(
            "=== 3. Train emotion models (RF, Ridge, ElasticNet, XGBoost) ===",
            flush=True,
        )
        if force or not step_done("train"):
            from src.modeling.train_emotion_models import (
                format_all_metrics_table,
                train_all_models,
            )

            all_metrics, run_id, versioned_models_dir = run_step(
                "train", train_all_models
            )
            table = format_all_metrics_table(all_metrics)
            print(table)
            versioned_models_dir.mkdir(parents=True, exist_ok=True)
            (versioned_models_dir / "training_metrics.txt").write_text(
                table, encoding="utf-8"
            )
            _run_info = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_hash": _get_git_hash(),
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5,
            }
            (versioned_models_dir / "run_info.json").write_text(
                json.dumps(_run_info, indent=2), encoding="utf-8"
            )
            print(
                f"  Models and metrics written to {versioned_models_dir}\n", flush=True
            )
            mark_done("train")
            step("train", "ok", str(versioned_models_dir))
        else:
            from src.config.data_paths import get_latest_models_dir

            versioned_models_dir = get_latest_models_dir()
            print(
                f"  Skipping (already done). Using models from {versioned_models_dir}\n",
                flush=True,
            )
            step("train", "skipped")

        # 4. Predict (use versioned dir from this run or latest)
        if versioned_models_dir is None:
            from src.config.data_paths import get_latest_models_dir

            versioned_models_dir = get_latest_models_dir()
        print("=== 4. Generate emotion_predictions.csv ===", flush=True)
        if force or not step_done("predict"):
            from src.pipeline.generate_emotion_predictions import (
                run_emotion_predictions,
            )

            out = run_step(
                "predict", run_emotion_predictions, models_dir=versioned_models_dir
            )
            print(f"  Wrote {len(out)} rows to emotion_predictions.csv\n", flush=True)
            mark_done("predict")
            step("predict", "ok", f"{len(out)} rows")
        else:
            print("  Skipping (already done). Use --force to re-run.\n", flush=True)
            step("predict", "skipped")

        # 5. Validate (always run)
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
    except SystemExit:
        raise
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
