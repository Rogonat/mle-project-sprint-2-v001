#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_improvement.utils import (  # noqa: E402
    load_data,
    log_environment,
    regression_metrics,
    save_json,
    split_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Регистрация baseline-модели в MLflow")
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "data/initial_data.csv"))
    parser.add_argument("--model", default=os.getenv("BASELINE_MODEL_PATH", "baseline/model.pkl"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "local-emulation.env")
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "real_estate_price_improvement")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "real_estate_price_model")
    experiment = mlflow.set_experiment(experiment_name)

    df = load_data(PROJECT_ROOT / args.data)
    X_train, X_test, y_train, y_test = split_data(
        df, random_state=int(os.getenv("RANDOM_STATE", "42"))
    )
    baseline = joblib.load(PROJECT_ROOT / args.model)
    predictions = baseline.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    metrics_path = PROJECT_ROOT / "mlflow_server" / "baseline_metrics.json"
    save_json(metrics, metrics_path)
    input_example = X_train.head(5)
    signature = infer_signature(input_example, baseline.predict(input_example))

    with mlflow.start_run(run_name="stage1_baseline") as run:
        mlflow.set_tags({"stage": "1", "model_type": "baseline", "data_split": "test_0.2_seed_42"})
        mlflow.log_param("n_estimators", baseline.named_steps["model"].n_estimators)
        mlflow.log_param("max_depth", baseline.named_steps["model"].max_depth)
        mlflow.log_param("random_state", baseline.named_steps["model"].random_state)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(PROJECT_ROOT / "mlflow_server" / "run_mlflow.sh"), artifact_path="code")
        mlflow.log_artifact(str(Path(__file__)), artifact_path="code")
        log_environment()
        mlflow.sklearn.log_model(
            sk_model=baseline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
        print(f"Experiment: {experiment_name}, id={experiment.experiment_id}")
        print(f"Run: stage1_baseline, id={run.info.run_id}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Модель зарегистрирована: {model_name}")


if __name__ == "__main__":
    main()
