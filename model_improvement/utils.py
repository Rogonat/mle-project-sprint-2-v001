from __future__ import annotations

import json
import os
import time
from importlib import metadata
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model_improvement" / "artifacts"
ID_COLS = ["id", "building_id"]
TARGET_COL = "price"
BOOL_COLS = ["is_apartment", "studio", "has_elevator"]
CATEGORICAL_COLS = ["building_type_int"]


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(ID_COLS + BOOL_COLS + CATEGORICAL_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"В данных отсутствуют колонки: {sorted(missing)}")
    df = df.drop_duplicates(subset=["id"]).copy()
    for col in BOOL_COLS:
        df[col] = df[col].astype(int)
    numeric_cols = [col for col in df.columns if col not in BOOL_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isna().any().any():
        raise ValueError("После преобразования типов появились пропуски; проверь исходные данные.")
    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def regression_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def timed_fit_predict(model, X_train, y_train, X_test):
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - started
    return predictions, fit_seconds, predict_seconds


def log_environment() -> None:
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        mlflow.log_artifact(str(requirements), artifact_path="environment")
    installed = "\n".join(
        f"{package.metadata['Name']}=={package.version}"
        for package in sorted(metadata.distributions(), key=lambda item: item.metadata.get("Name", "").lower())
        if package.metadata.get("Name") and package.version
    )
    mlflow.log_text(installed, "environment/pip_freeze.txt")
    mlflow.log_text(json.dumps(dict(python=os.sys.version), indent=2), "environment/runtime.json")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
