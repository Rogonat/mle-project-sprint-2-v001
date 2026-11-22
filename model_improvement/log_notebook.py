from pathlib import Path
import os
import sys

import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from model_improvement.utils import log_environment  # noqa: E402


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "local-emulation.env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "real_estate_price_improvement")
    mlflow.set_experiment(experiment_name)
    notebook = PROJECT_ROOT / "model_improvement" / "model_improvement.ipynb"
    if not notebook.exists():
        raise FileNotFoundError(notebook)
    with mlflow.start_run(run_name="project_notebook_final") as run:
        mlflow.set_tags({"artifact": "executed_notebook"})
        mlflow.log_artifact(str(notebook), artifact_path="notebook")
        log_environment()
        print(f"Выполненный ноутбук залогирован. Run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
