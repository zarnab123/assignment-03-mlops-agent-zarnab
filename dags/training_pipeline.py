from __future__ import annotations

import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pendulum
from airflow.exceptions import AirflowException
from airflow.sdk import dag, task
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "MLOps_Agent_Experiments"
QUALITY_THRESHOLD = 0.90


@dag(
    dag_id="mlops_training_pipeline",
    description="Validated MLOps training pipeline with MLflow quality gate.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "assignment-03", "training"],
)
def mlops_training_pipeline():
    @task(retries=1, retry_delay=timedelta(seconds=10))
    def validate_dataset() -> dict[str, int]:
        """Check that the DVC-managed dataset is suitable for training."""
        if not DATA_PATH.exists():
            raise AirflowException(f"Dataset not found: {DATA_PATH}")

        data = pd.read_csv(DATA_PATH)
        required_columns = {"target", "dataset_version"}

        if not required_columns.issubset(data.columns):
            raise AirflowException("Dataset is missing required columns.")

        if len(data) < 300:
            raise AirflowException("Dataset quality check failed: fewer than 300 rows.")

        print(f"Validation passed: {len(data)} rows, {len(data.columns)} columns")
        return {"rows": len(data), "columns": len(data.columns)}

    @task()
    def preprocess_data(validation: dict[str, int]) -> dict[str, int]:
        """Inspect feature columns before training."""
        data = pd.read_csv(DATA_PATH)
        feature_count = len(
            [
                column
                for column in data.columns
                if column not in {"target", "dataset_version"}
            ]
        )

        if feature_count == 0:
            raise AirflowException("Preprocessing failed: no feature columns found.")

        print(f"Preprocessing completed: {feature_count} feature columns")
        return {**validation, "feature_count": feature_count}

    @task(retries=1, retry_delay=timedelta(seconds=10))
    def train_model(preprocessed: dict[str, int]) -> dict[str, str | int]:
        """Run real training and log the experiment to MLflow."""
        result = subprocess.run(
            [sys.executable, "training/train.py", "--c", "1.0"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        print(result.stdout)

        if result.returncode != 0:
            raise AirflowException(
                f"Training failed: {result.stderr or result.stdout}"
            )

        match = re.search(r"Run ID: ([a-z0-9]+)", result.stdout)
        if match is None:
            raise AirflowException("Training finished but did not return an MLflow Run ID.")

        return {**preprocessed, "mlflow_run_id": match.group(1)}

    @task()
    def evaluate_and_log(training: dict[str, str | int]) -> dict[str, float | str]:
        """Read the recorded MLflow metrics for the new training run."""
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        run = client.get_run(str(training["mlflow_run_id"]))
        metrics = run.data.metrics

        if "f1_score" not in metrics or "accuracy" not in metrics:
            raise AirflowException("Evaluation failed: MLflow metrics are missing.")

        print(
            f"MLflow evaluation: accuracy={metrics['accuracy']:.4f}, "
            f"f1_score={metrics['f1_score']:.4f}"
        )

        return {
            "mlflow_run_id": str(training["mlflow_run_id"]),
            "accuracy": float(metrics["accuracy"]),
            "f1_score": float(metrics["f1_score"]),
        }

    @task()
    def quality_gate(evaluation: dict[str, float | str]) -> dict[str, float | str]:
        """Prevent unsuitable models from being promoted."""
        f1_score = float(evaluation["f1_score"])

        if f1_score < QUALITY_THRESHOLD:
            raise AirflowException(
                f"Quality gate failed: F1 {f1_score:.4f} < {QUALITY_THRESHOLD:.2f}"
            )

        print(f"Quality gate passed: F1 {f1_score:.4f}")
        return evaluation

    validated = validate_dataset()
    preprocessed = preprocess_data(validated)
    trained = train_model(preprocessed)
    evaluated = evaluate_and_log(trained)
    quality_gate(evaluated)


mlops_training_pipeline()
