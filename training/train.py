from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"
DVC_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv.dvc"


def get_dvc_hash() -> str:
    metadata = yaml.safe_load(DVC_PATH.read_text())
    return metadata["outs"][0]["md5"]


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()

    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=["target", "dataset_version"])
    target = data["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    model = LogisticRegression(
        C=args.c,
        max_iter=1000,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("MLOps_Agent_Experiments")

    with mlflow.start_run(run_name=f"LogisticRegression_C_{args.c}") as run:
        mlflow.log_params(
            {
                "algorithm": "LogisticRegression",
                "C": args.c,
                "max_iter": 1000,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        mlflow.set_tags(
            {
                "dataset_version": str(data["dataset_version"].iloc[0]),
                "dvc_hash": get_dvc_hash(),
                "git_commit": get_git_commit(),
            }
        )

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
        )

        mlflow.log_text(
            json.dumps(
                {
                    "dataset_version": data["dataset_version"].iloc[0],
                    "dvc_hash": get_dvc_hash(),
                    "git_commit": get_git_commit(),
                },
                indent=2,
            ),
            "provenance.json",
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
