from __future__ import annotations

import os
import re
from typing import Any

from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = "MLOps_Agent_Experiments"


class MlflowToolError(RuntimeError):
    """Raised when MLflow is unavailable or a request is invalid."""


def _client() -> MlflowClient:
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        client.search_experiments(max_results=1)
        return client
    except Exception as error:
        raise MlflowToolError(
            f"MLflow tracking server is unavailable: {error}"
        ) from error


def _run_summary(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "parameters": dict(run.data.params),
        "tags": {
            key: value
            for key, value in run.data.tags.items()
            if key in {"dataset_version", "dvc_hash", "git_commit"}
        },
    }


def list_experiments() -> dict[str, Any]:
    """List the experiments available on the real MLflow server."""
    experiments = _client().search_experiments()

    return {
        "experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "lifecycle_stage": experiment.lifecycle_stage,
            }
            for experiment in experiments
        ]
    }


def list_recent_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    max_runs: int = 5,
) -> dict[str, Any]:
    """Retrieve recent recorded runs with metrics, parameters, and provenance."""
    client = _client()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise MlflowToolError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )

    return {
        "experiment_name": experiment_name,
        "runs": [_run_summary(run) for run in runs],
    }


def get_best_run(
    metric_name: str = "f1_score",
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    """Select the best recorded run by a requested metric."""
    recorded_runs = list_recent_runs(experiment_name, max_runs=100)["runs"]
    eligible_runs = [
        run for run in recorded_runs if metric_name in run["metrics"]
    ]

    if not eligible_runs:
        raise MlflowToolError(f"No runs contain metric: {metric_name}")

    best_run = max(eligible_runs, key=lambda run: run["metrics"][metric_name])

    return {
        "selection_metric": metric_name,
        "best_run": best_run,
        "selection_note": (
            "This is the best recorded run for the requested metric only. "
            "It is a candidate, not automatically production-ready."
        ),
    }


def list_run_artifacts(run_id: str) -> dict[str, Any]:
    """List artifacts recorded for one real MLflow run."""
    artifacts = _client().list_artifacts(run_id)

    return {
        "run_id": run_id,
        "artifacts": [
            {
                "path": artifact.path,
                "is_directory": artifact.is_dir,
                "size_bytes": artifact.file_size,
            }
            for artifact in artifacts
        ],
    }


def get_model_registry_info(
    model_name: str = "BreastCancerClassifier",
) -> dict[str, Any]:
    """Retrieve model versions and aliases from MLflow Model Registry."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name):
        raise MlflowToolError("Invalid registered model name.")

    client = _client()
    registered_model = client.get_registered_model(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    return {
        "model_name": registered_model.name,
        "aliases": dict(registered_model.aliases),
        "versions": [
            {
                "version": version.version,
                "current_stage": version.current_stage,
                "run_id": version.run_id,
                "source": version.source,
            }
            for version in versions
        ],
    }
