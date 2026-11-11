from __future__ import annotations

import re
from typing import Any

from agent.tools.airflow_tools import (
    get_latest_pipeline_status,
    list_dags,
    trigger_training_pipeline,
)
from agent.tools.dvc_tools import (
    compare_dataset_versions,
    get_dataset_provenance,
    list_datasets,
)
from agent.tools.kubernetes_tools import (
    get_pod_logs,
    get_service_endpoints,
    list_kubernetes_resources,
    scale_inference_api,
)
from agent.tools.mlflow_tools import (
    get_best_run,
    get_model_registry_info,
    list_recent_runs,
)


def handle_request(message: str, confirmed: bool = False) -> dict[str, Any]:
    """
    Route a natural-language MLOps request to a real tool.
    Mutating actions require confirmed=True.
    """
    text = message.lower().strip()

    if any(word in text for word in ["scale", "replica", "replicas"]):
        match = re.search(r"\b([1-5])\b", text)
        replicas = int(match.group(1)) if match else 3
        return {
            "agent": "MLOps Orchestrator",
            "intent": "scale_kubernetes_deployment",
            "result": scale_inference_api(replicas, confirmed=confirmed),
        }

    if any(word in text for word in ["kubernetes", "k8s", "pod", "deployment", "service", "endpoint"]):
        if "endpoint" in text:
            result = get_service_endpoints()
            intent = "get_kubernetes_service_endpoints"
        elif "log" in text:
            result = get_pod_logs()
            intent = "get_kubernetes_pod_logs"
        else:
            result = list_kubernetes_resources()
            intent = "get_kubernetes_status"
        return {"agent": "MLOps Orchestrator", "intent": intent, "result": result}

    if any(word in text for word in ["airflow", "dag", "pipeline", "trigger"]):
        if "trigger" in text or "run pipeline" in text:
            result = trigger_training_pipeline(confirmed=confirmed)
            intent = "trigger_training_pipeline"
        elif "list" in text and "dag" in text:
            result = list_dags()
            intent = "list_airflow_dags"
        else:
            result = get_latest_pipeline_status()
            intent = "get_airflow_pipeline_status"
        return {"agent": "MLOps Orchestrator", "intent": intent, "result": result}

    if any(word in text for word in ["mlflow", "experiment", "run", "model", "registry", "best"]):
        if "best" in text:
            result = get_best_run()
            intent = "get_best_mlflow_run"
        elif "registry" in text or "registered" in text:
            result = get_model_registry_info()
            intent = "get_model_registry_info"
        else:
            result = list_recent_runs()
            intent = "list_mlflow_runs"
        return {"agent": "MLOps Orchestrator", "intent": intent, "result": result}

    if any(word in text for word in ["dvc", "dataset", "data", "provenance", "version"]):
        if "compare" in text or "difference" in text:
            result = compare_dataset_versions()
            intent = "compare_dataset_versions"
        elif "provenance" in text or "history" in text:
            result = get_dataset_provenance()
            intent = "get_dataset_provenance"
        else:
            result = list_datasets()
            intent = "list_datasets"
        return {"agent": "MLOps Orchestrator", "intent": intent, "result": result}

    return {
        "agent": "MLOps Orchestrator",
        "intent": "help",
        "result": {
            "message": (
                "I can inspect DVC datasets, MLflow runs, Airflow pipelines, "
                "or Kubernetes workloads. Mutating actions need confirmation."
            )
        },
    }
