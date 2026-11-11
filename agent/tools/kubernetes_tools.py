from __future__ import annotations

import json
import subprocess
from typing import Any

NAMESPACE = "mlops-agent"
DEPLOYMENT = "mlops-agent-api"
SERVICE = "mlops-agent-api"
LABEL_SELECTOR = "app=mlops-agent-api"


def _run_kubectl(*args: str) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", "-n", NAMESPACE, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or result.stdout.strip()}
    return {"ok": True, "output": result.stdout.strip()}


def _get_json(*args: str) -> dict[str, Any]:
    response = _run_kubectl(*args, "-o", "json")
    if not response["ok"]:
        return response
    return {"ok": True, "data": json.loads(response["output"])}


def list_kubernetes_resources() -> dict[str, Any]:
    """Return real deployment, pod, and service status."""
    deployment = _get_json("get", "deployment", DEPLOYMENT)
    pods = _get_json("get", "pods", "-l", LABEL_SELECTOR)
    service = _get_json("get", "service", SERVICE)

    if not all(item["ok"] for item in [deployment, pods, service]):
        return {"ok": False, "message": "Could not read Kubernetes resources."}

    deployment_data = deployment["data"]
    pod_items = pods["data"].get("items", [])
    service_data = service["data"]

    return {
        "ok": True,
        "namespace": NAMESPACE,
        "deployment": {
            "name": DEPLOYMENT,
            "desired_replicas": deployment_data["spec"].get("replicas", 0),
            "available_replicas": deployment_data["status"].get("availableReplicas", 0),
            "ready_replicas": deployment_data["status"].get("readyReplicas", 0),
        },
        "pods": [
            {
                "name": pod["metadata"]["name"],
                "phase": pod["status"].get("phase"),
                "ready": all(
                    status.get("ready", False)
                    for status in pod["status"].get("containerStatuses", [])
                ),
            }
            for pod in pod_items
        ],
        "service": {
            "name": SERVICE,
            "type": service_data["spec"].get("type"),
            "cluster_ip": service_data["spec"].get("clusterIP"),
            "ports": service_data["spec"].get("ports", []),
        },
    }


def get_service_endpoints() -> dict[str, Any]:
    """Show which pod IPs currently back the ClusterIP service."""
    response = _get_json("get", "endpoints", SERVICE)
    if not response["ok"]:
        return response

    subsets = response["data"].get("subsets", [])
    addresses = [
        address.get("ip")
        for subset in subsets
        for address in subset.get("addresses", [])
    ]
    return {
        "ok": True,
        "service": SERVICE,
        "namespace": NAMESPACE,
        "ready_endpoint_ips": addresses,
    }


def get_pod_logs(tail_lines: int = 50) -> dict[str, Any]:
    """Return recent logs from one real API pod."""
    pods = _get_json("get", "pods", "-l", LABEL_SELECTOR)
    if not pods["ok"]:
        return pods

    items = pods["data"].get("items", [])
    if not items:
        return {"ok": False, "message": "No API pods found."}

    pod_name = items[0]["metadata"]["name"]
    response = _run_kubectl("logs", pod_name, f"--tail={tail_lines}")
    return {
        "ok": response["ok"],
        "pod": pod_name,
        "logs": response.get("output", ""),
        "error": response.get("error"),
    }


def scale_inference_api(replicas: int, confirmed: bool = False) -> dict[str, Any]:
    """Scale only after explicit user confirmation."""
    if not confirmed:
        return {
            "ok": False,
            "action": "confirmation_required",
            "message": (
                f"This will scale {DEPLOYMENT} to {replicas} replicas. "
                "Call again with confirmed=True only after user approval."
            ),
        }

    if not isinstance(replicas, int) or not 1 <= replicas <= 5:
        return {
            "ok": False,
            "message": "replicas must be an integer between 1 and 5.",
        }

    response = _run_kubectl("scale", "deployment", DEPLOYMENT, f"--replicas={replicas}")
    return {
        "ok": response["ok"],
        "action": "scaled" if response["ok"] else "failed",
        "replicas": replicas,
        "message": response.get("output") or response.get("error"),
    }
