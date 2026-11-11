from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIRFLOW_HOME = PROJECT_ROOT / "airflow"
DAGS_FOLDER = PROJECT_ROOT / "dags"
TRAINING_DAG_ID = "mlops_training_pipeline"


def _airflow_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AIRFLOW_HOME"] = str(AIRFLOW_HOME)
    env["AIRFLOW__CORE__DAGS_FOLDER"] = str(DAGS_FOLDER)
    env["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    return env


def _run_airflow(arguments: list[str]) -> Any:
    result = subprocess.run(
        ["airflow", *arguments],
        cwd=PROJECT_ROOT,
        env=_airflow_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        return {
            "ok": False,
            "error": result.stderr.strip() or result.stdout.strip(),
            "command": ["airflow", *arguments],
        }

    payload = result.stdout.strip()
    json_array_start = payload.find("[{")

    if json_array_start != -1:
        payload = payload[json_array_start:]

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"ok": True, "raw_output": result.stdout.strip()}

def list_dags() -> dict[str, Any]:
    """List real DAGs known by the local Airflow instance."""
    result = _run_airflow(["dags", "list", "--output", "json"])
    return {"tool": "list_dags", "result": result}


def list_dag_runs(dag_id: str = TRAINING_DAG_ID) -> dict[str, Any]:
    """Return real history for one DAG."""
    result = _run_airflow(["dags", "list-runs", dag_id, "--output", "json"])
    return {"tool": "list_dag_runs", "dag_id": dag_id, "result": result}


def get_latest_pipeline_status() -> dict[str, Any]:
    """Summarize the newest real training-pipeline run."""
    response = list_dag_runs(TRAINING_DAG_ID)
    runs = response["result"]

    if not isinstance(runs, list):
        return {
            "tool": "get_latest_pipeline_status",
            "ok": False,
            "message": "Airflow did not return readable DAG-run history.",
            "details": response,
        }

    if not runs:
        return {
            "tool": "get_latest_pipeline_status",
            "ok": True,
            "message": "No pipeline runs exist yet.",
        }

    latest_run = runs[0]
    return {
        "tool": "get_latest_pipeline_status",
        "ok": True,
        "dag_id": TRAINING_DAG_ID,
        "latest_run": latest_run,
        "run_count": len(runs),
    }


def trigger_training_pipeline(confirmed: bool = False) -> dict[str, Any]:
    """
    Trigger only the approved training DAG.
    The caller must explicitly confirm before this changes Airflow state.
    """
    if not confirmed:
        return {
            "tool": "trigger_training_pipeline",
            "ok": False,
            "action": "confirmation_required",
            "message": (
                "This will trigger mlops_training_pipeline. "
                "Call again with confirmed=True only after user approval."
            ),
        }

    result = _run_airflow(
        ["dags", "trigger", TRAINING_DAG_ID, "--output", "json"]
    )
    return {
        "tool": "trigger_training_pipeline",
        "confirmed": True,
        "result": result,
    }
