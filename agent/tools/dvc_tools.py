from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DvcToolError(RuntimeError):
    """Raised when a safe DVC or Git command fails."""


def _run(command: list[str], timeout: int = 20) -> str:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise DvcToolError(message or f"Command failed: {' '.join(command)}")

    return result.stdout.strip()


def list_datasets() -> dict[str, Any]:
    """List datasets tracked by DVC and their content hashes."""
    datasets = []

    for metadata_path in sorted((PROJECT_ROOT / "data").glob("*.dvc")):
        metadata = yaml.safe_load(metadata_path.read_text())
        output = metadata["outs"][0]

        datasets.append(
            {
                "path": output["path"],
                "dvc_hash": output.get("md5"),
                "size_bytes": output.get("size"),
            }
        )

    return {"datasets": datasets}


def get_dataset_provenance() -> dict[str, Any]:
    """Return the current dataset, Git revision, DVC hash, and remote."""
    metadata_path = PROJECT_ROOT / "data" / "breast_cancer.csv.dvc"
    metadata = yaml.safe_load(metadata_path.read_text())
    output = metadata["outs"][0]

    tags = _run(["git", "tag", "--points-at", "HEAD"]).splitlines()

    return {
        "dataset_path": output["path"],
        "dvc_hash": output.get("md5"),
        "size_bytes": output.get("size"),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_tags": tags,
        "dvc_remote": _run(["dvc", "remote", "list"]),
    }


def compare_dataset_versions(
    from_revision: str,
    to_revision: str,
) -> dict[str, Any]:
    """Compare two Git/DVC revisions without changing project state."""
    _run(["git", "rev-parse", "--verify", f"{from_revision}^{{commit}}"])
    _run(["git", "rev-parse", "--verify", f"{to_revision}^{{commit}}"])

    diff = _run(["dvc", "diff", from_revision, to_revision])

    return {
        "from_revision": from_revision,
        "to_revision": to_revision,
        "dvc_diff": diff or "No tracked dataset changes found.",
    }


def detect_data_changes() -> dict[str, Any]:
    """Check whether the working dataset differs from DVC tracking."""
    status = _run(["dvc", "status"])

    return {
        "has_changes": status != "Data and pipelines are up to date.",
        "dvc_status": status,
    }


def reproduction_guidance(revision: str) -> dict[str, str]:
    """Provide safe, reproducible commands without checking out automatically."""
    commit = _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"])

    return {
        "revision": revision,
        "resolved_commit": commit,
        "commands": (
            f"git checkout {revision}\n"
            "dvc pull\n"
            "python training/train.py"
        ),
        "note": "Checkout changes project state, so the agent must ask first.",
    }
