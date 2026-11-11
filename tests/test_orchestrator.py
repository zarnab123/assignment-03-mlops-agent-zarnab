from unittest.mock import patch

from agent.orchestrator import handle_request


@patch("agent.orchestrator.list_datasets")
def test_routes_dataset_request(mock_tool):
    mock_tool.return_value = {"datasets": ["breast_cancer.csv"]}
    result = handle_request("Show my DVC datasets")
    assert result["intent"] == "list_datasets"
    assert result["result"]["datasets"] == ["breast_cancer.csv"]


@patch("agent.orchestrator.get_dataset_provenance")
def test_routes_provenance_request(mock_tool):
    mock_tool.return_value = {"version": "v2"}
    result = handle_request("Show dataset provenance")
    assert result["intent"] == "get_dataset_provenance"
    assert result["result"]["version"] == "v2"


@patch("agent.orchestrator.get_best_run")
def test_routes_best_mlflow_run(mock_tool):
    mock_tool.return_value = {"best_run": "run-123"}
    result = handle_request("Which MLflow run is best?")
    assert result["intent"] == "get_best_mlflow_run"
    assert result["result"]["best_run"] == "run-123"


@patch("agent.orchestrator.get_model_registry_info")
def test_routes_registry_request(mock_tool):
    mock_tool.return_value = {"model_name": "BreastCancerClassifier"}
    result = handle_request("Show model registry")
    assert result["intent"] == "get_model_registry_info"
    assert result["result"]["model_name"] == "BreastCancerClassifier"


@patch("agent.orchestrator.get_latest_pipeline_status")
def test_routes_airflow_status(mock_tool):
    mock_tool.return_value = {"state": "success"}
    result = handle_request("What is the latest Airflow pipeline status?")
    assert result["intent"] == "get_airflow_pipeline_status"
    assert result["result"]["state"] == "success"


@patch("agent.orchestrator.list_kubernetes_resources")
def test_routes_kubernetes_status(mock_tool):
    mock_tool.return_value = {"available_replicas": 2}
    result = handle_request("Show Kubernetes deployment status")
    assert result["intent"] == "get_kubernetes_status"
    assert result["result"]["available_replicas"] == 2


@patch("agent.orchestrator.get_service_endpoints")
def test_routes_kubernetes_endpoints(mock_tool):
    mock_tool.return_value = {"ready_endpoint_ips": ["10.1.0.6"]}
    result = handle_request("Show Kubernetes service endpoints")
    assert result["intent"] == "get_kubernetes_service_endpoints"
    assert result["result"]["ready_endpoint_ips"] == ["10.1.0.6"]


def test_scale_requires_confirmation():
    result = handle_request("Scale Kubernetes deployment to 3 replicas")
    assert result["intent"] == "scale_kubernetes_deployment"
    assert result["result"]["action"] == "confirmation_required"


@patch("agent.orchestrator.scale_inference_api")
def test_confirmed_scale_is_forwarded(mock_tool):
    mock_tool.return_value = {"ok": True, "action": "scaled", "replicas": 3}
    result = handle_request(
        "Scale Kubernetes deployment to 3 replicas",
        confirmed=True,
    )
    mock_tool.assert_called_once_with(3, confirmed=True)
    assert result["result"]["action"] == "scaled"
