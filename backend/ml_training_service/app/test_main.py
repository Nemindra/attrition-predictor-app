"""
test_main.py
------------
Pytest-based tests for ML Training Service FastAPI app.
Covers /trigger-training and /model-metrics endpoints, mocks Kubeflow and MLflow, and checks edge cases.
"""
import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Ensure the parent directory is in sys.path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_kubeflow_and_mlflow():
    # Patch Kubeflow Client and MLflow for all tests
    with patch("retrain.Client") as mock_kfp, \
         patch("retrain.start_run"), \
         patch("retrain.log_param"), \
         patch("retrain.log_metric"):
        # Mock Kubeflow run_pipeline return
        mock_run = MagicMock()
        mock_run.id = "run-123"
        mock_run.status = "Succeeded"
        mock_kfp.return_value.run_pipeline.return_value = mock_run
        mock_kfp.return_value.create_experiment.return_value.id = "exp-1"
        yield

@patch("model_comparison.fetch_model_metrics")
def test_get_model_metrics_success(mock_fetch):
    """
    Test GET /model-metrics returns correct JSON structure and metrics.
    """
    mock_fetch.return_value = [
        {"model_version": "v1", "training_date": "2025-06-01", "accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.85},
        {"model_version": "v2", "training_date": "2025-06-02", "accuracy": 0.92, "precision": 0.85, "recall": 0.8, "f1": 0.82, "roc_auc": 0.88}
    ]
    response = client.get("/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert "comparison" in data
    assert data["models"][0]["model_version"] == "v1"

@patch("retrain.retrain_model_with_kubeflow")
def test_trigger_training_success(mock_retrain):
    """
    Test POST /trigger-training triggers pipeline and returns run info.
    """
    mock_retrain.return_value = {"run_id": "run-123", "status": "Succeeded"}
    payload = {
        "data_location": "s3://bucket/data.csv",
        "reporting_date": "2025-06-01",
        "hyperparameters": {"n_estimators": 100},
        "pipeline_id": "pipeline-abc"
    }
    response = client.post("/trigger-training", json=payload)
    assert response.status_code == 200
    assert response.json()["run_id"] == "run-123"
    assert response.json()["status"] == "Succeeded"

def test_trigger_training_invalid_params():
    """
    Test POST /trigger-training with missing params returns 422.
    """
    response = client.post("/trigger-training", json={"data_location": "s3://bucket/data.csv"})
    assert response.status_code == 422

@patch("retrain.retrain_model_with_kubeflow", side_effect=Exception("Pipeline failed"))
def test_trigger_training_pipeline_failure(mock_retrain):
    """
    Test POST /trigger-training handles pipeline failure and returns 500.
    """
    payload = {
        "data_location": "s3://bucket/data.csv",
        "reporting_date": "2025-06-01",
        "hyperparameters": {"n_estimators": 100},
        "pipeline_id": "pipeline-abc"
    }
    response = client.post("/trigger-training", json=payload)
    assert response.status_code == 500
    assert "Pipeline failed" in response.text
