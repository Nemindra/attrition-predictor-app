import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Any

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Save a trained model artifact to MLflow’s artifact store (e.g., S3)
def save_model(model: Any, model_name: str, artifact_path: str = "model") -> str:
    """
    Save the trained model to MLflow’s artifact store (S3, local, etc.).
    Returns the MLflow run ID for traceability.
    """
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path)
        mlflow.set_tag("model_name", model_name)
        return run.info.run_id

# Register a new model version in the MLflow Model Registry
def register_model_version(model_uri: str, model_name: str) -> str:
    """
    Register a new model version in the MLflow Model Registry.
    Returns the version number.
    """
    result = mlflow.register_model(model_uri, model_name)
    # Optionally transition to 'Staging' or 'Production' after validation
    # mlflow.client.MlflowClient().transition_model_version_stage(...)
    return result.version

# ---
# Version control and rollback strategies:
# - Each model version is tracked in the Model Registry with metadata and run info.
# - To rollback, transition a previous version to 'Production' using MLflow’s UI or API.
# - Artifacts are stored in S3 (or configured backend) and are immutable for auditability.
# ---
