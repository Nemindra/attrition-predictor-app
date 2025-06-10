import pandas as pd
import os
from datetime import datetime
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from kfp import Client
from mlflow import log_param, log_metric, start_run
from typing import Dict, Any

from .model_storage import save_model
from .model_comparison import compare_models

REQUIRED_COLUMNS = [
    "employee_id", "department", "designation", "monthly_salary", "attrition",
    "num_prev_companies", "years_in_company", "age", "education_level", 
    "marital_status", "gender", "work_location"
]

def retrain_model(file_path: str, db: Session):
    """Retrain the model using new data and save metrics."""
    
    # Load Data
    try:
        data = pd.read_csv(file_path)
        if not set(REQUIRED_COLUMNS).issubset(data.columns):
            raise Exception("Missing required columns in dataset")

        X = data.drop("attrition", axis=1)
        y = data["attrition"]

    except Exception as e:
        return {"error": str(e)}

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_pred)
    }

    # Save model
    run_id = save_model(model, "attrition_model")
    metrics["model_version"] = run_id

    # Save metrics to DB
    db.execute(
        """
        INSERT INTO model_matrix (
            model_version, training_date, accuracy_score, precision_score, 
            recall_score, f1_score, roc_auc_score
        ) VALUES (
            :model_version, :training_date, :accuracy_score, :precision_score, 
            :recall_score, :f1_score, :roc_auc_score
        )
        """,
        {
            "model_version": run_id,
            "training_date": datetime.now().date(),
            **metrics
        }
    )
    db.commit()

    # Compare models after retraining
    comparison = compare_models(db)

    return {
        "message": "Model retrained successfully",
        "metrics": metrics,
        "comparison": comparison
    }

def trigger_kubeflow_pipeline(
    pipeline_id: str,
    params: Dict[str, Any],
    host: str = "http://localhost:8080"
) -> Dict[str, Any]:
    """
    Trigger a Kubeflow pipeline run with the given parameters.
    Returns run info dict (run_id, status, etc.).
    Raises Exception on failure.
    """
    try:
        client = Client(host=host)
        run = client.run_pipeline(
            experiment_id=client.create_experiment("AttritionTraining").id,
            job_name="attrition-training-run",
            pipeline_id=pipeline_id,
            params=params
        )
        return {"run_id": run.id, "status": run.status}
    except Exception as e:
        raise RuntimeError(f"Failed to trigger Kubeflow pipeline: {e}")

def log_pipeline_run(run_id: str, params: Dict[str, Any], metrics: Dict[str, Any] = None):
    """
    Log pipeline run parameters and metrics to MLflow.
    """
    with start_run(run_name=f"kubeflow_run_{run_id}"):
        for k, v in params.items():
            log_param(k, v)
        if metrics:
            for k, v in metrics.items():
                log_metric(k, v)

def retrain_model_with_kubeflow(
    data_location: str,
    reporting_date: str,
    hyperparameters: Dict[str, Any],
    pipeline_id: str,
    kfp_host: str = "http://localhost:8080"
) -> Dict[str, Any]:
    """
    Triggers Kubeflow pipeline for retraining and logs run to MLflow.
    Returns pipeline run info.
    """
    params = {"data_location": data_location, "reporting_date": reporting_date, **hyperparameters}
    run_info = trigger_kubeflow_pipeline(pipeline_id, params, host=kfp_host)
    log_pipeline_run(run_info["run_id"], params)
    return run_info

# FastAPI endpoint (to be imported in main.py)
from fastapi import APIRouter, HTTPException
router = APIRouter()

@router.post("/trigger-training", summary="Trigger Kubeflow pipeline for model retraining")
def trigger_training_endpoint(
    data_location: str,
    reporting_date: str,
    hyperparameters: Dict[str, Any],
    pipeline_id: str,
    kfp_host: str = "http://localhost:8080"
):
    """
    Trigger a Kubeflow pipeline run for model retraining.
    - data_location: Path/URI to training data
    - reporting_date: Reporting date for the data
    - hyperparameters: Model hyperparameters
    - pipeline_id: Kubeflow pipeline ID
    - kfp_host: Kubeflow Pipelines API host
    Returns pipeline run info (run_id, status).
    """
    try:
        run_info = retrain_model_with_kubeflow(
            data_location, reporting_date, hyperparameters, pipeline_id, kfp_host
        )
        return run_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
