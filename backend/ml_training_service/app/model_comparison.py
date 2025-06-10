from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from .config import config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

router = APIRouter()

# Set up DB session factory using config
engine = create_engine(config.DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def fetch_model_metrics(db: Session) -> List[Dict[str, Any]]:
    """
    Query the model_matrix table for all model versions and their metrics.
    Returns a list of dicts with metrics for each model version.
    """
    try:
        result = db.execute(text("""
            SELECT model_version, training_date, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            FROM model_matrix
            ORDER BY training_date DESC
        """)).fetchall()
        metrics = [
            {
                "model_version": row[0],
                "training_date": str(row[1]),
                "accuracy": row[2],
                "precision": row[3],
                "recall": row[4],
                "f1": row[5],
                "roc_auc": row[6]
            }
            for row in result
        ]
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

def compare_model_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare the latest model to previous models and identify improvements.
    Returns a dict with best model, latest model, and improvement flags.
    """
    if not metrics:
        return {"message": "No model metrics available."}
    latest = metrics[0]
    best = max(metrics, key=lambda m: m["f1"])
    improvement = latest["f1"] >= best["f1"]
    return {
        "latest_model": latest,
        "best_model": best,
        "latest_is_best": improvement
    }

@router.get("/model-metrics", summary="Get metrics and comparison for all trained models")
def get_model_metrics_endpoint(db: Session = Depends(get_db)):
    """
    Retrieve metrics for all trained models and compare latest vs best.
    Returns a list of model metrics and a comparison summary.
    """
    metrics = fetch_model_metrics(db)
    comparison = compare_model_metrics(metrics)
    return {"models": metrics, "comparison": comparison}

def compare_models(db: Session):
    """Compare all models in the model_matrix table based on F1 score."""
    models = db.execute("""
        SELECT model_version, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
        FROM model_matrix
        ORDER BY f1_score DESC
    """).fetchall()

    if not models:
        return {"message": "No models available for comparison"}

    best_model = models[0]

    comparison = {
        "best_model": {
            "version": best_model[0],
            "accuracy": best_model[1],
            "precision": best_model[2],
            "recall": best_model[3],
            "f1_score": best_model[4],
            "roc_auc_score": best_model[5]
        },
        "all_models": [
            {
                "version": model[0],
                "accuracy": model[1],
                "precision": model[2],
                "recall": model[3],
                "f1_score": model[4],
                "roc_auc_score": model[5]
            } for model in models
        ]
    }

    return comparison
