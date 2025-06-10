"""
ML Training Service
------------------
This FastAPI app provides endpoints to retrain the employee attrition prediction model and to retrieve model metrics.
- POST /trigger-training: Retrain the model with new data (CSV upload).
- GET /model-metrics: Retrieve metrics and comparison for all trained models.

Environment variables (see config.py):
- DATABASE_URL: Database connection string
- MLFLOW_TRACKING_URI: MLflow tracking server URI
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
from .config import DATABASE_URL
from .retrain import retrain_model
from .model_comparison import compare_models

# Database session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for DB session injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="ML Training Service",
    description="Retrain attrition prediction model and serve model metrics.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.post("/trigger-training", summary="Retrain the attrition model with new data")
def trigger_training(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Retrain the attrition prediction model using a new CSV dataset.
    Upload a CSV file with employee data (must include all required columns).
    Returns training metrics and model comparison.
    """
    # Save uploaded file to a temp location
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    result = retrain_model(tmp_path, db)
    os.remove(tmp_path)
    return result

@app.get("/model-metrics", summary="Get metrics and comparison for all trained models")
def get_model_metrics(db: Session = Depends(get_db)):
    """
    Retrieve metrics and comparison for all trained models.
    Returns best model and all model metrics.
    """
    return compare_models(db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)