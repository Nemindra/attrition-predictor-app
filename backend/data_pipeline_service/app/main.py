from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from .database import get_db
from .data_ingestion import process_monthly_data
from .prediction_data_handler import process_prediction_data

app = FastAPI()

@app.post("/upload/monthly-data")
async def upload_monthly_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return process_monthly_data(file, db)

@app.post("/upload/predictor-data")
async def upload_predictor_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return process_prediction_data(file, db)
