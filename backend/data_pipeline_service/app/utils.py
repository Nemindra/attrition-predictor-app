import os
from fastapi import HTTPException
import pandas as pd

def validate_file_extension(filename: str, allowed_extensions: list):
    """Validate the file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File extension '{ext}' is not allowed. Allowed extensions: {allowed_extensions}")

def read_csv_file(file) -> pd.DataFrame:
    """Read and parse CSV file."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

def validate_dataframe_columns(df: pd.DataFrame, required_columns: list):
    """Validate if all required columns are present in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
