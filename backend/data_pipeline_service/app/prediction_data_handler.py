from sqlalchemy.orm import Session
from fastapi import HTTPException
import pandas as pd
from io import BytesIO

def process_prediction_data(file, db: Session):
    try:
        df = pd.read_csv(BytesIO(file.read()))

        for _, row in df.iterrows():
            db.execute(
                """
                INSERT INTO predictions (
                    prediction_date, employee_id, predicted_risk, 
                    model_version, actual_attrition, prediction_accuracy
                ) VALUES (
                    NOW(), :employee_id, 'Pending', 'N/A', NULL, NULL
                )
                """,
                {
                    "employee_id": row["employee_id"]
                }
            )
        db.commit()
        return {"message": "Prediction data successfully uploaded."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
