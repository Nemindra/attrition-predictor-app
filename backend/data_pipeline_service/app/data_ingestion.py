from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from .utils import validate_file_extension, read_csv_file, validate_dataframe_columns

REQUIRED_COLUMNS = [
    "reporting_date", "employee_id", "department", "designation",
    "monthly_salary", "attrition", "num_prev_companies", 
    "years_in_company", "age", "education_level", 
    "marital_status", "gender", "work_location"
]

def process_monthly_data(file: UploadFile, db: Session):
    validate_file_extension(file.filename, [".csv"])

    # Read and validate CSV
    df = read_csv_file(file.file)
    validate_dataframe_columns(df, REQUIRED_COLUMNS)

    # Insert data into the database
    try:
        for _, row in df.iterrows():
            db.execute(
                """
                INSERT INTO employee_data (
                    reporting_date, employee_id, department, designation, 
                    monthly_salary, attrition, num_prev_companies, 
                    years_in_company, age, education_level, marital_status, 
                    gender, work_location
                ) VALUES (
                    :reporting_date, :employee_id, :department, :designation, 
                    :monthly_salary, :attrition, :num_prev_companies, 
                    :years_in_company, :age, :education_level, :marital_status, 
                    :gender, :work_location
                )
                """,
                {
                    "reporting_date": row["reporting_date"],
                    "employee_id": row["employee_id"],
                    "department": row["department"],
                    "designation": row["designation"],
                    "monthly_salary": row["monthly_salary"],
                    "attrition": row["attrition"],
                    "num_prev_companies": row["num_prev_companies"],
                    "years_in_company": row["years_in_company"],
                    "age": row["age"],
                    "education_level": row["education_level"],
                    "marital_status": row["marital_status"],
                    "gender": row["gender"],
                    "work_location": row["work_location"],
                }
            )
        db.commit()
        return {"message": "Data successfully uploaded and saved."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
