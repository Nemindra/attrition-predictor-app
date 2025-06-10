from pydantic import BaseModel
from datetime import date

class EmployeeData(BaseModel):
    reporting_date: date
    employee_id: str
    department: str
    designation: str
    monthly_salary: float
    attrition: bool
    num_prev_companies: int
    years_in_company: float
    age: int
    education_level: str
    marital_status: str
    gender: str
    work_location: str

class PredictionData(BaseModel):
    employee_id: str
    reporting_date: date
