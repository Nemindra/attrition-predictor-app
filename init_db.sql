-- Create employee_data table
CREATE TABLE IF NOT EXISTS employee_data (
    id SERIAL PRIMARY KEY,
    reporting_date DATE NOT NULL,
    employee_id VARCHAR(50) NOT NULL,
    department VARCHAR(50),
    designation VARCHAR(50),
    monthly_salary FLOAT,
    attrition BOOLEAN,
    num_prev_companies INT,
    years_in_company FLOAT,
    age INT,
    education_level VARCHAR(50),
    marital_status VARCHAR(50),
    gender VARCHAR(50),
    work_location VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    employee_id VARCHAR(50) NOT NULL,
    predicted_risk VARCHAR(50),
    model_version VARCHAR(50),
    actual_attrition BOOLEAN,
    prediction_accuracy FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create model_matrix table
CREATE TABLE IF NOT EXISTS model_matrix (
    model_version VARCHAR(50) PRIMARY KEY,
    training_date DATE NOT NULL,
    accuracy_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    roc_auc_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
