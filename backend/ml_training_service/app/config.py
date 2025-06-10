"""
config.py
---------
Configuration management for the ML Training Service using Pydantic BaseSettings.
Loads environment variables from the environment or a .env file for local/dev use.
Exports a config instance for use in other modules.
"""

from pydantic import BaseSettings, Field
from typing import Optional
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

class Config(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.
    Environment variables:
        - DB_URL: Database connection string (required).
        - MLFLOW_TRACKING_URI: MLflow tracking server URI (default: http://localhost:5000).
        - S3_BUCKET: S3 bucket name for model artifacts (optional).
    """
    DB_URL: str = Field(..., env="DATABASE_URL", description="Database connection string (e.g., postgresql://user:pass@host:port/dbname)")
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI", description="MLflow tracking server URI.")
    S3_BUCKET: Optional[str] = Field(None, env="S3_BUCKET", description="S3 bucket for model artifacts (optional)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Export a single config instance for use throughout the app
config = Config()
