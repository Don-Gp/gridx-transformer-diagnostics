"""
GRIDX Configuration Settings
"""
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./gridx.db")
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GRIDX"
    
    # ML Models
    MODEL_PATH: str = "app/ml_models/trained"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # SCADA Integration
    SCADA_ENABLED: bool = os.getenv("SCADA_ENABLED", "false").lower() == "true"
    OPC_UA_ENDPOINT: str = os.getenv("OPC_UA_ENDPOINT", "")
    
    class Config:
        case_sensitive = True

settings = Settings()
