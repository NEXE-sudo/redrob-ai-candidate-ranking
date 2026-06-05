from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application configuration."""
    
    # API
    API_TITLE: str = "Redrob AI Recruiter"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered recruitment intelligence platform"
    
    # Server
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://localhost:5432/redrob"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # AI/ML
    GEMINI_API_KEY: str = ""
    MODEL_NAME: str = "sentence-transformers/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    # Retrieval
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    MAX_CANDIDATES_TO_LLM: int = 5
    
    # Ranking weights (configurable)
    WEIGHT_TECHNICAL: float = 0.35
    WEIGHT_EXPERIENCE: float = 0.20
    WEIGHT_PROJECTS: float = 0.15
    WEIGHT_BEHAVIOUR: float = 0.10
    WEIGHT_SEMANTIC: float = 0.20
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
