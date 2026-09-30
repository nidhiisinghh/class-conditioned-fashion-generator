"""
Weave Application Configuration
Centralized configuration management using pydantic-settings.
"""

import os
from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # root /Users/nidhisingh/Desktop/ojt

class Settings(BaseSettings):
    PROJECT_NAME: str = "Weave Fashion AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    BASE_DIR: Path = BASE_DIR

    # Security & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "weave-super-secret-production-jwt-key-2026-fashion-ideation")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14

    # Database
    # Defaults to PostgreSQL, falls back to SQLite for local dev if Postgres is offline
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/weave_db"
    )

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # Storage Paths
    MEDIA_DIR: Path = BASE_DIR / "backend" / "media"
    CHECKPOINTS_DIR: Path = BASE_DIR / "ml" / "checkpoints"

    # Device & ML Inference
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto")
    USE_MOCK_DIFFUSION: bool = os.getenv("USE_MOCK_DIFFUSION", "false").lower() == "true"

    # External LLM API (Optional, fallback heuristic parser is built-in)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

# Ensure media storage directories exist
os.makedirs(settings.MEDIA_DIR / "generations", exist_ok=True)
os.makedirs(settings.MEDIA_DIR / "uploads", exist_ok=True)
os.makedirs(settings.MEDIA_DIR / "exports", exist_ok=True)
