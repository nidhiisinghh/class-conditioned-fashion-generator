"""
Weave - Multimodal AI Fashion Ideation Assistant
Main FastAPI Application Entrypoint
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import init_db, get_db, engine
from backend.app.api.v1.api import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("weave.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown hooks."""
    logger.info("Initializing Weave Fashion AI Backend...")
    # Initialize relational database tables
    try:
        init_db()
        logger.info("Database schema initialized and validated.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    # Ensure media directories exist
    os.makedirs(settings.MEDIA_DIR / "generations", exist_ok=True)
    os.makedirs(settings.MEDIA_DIR / "uploads", exist_ok=True)
    os.makedirs(settings.MEDIA_DIR / "exports", exist_ok=True)
    logger.info(f"Media directories verified at {settings.MEDIA_DIR}")

    yield

    logger.info("Weave Fashion AI Backend shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend services for Weave: Multimodal AI Fashion Ideation Assistant. "
                "Featuring fine-tuned Fashion LoRA diffusion, sketch ControlNet, "
                "automated category consistency scoring, and moodboard curation.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Media Storage for Concept Images and Exports
app.mount(
    "/media",
    StaticFiles(directory=str(settings.MEDIA_DIR)),
    name="media"
)

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    """Root health and welcome endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": f"{settings.API_V1_STR}/docs",
        "device": settings.DEVICE,
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """System health check and database ping."""
    db_status = "connected"
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "checkpoints_present": {
            "fashion_lora": (settings.CHECKPOINTS_DIR / "final_fashion_lora" / "adapter_model.safetensors").exists(),
            "category_classifier": (settings.CHECKPOINTS_DIR / "classifier" / "fashion_classifier_best.pt").exists(),
        },
        "media_storage": str(settings.MEDIA_DIR),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
