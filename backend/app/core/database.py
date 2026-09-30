"""
Weave Database Engine & Session Management
Supports PostgreSQL 15 with resilient local SQLite fallback for development.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Determine active database URL with fallback
db_url = settings.DATABASE_URL
connect_args = {}

if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    # Quick probe
    with engine.connect() as conn:
        logger.info(f"Database connection established successfully: {engine.url.render_as_string(hide_password=True)}")
except Exception as e:
    logger.warning(f"Could not connect to configured DB ({db_url}): {e}")
    logger.warning("Falling back to local SQLite database: sqlite:///./weave_dev.db")
    db_url = "sqlite:///./weave_dev.db"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency that yields an active database session and closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes all database tables defined in SQLAlchemy models."""
    from backend.app.models.models import (
        User, RefreshToken, Upload, Generation, ConceptImage, ImageScore, EditEvent, Board, BoardItem
    )
    Base.metadata.create_all(bind=engine)
    logger.info("All Weave database tables verified/created successfully.")
