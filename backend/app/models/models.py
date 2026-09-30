"""
Weave SQLAlchemy Relational Models
Implements normalized relational schema strictly defined in PRD Document 6.
Compatible with PostgreSQL 15 & SQLite (for local dev).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Text, Float, Integer, JSON
)
from sqlalchemy.orm import relationship
from backend.app.core.database import Base

def gen_uuid() -> str:
    return str(uuid.uuid4())

def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default="designer")  # designer | analyst | admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    uploads = relationship("Upload", back_populates="user", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="user", cascade="all, delete-orphan")
    boards = relationship("Board", back_populates="owner", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    user = relationship("User", back_populates="refresh_tokens")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    upload_type = Column(String(20), nullable=False)  # sketch | reference | mask
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    user = relationship("User", back_populates="uploads")


class Generation(Base):
    __tablename__ = "generations"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    structured_brief = Column(JSON, nullable=False)  # {category, style, color, fabric, mood}
    raw_brief_text = Column(Text, nullable=True)
    sketch_upload_id = Column(String(36), ForeignKey("uploads.id", ondelete="SET NULL"), nullable=True)
    reference_upload_id = Column(String(36), ForeignKey("uploads.id", ondelete="SET NULL"), nullable=True)
    operation_type = Column(String(20), nullable=False, default="generate")  # generate | inpaint | variation
    parent_generation_id = Column(String(36), ForeignKey("generations.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True)

    # Relationships
    user = relationship("User", back_populates="generations")
    concepts = relationship("ConceptImage", back_populates="generation", cascade="all, delete-orphan")
    parent = relationship("Generation", remote_side=[id], backref="children")


class ConceptImage(Base):
    __tablename__ = "concept_images"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    generation_id = Column(String(36), ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    seed = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    generation = relationship("Generation", back_populates="concepts")
    score = relationship("ImageScore", back_populates="concept_image", uselist=False, cascade="all, delete-orphan")
    edit_events = relationship("EditEvent", back_populates="concept_image", cascade="all, delete-orphan")
    board_items = relationship("BoardItem", back_populates="concept_image")


class ImageScore(Base):
    __tablename__ = "image_scores"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    concept_image_id = Column(String(36), ForeignKey("concept_images.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    category_consistency_prob = Column(Float, nullable=False)
    predicted_category = Column(String(50), nullable=False)
    diversity_score = Column(Float, nullable=False)
    style_alignment_score = Column(Float, nullable=False)
    scored_at = Column(DateTime(timezone=True), default=utc_now)

    concept_image = relationship("ConceptImage", back_populates="score")


class EditEvent(Base):
    __tablename__ = "edit_events"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    concept_image_id = Column(String(36), ForeignKey("concept_images.id", ondelete="CASCADE"), nullable=False, index=True)
    feedback_text = Column(Text, nullable=False)
    routed_operation = Column(String(20), nullable=False)  # inpaint | generate | variation
    resulting_generation_id = Column(String(36), ForeignKey("generations.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    concept_image = relationship("ConceptImage", back_populates="edit_events")


class Board(Base):
    __tablename__ = "boards"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    owner = relationship("User", back_populates="boards")
    items = relationship("BoardItem", back_populates="board", cascade="all, delete-orphan")


class BoardItem(Base):
    __tablename__ = "board_items"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    board_id = Column(String(36), ForeignKey("boards.id", ondelete="CASCADE"), nullable=False, index=True)
    concept_image_id = Column(String(36), ForeignKey("concept_images.id", ondelete="CASCADE"), nullable=False, index=True)
    position_x = Column(Float, default=100.0)
    position_y = Column(Float, default=100.0)
    annotation = Column(Text, nullable=True)
    added_at = Column(DateTime(timezone=True), default=utc_now)

    board = relationship("Board", back_populates="items")
    concept_image = relationship("ConceptImage", back_populates="board_items")
