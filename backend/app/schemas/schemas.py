"""
Weave Pydantic Data Transfer Objects (Schemas)
Strictly adheres to API contracts in PRD Document 7.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# --- Auth Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    display_name: str
    role: str = "designer"

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    display_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    user: UserResponse
    access_token: str
    expires_in: int


# --- Generation & Brief Schemas ---
class StructuredBrief(BaseModel):
    category: str
    style: Optional[str] = "contemporary"
    color: Optional[str] = "terracotta"
    fabric: Optional[str] = "silk organza"
    mood: Optional[str] = "studio"

class GenerationCreate(BaseModel):
    brief_text: str
    batch_size: int = Field(default=4, ge=1, le=8)

class ConceptImageResponse(BaseModel):
    concept_image_id: str
    url: str
    category_consistency_prob: float
    predicted_category: str
    diversity_score: float
    style_alignment_score: float

    model_config = ConfigDict(from_attributes=True)

class GenerationResponse(BaseModel):
    generation_id: str
    structured_brief: Dict[str, Any]
    clarifying_question: Optional[str] = None
    operation_type: str = "generate"
    concepts: List[ConceptImageResponse] = []

    model_config = ConfigDict(from_attributes=True)


# --- Editing & Feedback Schemas ---
class FeedbackRequest(BaseModel):
    feedback_text: str

class FeedbackResponse(BaseModel):
    routed_operation: str  # inpaint | generate | variation
    reasoning_summary: str
    new_generation_id: str
    concepts: List[ConceptImageResponse]

class InpaintRequest(BaseModel):
    feedback_text: Optional[str] = "refine this masked region"

class VariationRequest(BaseModel):
    strength: float = Field(default=0.35, ge=0.1, le=0.9)
    count: int = Field(default=4, ge=1, le=8)


# --- Board Schemas ---
class BoardCreate(BaseModel):
    title: str

class BoardItemCreate(BaseModel):
    concept_image_id: str
    position_x: float = 120.0
    position_y: float = 140.0
    annotation: Optional[str] = None

class BoardItemUpdate(BaseModel):
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    annotation: Optional[str] = None

class BoardItemResponse(BaseModel):
    id: str
    board_id: str
    concept_image_id: str
    position_x: float
    position_y: float
    annotation: Optional[str] = None
    concept_image: Optional[ConceptImageResponse] = None
    added_at: datetime

    model_config = ConfigDict(from_attributes=True)

class BoardResponse(BaseModel):
    id: str
    title: str
    owner_id: str
    created_at: datetime
    item_count: int = 0
    items: Optional[List[BoardItemResponse]] = None

    model_config = ConfigDict(from_attributes=True)


# --- Export Schemas ---
class ExportResponse(BaseModel):
    export_id: str
    pdf_url: str
    structured_brief: Dict[str, Any]
    prompt_lineage: List[str]
