"""
Weave Generation Endpoints (/api/v1/generations)
Handles Multimodal Generation, Conversational Feedback Routing, Inpainting, and Variations.
"""

import os
import uuid
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.models import User, Upload
from backend.app.schemas.schemas import (
    GenerationResponse,
    ConceptImageResponse,
    FeedbackRequest,
    FeedbackResponse,
    VariationRequest,
)
from backend.app.repositories.generation_repository import GenerationRepository
from backend.app.services.orchestrator_service import LLMOrchestrator
from backend.app.services.generation_service import GenerationService
from backend.app.services.evaluation_service import EvaluationService
from backend.app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generations", tags=["Generations"])

# Singleton services
orchestrator = LLMOrchestrator()
generator = GenerationService()
evaluator = EvaluationService()


def save_upload_file(file: UploadFile, user_id: str, upload_type: str, db: Session) -> Upload:
    """Helper to persist uploaded sketch, reference, or mask images to disk and DB."""
    file_ext = Path = file.filename.split(".")[-1] if "." in file.filename else "png"
    upload_id = str(uuid.uuid4())
    filename = f"{upload_id}.{file_ext}"
    dest_path = settings.MEDIA_DIR / "uploads" / filename

    content = file.file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    upload = Upload(
        id=upload_id,
        user_id=user_id,
        upload_type=upload_type,
        file_path=f"/media/uploads/{filename}"
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


@router.post("", response_model=GenerationResponse, status_code=status.HTTP_201_CREATED)
async def create_generation(
    brief_text: str = Form(...),
    batch_size: int = Form(4),
    sketch: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Primary Multimodal Generation Endpoint.
    1. Extracts structured brief using LLM Orchestrator.
    2. Runs Stable Diffusion 1.5 + Fashion LoRA + ControlNet/sketch + IP-Adapter/reference.
    3. Runs Category Classifier, CLIP Style Alignment, and LPIPS Diversity evaluation.
    4. Persists generation and concepts in PostgreSQL/SQLite.
    """
    repo = GenerationRepository(db)

    # 1. Process Uploads if provided
    sketch_upload_id = None
    sketch_img = None
    if sketch and sketch.filename:
        u = save_upload_file(sketch, current_user.id, "sketch", db)
        sketch_upload_id = u.id
        sketch_path = settings.BASE_DIR / u.file_path.lstrip("/")
        if sketch_path.exists():
            sketch_img = Image.open(sketch_path)

    reference_upload_id = None
    ref_img = None
    if reference and reference.filename:
        u = save_upload_file(reference, current_user.id, "reference", db)
        reference_upload_id = u.id
        ref_path = settings.BASE_DIR / u.file_path.lstrip("/")
        if ref_path.exists():
            ref_img = Image.open(ref_path)

    # 2. Extract Structured Brief & Clarifying Question
    structured_brief, clarifying_question = orchestrator.extract_structured_brief(brief_text)

    # 3. Generate Fashion Concepts
    batch_results = generator.generate(
        structured_brief=structured_brief,
        batch_size=max(1, min(batch_size, 8)),
        sketch_image=sketch_img,
        reference_image=ref_img,
    )

    # 4. Evaluate Generated Concepts
    pil_images = [img for img, seed in batch_results]
    evaluations = evaluator.evaluate_concept_batch(pil_images, structured_brief)

    # 5. Persist Generation & Concept Images
    db_generation = repo.save_generation(
        user_id=current_user.id,
        structured_brief=structured_brief,
        raw_brief_text=brief_text,
        operation_type="generate",
        sketch_upload_id=sketch_upload_id,
        reference_upload_id=reference_upload_id,
    )

    concept_responses = []
    for (img, seed), ev in zip(batch_results, evaluations):
        concept_id, media_url = generator.save_concept_image(img)
        concept = repo.add_concept_image(
            generation_id=db_generation.id,
            file_path=media_url,
            seed=seed,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        )
        concept_responses.append(ConceptImageResponse(
            concept_image_id=concept.id,
            url=concept.file_path,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        ))

    return GenerationResponse(
        generation_id=db_generation.id,
        structured_brief=structured_brief,
        clarifying_question=clarifying_question,
        operation_type="generate",
        concepts=concept_responses,
    )


@router.post("/{generation_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    generation_id: str,
    payload: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Conversational Editing & Feedback Routing.
    Analyzes natural language feedback (e.g. "make the collar higher", "try in emerald green")
    and routes automatically to either Inpaint, Variations, or Parameterized Regeneration.
    """
    repo = GenerationRepository(db)
    parent_gen = repo.get_generation_by_id(generation_id)
    if not parent_gen:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Generation with ID '{generation_id}' not found."
        )

    # Pick representative concept from parent generation
    lead_concept = parent_gen.concepts[0] if parent_gen.concepts else None
    if not lead_concept:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parent generation contains no concept images."
        )

    # Route feedback via LLM Orchestrator
    routing = orchestrator.route_feedback(
        feedback_text=payload.feedback_text,
        current_brief=parent_gen.structured_brief or {}
    )

    routed_op = routing["operation"]
    reasoning = routing["reasoning"]
    updated_brief = routing["updated_brief"]

    # Execute routed operation
    if routed_op == "variation":
        base_img_path = settings.BASE_DIR / lead_concept.file_path.lstrip("/")
        base_img = Image.open(base_img_path) if base_img_path.exists() else Image.new("RGB", (512, 512), (220, 220, 220))
        batch_results = generator.generate_variations(base_img, strength=0.35, count=4, structured_brief=updated_brief)
    else:
        # Generate refined batch with updated brief
        batch_results = generator.generate(structured_brief=updated_brief, batch_size=4)

    # Evaluate
    pil_images = [img for img, seed in batch_results]
    evaluations = evaluator.evaluate_concept_batch(pil_images, updated_brief)

    # Save new generation linked to parent
    new_gen = repo.save_generation(
        user_id=current_user.id,
        structured_brief=updated_brief,
        raw_brief_text=payload.feedback_text,
        operation_type=routed_op,
        parent_generation_id=parent_gen.id,
    )

    # Record edit event
    repo.record_edit_event(
        concept_image_id=lead_concept.id,
        feedback_text=payload.feedback_text,
        routed_operation=routed_op,
        resulting_generation_id=new_gen.id,
    )

    concepts = []
    for (img, seed), ev in zip(batch_results, evaluations):
        cid, m_url = generator.save_concept_image(img)
        c = repo.add_concept_image(
            generation_id=new_gen.id,
            file_path=m_url,
            seed=seed,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        )
        concepts.append(ConceptImageResponse(
            concept_image_id=c.id,
            url=c.file_path,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        ))

    return FeedbackResponse(
        routed_operation=routed_op,
        reasoning_summary=reasoning,
        new_generation_id=new_gen.id,
        concepts=concepts,
    )


@router.post("/{generation_id}/inpaint", response_model=GenerationResponse)
async def inpaint_concept(
    generation_id: str,
    concept_image_id: str = Form(...),
    instruction: str = Form("refine masked area"),
    mask: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Direct Inpainting on a specific Concept Image with a user-drawn mask.
    """
    repo = GenerationRepository(db)
    parent_gen = repo.get_generation_by_id(generation_id)
    if not parent_gen:
        raise HTTPException(status_code=404, detail="Generation not found.")

    concept = repo.get_concept_by_id(concept_image_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept image not found.")

    # Load base image and mask
    base_path = settings.BASE_DIR / concept.file_path.lstrip("/")
    base_img = Image.open(base_path) if base_path.exists() else Image.new("RGB", (512, 512), (230, 225, 220))

    mask_upload = save_upload_file(mask, current_user.id, "mask", db)
    mask_path = settings.BASE_DIR / mask_upload.file_path.lstrip("/")
    mask_img = Image.open(mask_path) if mask_path.exists() else Image.new("L", (512, 512), 255)

    brief = parent_gen.structured_brief or {}
    inpainted_img, seed = generator.inpaint(
        base_image=base_img,
        mask_image=mask_img,
        prompt_instruction=instruction,
        structured_brief=brief,
    )

    ev_list = evaluator.evaluate_concept_batch([inpainted_img], brief)
    ev = ev_list[0]

    # Save as new inpaint generation linked to parent
    new_gen = repo.save_generation(
        user_id=current_user.id,
        structured_brief=brief,
        raw_brief_text=f"Inpaint: {instruction}",
        operation_type="inpaint",
        parent_generation_id=parent_gen.id,
    )

    repo.record_edit_event(
        concept_image_id=concept.id,
        feedback_text=instruction,
        routed_operation="inpaint",
        resulting_generation_id=new_gen.id
    )

    cid, m_url = generator.save_concept_image(inpainted_img)
    saved_concept = repo.add_concept_image(
        generation_id=new_gen.id,
        file_path=m_url,
        seed=seed,
        category_consistency_prob=ev["category_consistency_prob"],
        predicted_category=ev["predicted_category"],
        diversity_score=ev["diversity_score"],
        style_alignment_score=ev["style_alignment_score"],
    )

    concept_resp = ConceptImageResponse(
        concept_image_id=saved_concept.id,
        url=saved_concept.file_path,
        category_consistency_prob=ev["category_consistency_prob"],
        predicted_category=ev["predicted_category"],
        diversity_score=ev["diversity_score"],
        style_alignment_score=ev["style_alignment_score"],
    )

    return GenerationResponse(
        generation_id=new_gen.id,
        structured_brief=brief,
        clarifying_question=None,
        operation_type="inpaint",
        concepts=[concept_resp],
    )


@router.post("/{generation_id}/variations", response_model=GenerationResponse)
async def create_variations(
    generation_id: str,
    concept_image_id: str = Form(...),
    strength: float = Form(0.35),
    count: int = Form(4),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generates image-to-image variations of a specific concept image."""
    repo = GenerationRepository(db)
    parent_gen = repo.get_generation_by_id(generation_id)
    if not parent_gen:
        raise HTTPException(status_code=404, detail="Generation not found.")

    concept = repo.get_concept_by_id(concept_image_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found.")

    base_path = settings.BASE_DIR / concept.file_path.lstrip("/")
    base_img = Image.open(base_path) if base_path.exists() else Image.new("RGB", (512, 512), (230, 225, 220))

    brief = parent_gen.structured_brief or {}
    var_results = generator.generate_variations(
        base_image=base_img,
        strength=strength,
        count=max(1, min(count, 8)),
        structured_brief=brief
    )

    pil_images = [img for img, seed in var_results]
    evaluations = evaluator.evaluate_concept_batch(pil_images, brief)

    new_gen = repo.save_generation(
        user_id=current_user.id,
        structured_brief=brief,
        raw_brief_text=f"Variations of {concept.id[:8]}",
        operation_type="variation",
        parent_generation_id=parent_gen.id,
    )

    repo.record_edit_event(
        concept_image_id=concept.id,
        feedback_text="Generate variations",
        routed_operation="variation",
        resulting_generation_id=new_gen.id,
    )

    concepts = []
    for (img, seed), ev in zip(var_results, evaluations):
        cid, m_url = generator.save_concept_image(img)
        c = repo.add_concept_image(
            generation_id=new_gen.id,
            file_path=m_url,
            seed=seed,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        )
        concepts.append(ConceptImageResponse(
            concept_image_id=c.id,
            url=c.file_path,
            category_consistency_prob=ev["category_consistency_prob"],
            predicted_category=ev["predicted_category"],
            diversity_score=ev["diversity_score"],
            style_alignment_score=ev["style_alignment_score"],
        ))

    return GenerationResponse(
        generation_id=new_gen.id,
        structured_brief=brief,
        clarifying_question=None,
        operation_type="variation",
        concepts=concepts,
    )


@router.get("/{generation_id}", response_model=GenerationResponse)
def get_generation(
    generation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves a generation and all its scored concepts."""
    repo = GenerationRepository(db)
    gen = repo.get_generation_by_id(generation_id)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found.")

    concepts = []
    for c in gen.concepts:
        score = c.score
        concepts.append(ConceptImageResponse(
            concept_image_id=c.id,
            url=c.file_path,
            category_consistency_prob=score.category_consistency_prob if score else 0.90,
            predicted_category=score.predicted_category if score else "Garment",
            diversity_score=score.diversity_score if score else 0.50,
            style_alignment_score=score.style_alignment_score if score else 0.70,
        ))

    return GenerationResponse(
        generation_id=gen.id,
        structured_brief=gen.structured_brief or {},
        clarifying_question=None,
        operation_type=gen.operation_type,
        concepts=concepts,
    )


@router.get("", response_model=List[GenerationResponse])
def list_generations(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists recent generations for the logged in designer."""
    repo = GenerationRepository(db)
    gens = repo.get_recent_by_user(current_user.id, limit=limit)
    response = []
    for g in gens:
        concepts = []
        for c in g.concepts:
            score = c.score
            concepts.append(ConceptImageResponse(
                concept_image_id=c.id,
                url=c.file_path,
                category_consistency_prob=score.category_consistency_prob if score else 0.90,
                predicted_category=score.predicted_category if score else "Garment",
                diversity_score=score.diversity_score if score else 0.50,
                style_alignment_score=score.style_alignment_score if score else 0.70,
            ))
        response.append(GenerationResponse(
            generation_id=g.id,
            structured_brief=g.structured_brief or {},
            clarifying_question=None,
            operation_type=g.operation_type,
            concepts=concepts,
        ))
    return response
