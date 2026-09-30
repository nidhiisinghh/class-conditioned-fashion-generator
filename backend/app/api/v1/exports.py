"""
Weave Export Endpoints (/api/v1/concepts/{id}/export)
Generates high-fashion PDF design briefs with prompt lineage, metrics, and fabric specifications (PRD Document 8.2).
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.models import User
from backend.app.schemas.schemas import ExportResponse
from backend.app.repositories.generation_repository import GenerationRepository
from backend.app.services.export_service import ExportService
from backend.app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Exports"])
export_service = ExportService()


@router.post("/concepts/{concept_id}/export", response_model=ExportResponse, status_code=status.HTTP_201_CREATED)
def export_concept_brief(
    concept_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Exports a comprehensive PDF design brief & tech pack for a concept image.
    Compiles:
    - Concept image render
    - Structured brief & color palette
    - Full prompt ancestry lineage
    - AI evaluation benchmark scores (Category Consistency, CLIP Alignment, LPIPS Diversity)
    """
    repo = GenerationRepository(db)
    concept = repo.get_concept_by_id(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept image with ID '{concept_id}' not found."
        )

    # Trace ancestry lineage
    lineage = repo.get_lineage(concept_id)

    # Generate PDF export
    export_result = export_service.generate_pdf_export(concept=concept, lineage=lineage)

    return ExportResponse(
        export_id=export_result["export_id"],
        pdf_url=export_result["pdf_url"],
        structured_brief=export_result["structured_brief"],
        prompt_lineage=export_result["prompt_lineage"],
    )
