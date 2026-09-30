"""
Weave Generation Repository
Handles database access for generations, concept images, scores, and edit history.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from backend.app.models.models import Generation, ConceptImage, ImageScore, EditEvent

class GenerationRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_generation(
        self,
        user_id: str,
        structured_brief: Dict[str, Any],
        raw_brief_text: Optional[str] = None,
        operation_type: str = "generate",
        sketch_upload_id: Optional[str] = None,
        reference_upload_id: Optional[str] = None,
        parent_generation_id: Optional[str] = None,
    ) -> Generation:
        generation = Generation(
            user_id=user_id,
            structured_brief=structured_brief,
            raw_brief_text=raw_brief_text,
            operation_type=operation_type,
            sketch_upload_id=sketch_upload_id,
            reference_upload_id=reference_upload_id,
            parent_generation_id=parent_generation_id,
        )
        self.db.add(generation)
        self.db.commit()
        self.db.refresh(generation)
        return generation

    def add_concept_image(
        self,
        generation_id: str,
        file_path: str,
        seed: Optional[int] = None,
        category_consistency_prob: float = 0.90,
        predicted_category: str = "Dress",
        diversity_score: float = 0.50,
        style_alignment_score: float = 0.75,
    ) -> ConceptImage:
        concept = ConceptImage(
            generation_id=generation_id,
            file_path=file_path,
            seed=seed,
        )
        self.db.add(concept)
        self.db.flush()

        score = ImageScore(
            concept_image_id=concept.id,
            category_consistency_prob=category_consistency_prob,
            predicted_category=predicted_category,
            diversity_score=diversity_score,
            style_alignment_score=style_alignment_score,
        )
        self.db.add(score)
        self.db.commit()
        self.db.refresh(concept)
        return concept

    def get_generation_by_id(self, generation_id: str) -> Optional[Generation]:
        return (
            self.db.query(Generation)
            .options(joinedload(Generation.concepts).joinedload(ConceptImage.score))
            .filter(Generation.id == generation_id)
            .first()
        )

    def get_concept_by_id(self, concept_image_id: str) -> Optional[ConceptImage]:
        return (
            self.db.query(ConceptImage)
            .options(joinedload(ConceptImage.score), joinedload(ConceptImage.generation))
            .filter(ConceptImage.id == concept_image_id)
            .first()
        )

    def record_edit_event(
        self,
        concept_image_id: str,
        feedback_text: str,
        routed_operation: str,
        resulting_generation_id: Optional[str] = None
    ) -> EditEvent:
        event = EditEvent(
            concept_image_id=concept_image_id,
            feedback_text=feedback_text,
            routed_operation=routed_operation,
            resulting_generation_id=resulting_generation_id
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_recent_by_user(self, user_id: str, limit: int = 20) -> List[Generation]:
        return (
            self.db.query(Generation)
            .options(joinedload(Generation.concepts).joinedload(ConceptImage.score))
            .filter(Generation.user_id == user_id)
            .order_by(Generation.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_lineage(self, concept_image_id: str) -> List[Generation]:
        """
        Walks parent_generation_id backwards to trace the original root brief.
        Powers the Design Brief Export (PRD Document 8.2).
        """
        lineage = []
        concept = self.get_concept_by_id(concept_image_id)
        if not concept:
            return lineage

        current_gen = concept.generation
        visited = set()

        while current_gen and current_gen.id not in visited:
            visited.add(current_gen.id)
            lineage.append(current_gen)
            if current_gen.parent_generation_id:
                current_gen = self.db.query(Generation).filter(Generation.id == current_gen.parent_generation_id).first()
            else:
                break

        lineage.reverse()  # Chronological order from root to current
        return lineage
