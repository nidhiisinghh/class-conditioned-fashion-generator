"""
Weave Moodboard Endpoints (/api/v1/boards)
Allows designers to curate concepts, annotate, and arrange visual moodboards (PRD Document 8.1).
"""

from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.models import User
from backend.app.schemas.schemas import (
    BoardCreate,
    BoardResponse,
    BoardItemCreate,
    BoardItemUpdate,
    BoardItemResponse,
    ConceptImageResponse,
)
from backend.app.services.board_service import BoardService
from backend.app.api.deps import get_current_user

router = APIRouter(prefix="/boards", tags=["Boards"])


@router.get("", response_model=List[BoardResponse])
def get_user_boards(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all moodboards created by the current designer."""
    service = BoardService(db)
    boards = service.list_user_boards(current_user.id)
    results = []
    for b in boards:
        results.append(BoardResponse(
            id=b.id,
            title=b.title,
            owner_id=b.owner_id,
            created_at=b.created_at,
            item_count=len(b.items) if b.items else 0,
        ))
    return results


@router.post("", response_model=BoardResponse, status_code=status.HTTP_201_CREATED)
def create_board(
    payload: BoardCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new empty moodboard."""
    service = BoardService(db)
    board = service.create_board(owner_id=current_user.id, title=payload.title)
    return BoardResponse(
        id=board.id,
        title=board.title,
        owner_id=board.owner_id,
        created_at=board.created_at,
        item_count=0,
        items=[]
    )


@router.get("/{board_id}", response_model=BoardResponse)
def get_board(
    board_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves a moodboard along with its pinned concept items."""
    service = BoardService(db)
    board = service.get_board(board_id, current_user.id)

    items_data = []
    for item in board.items or []:
        c = item.concept_image
        c_resp = None
        if c:
            score = c.score
            c_resp = ConceptImageResponse(
                concept_image_id=c.id,
                url=c.file_path,
                category_consistency_prob=score.category_consistency_prob if score else 0.90,
                predicted_category=score.predicted_category if score else "Garment",
                diversity_score=score.diversity_score if score else 0.50,
                style_alignment_score=score.style_alignment_score if score else 0.70,
            )

        items_data.append(BoardItemResponse(
            id=item.id,
            board_id=item.board_id,
            concept_image_id=item.concept_image_id,
            position_x=item.position_x,
            position_y=item.position_y,
            annotation=item.annotation,
            concept_image=c_resp,
            added_at=item.added_at,
        ))

    return BoardResponse(
        id=board.id,
        title=board.title,
        owner_id=board.owner_id,
        created_at=board.created_at,
        item_count=len(items_data),
        items=items_data
    )


@router.delete("/{board_id}")
def delete_board(
    board_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes a moodboard and its item mappings."""
    service = BoardService(db)
    service.delete_board(board_id, current_user.id)
    return {"message": "Board deleted successfully"}


@router.post("/{board_id}/items", response_model=BoardItemResponse, status_code=status.HTTP_201_CREATED)
def pin_concept_to_board(
    board_id: str,
    payload: BoardItemCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Pins a concept image to the board with coordinates and annotation."""
    service = BoardService(db)
    item = service.add_concept_to_board(
        board_id=board_id,
        user_id=current_user.id,
        concept_image_id=payload.concept_image_id,
        position_x=payload.position_x,
        position_y=payload.position_y,
        annotation=payload.annotation
    )
    return BoardItemResponse(
        id=item.id,
        board_id=item.board_id,
        concept_image_id=item.concept_image_id,
        position_x=item.position_x,
        position_y=item.position_y,
        annotation=item.annotation,
        concept_image=None,
        added_at=item.added_at,
    )


@router.patch("/{board_id}/items/{item_id}", response_model=BoardItemResponse)
def update_board_item(
    board_id: str,
    item_id: str,
    payload: BoardItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates position or annotation of a pinned concept."""
    service = BoardService(db)
    item = service.update_item_layout(
        board_id=board_id,
        item_id=item_id,
        user_id=current_user.id,
        position_x=payload.position_x,
        position_y=payload.position_y,
        annotation=payload.annotation
    )
    return BoardItemResponse(
        id=item.id,
        board_id=item.board_id,
        concept_image_id=item.concept_image_id,
        position_x=item.position_x,
        position_y=item.position_y,
        annotation=item.annotation,
        concept_image=None,
        added_at=item.added_at,
    )


@router.delete("/{board_id}/items/{item_id}")
def delete_board_item(
    board_id: str,
    item_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unpins a concept from the board."""
    service = BoardService(db)
    service.remove_item(board_id, item_id, current_user.id)
    return {"message": "Board item removed successfully"}
