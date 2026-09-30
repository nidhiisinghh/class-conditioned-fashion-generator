"""
Weave Board Service
Business logic layer managing moodboards and canvas item arrangements.
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.repositories.board_repository import BoardRepository
from backend.app.models.models import Board, BoardItem

logger = logging.getLogger(__name__)


class BoardService:
    def __init__(self, db: Session):
        self.repo = BoardRepository(db)

    def create_board(self, owner_id: str, title: str) -> Board:
        if not title or not title.strip():
            title = "Untitled Moodboard"
        return self.repo.create(owner_id=owner_id, title=title.strip())

    def get_board(self, board_id: str, user_id: str) -> Board:
        board = self.repo.get_by_id(board_id)
        if not board:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board with ID '{board_id}' not found."
            )
        if board.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this board."
            )
        return board

    def list_user_boards(self, user_id: str) -> List[Board]:
        return self.repo.list_by_owner(user_id)

    def add_concept_to_board(
        self,
        board_id: str,
        user_id: str,
        concept_image_id: str,
        position_x: float = 120.0,
        position_y: float = 140.0,
        annotation: Optional[str] = None
    ) -> BoardItem:
        # Verify ownership
        self.get_board(board_id, user_id)
        return self.repo.add_item(
            board_id=board_id,
            concept_image_id=concept_image_id,
            position_x=position_x,
            position_y=position_y,
            annotation=annotation
        )

    def update_item_layout(
        self,
        board_id: str,
        item_id: str,
        user_id: str,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        annotation: Optional[str] = None
    ) -> BoardItem:
        # Verify ownership
        self.get_board(board_id, user_id)
        item = self.repo.update_item(
            item_id=item_id,
            position_x=position_x,
            position_y=position_y,
            annotation=annotation
        )
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board item '{item_id}' not found."
            )
        return item

    def remove_item(self, board_id: str, item_id: str, user_id: str) -> bool:
        self.get_board(board_id, user_id)
        deleted = self.repo.delete_item(item_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board item '{item_id}' not found."
            )
        return True

    def delete_board(self, board_id: str, user_id: str) -> bool:
        self.get_board(board_id, user_id)
        return self.repo.delete_board(board_id)
