"""
Weave Board Repository
Handles database access for moodboards and board items.
"""

from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from backend.app.models.models import Board, BoardItem, ConceptImage

class BoardRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, owner_id: str, title: str) -> Board:
        board = Board(owner_id=owner_id, title=title)
        self.db.add(board)
        self.db.commit()
        self.db.refresh(board)
        return board

    def get_by_id(self, board_id: str) -> Optional[Board]:
        return (
            self.db.query(Board)
            .options(
                joinedload(Board.items)
                .joinedload(BoardItem.concept_image)
                .joinedload(ConceptImage.score)
            )
            .filter(Board.id == board_id)
            .first()
        )

    def list_by_owner(self, owner_id: str) -> List[Board]:
        return (
            self.db.query(Board)
            .options(joinedload(Board.items))
            .filter(Board.owner_id == owner_id)
            .order_by(Board.created_at.desc())
            .all()
        )

    def add_item(
        self,
        board_id: str,
        concept_image_id: str,
        position_x: float = 120.0,
        position_y: float = 140.0,
        annotation: Optional[str] = None
    ) -> BoardItem:
        item = BoardItem(
            board_id=board_id,
            concept_image_id=concept_image_id,
            position_x=position_x,
            position_y=position_y,
            annotation=annotation
        )
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    def update_item(
        self,
        item_id: str,
        position_x: Optional[float] = None,
        position_y: Optional[float] = None,
        annotation: Optional[str] = None
    ) -> Optional[BoardItem]:
        item = self.db.query(BoardItem).filter(BoardItem.id == item_id).first()
        if not item:
            return None
        if position_x is not None:
            item.position_x = position_x
        if position_y is not None:
            item.position_y = position_y
        if annotation is not None:
            item.annotation = annotation
        self.db.commit()
        self.db.refresh(item)
        return item

    def delete_item(self, item_id: str) -> bool:
        item = self.db.query(BoardItem).filter(BoardItem.id == item_id).first()
        if item:
            self.db.delete(item)
            self.db.commit()
            return True
        return False

    def delete_board(self, board_id: str) -> bool:
        board = self.db.query(Board).filter(Board.id == board_id).first()
        if board:
            self.db.delete(board)
            self.db.commit()
            return True
        return False
