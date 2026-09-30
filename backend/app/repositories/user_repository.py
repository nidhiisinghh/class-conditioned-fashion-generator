"""
Weave User Repository
Handles database access for users and refresh tokens.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from backend.app.models.models import User, RefreshToken
from backend.app.core.security import get_password_hash

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email.lower().strip()).first()

    def create(self, email: str, password: str, display_name: str, role: str = "designer") -> User:
        user = User(
            email=email.lower().strip(),
            hashed_password=get_password_hash(password),
            display_name=display_name,
            role=role,
            is_active=True
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def save_refresh_token(self, user_id: str, token_hash: str, expires_at: datetime) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            revoked=False
        )
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def get_valid_refresh_token(self, token_hash: str) -> Optional[RefreshToken]:
        now = datetime.now(timezone.utc)
        return self.db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked == False,
            RefreshToken.expires_at > now
        ).first()

    def revoke_refresh_token(self, token_hash: str) -> bool:
        token = self.db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
        if token:
            token.revoked = True
            self.db.commit()
            return True
        return False

    # Aliases for API layer compatibility
    create_refresh_token = save_refresh_token
    get_active_refresh_token = get_valid_refresh_token
