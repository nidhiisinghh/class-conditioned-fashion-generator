"""
Weave API v1 Router Consolidator
Combines Auth, Generations, Boards, and Exports into a single API router.
"""

from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.generations import router as generations_router
from backend.app.api.v1.boards import router as boards_router
from backend.app.api.v1.exports import router as exports_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(generations_router)
api_router.include_router(boards_router)
api_router.include_router(exports_router)
