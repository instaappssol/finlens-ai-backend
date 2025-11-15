"""
Central place to register all API routes
Import and include routers here
"""

from fastapi import APIRouter
from app.api.v1.auth_controller import router as auth_router
from app.api.v1.transactions_controller import router as transactions_router

# Create a combined router
router = APIRouter()

# Include all routers
router.include_router(auth_router)
router.include_router(transactions_router)

__all__ = ["router"]
