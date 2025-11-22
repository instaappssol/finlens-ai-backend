"""Dependency injection for clean architecture"""

from typing import Optional
from fastapi import Request, Depends
from app.core.db import get_db_from_request
from app.repositories.transaction_repository import TransactionRepository
from app.repositories.user_repository import UserRepository
from app.repositories.admin_repository import AdminRepository
from app.repositories.mongo_transaction_repository import MongoTransactionRepository
from app.repositories.mongo_user_repository import MongoUserRepository
from app.repositories.mongo_admin_repository import MongoAdminRepository
from app.services.transactions_service import TransactionService
from app.services.auth_service import AuthService
from app.services.admin_auth_service import AdminAuthService
from app.services.categorization_service import CategorizationService
from app.ml.transaction_categorization_engine import MerchantKnowledgeBase


def get_transaction_repository(request: Request) -> TransactionRepository:
    """
    Get transaction repository instance.

    Args:
        request: FastAPI request object

    Returns:
        TransactionRepository instance
    """
    db = get_db_from_request(request)
    return MongoTransactionRepository(db)


def get_user_repository(request: Request) -> UserRepository:
    """
    Get user repository instance.

    Args:
        request: FastAPI request object

    Returns:
        UserRepository instance
    """
    db = get_db_from_request(request)
    return MongoUserRepository(db)


# Singleton for categorization service
_categorization_service: Optional[CategorizationService] = None


def get_categorization_service(request: Request) -> CategorizationService:
    """
    Get categorization service instance (singleton pattern).

    Args:
        request: FastAPI request object

    Returns:
        CategorizationService instance
    """
    global _categorization_service
    
    # Get database connection
    db = get_db_from_request(request)
    
    if _categorization_service is None:
        _categorization_service = CategorizationService(db=db)
        # Try to load existing model
        _categorization_service.load_model()
    elif db is not None:
        # Update database connection if it changed
        if _categorization_service.db != db:
            _categorization_service.db = db
            # Reload merchant KB from new DB connection
            try:
                _categorization_service.default_merchant_kb = MerchantKnowledgeBase.from_mongodb(
                    db, collection_name="merchant_knowledge_base"
                )
            except Exception:
                pass
    
    return _categorization_service


def get_transaction_service(
    transaction_repo: TransactionRepository = Depends(get_transaction_repository),
    categorization_service: CategorizationService = Depends(get_categorization_service)
) -> TransactionService:
    """
    Get transaction service instance.

    Args:
        transaction_repo: Transaction repository (injected)
        categorization_service: Categorization service (injected)

    Returns:
        TransactionService instance
    """
    return TransactionService(
        transaction_repository=transaction_repo,
        categorization_service=categorization_service
    )


def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository)
) -> AuthService:
    """
    Get authentication service instance.

    Args:
        user_repo: User repository (injected)

    Returns:
        AuthService instance
    """
    return AuthService(user_repository=user_repo)


def get_admin_repository(request: Request) -> AdminRepository:
    """
    Get admin repository instance.

    Args:
        request: FastAPI request object

    Returns:
        AdminRepository instance
    """
    db = get_db_from_request(request)
    return MongoAdminRepository(db)


def get_admin_auth_service(request: Request) -> AdminAuthService:
    """
    Get admin authentication service instance.

    Args:
        request: FastAPI request object

    Returns:
        AdminAuthService instance
    """
    admin_repo = get_admin_repository(request)
    return AdminAuthService(admin_repository=admin_repo)

