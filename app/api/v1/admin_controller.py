from fastapi import APIRouter, Request, UploadFile, File, status, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.core.dependencies import (
    get_transaction_service,
    get_categorization_service
)
from app.core.db import get_db_from_request
from app.core.exceptions import InternalServerErrorException, BadRequestException, ResponseBody
from app.services.transactions_service import TransactionService
from app.services.categorization_service import CategorizationService
from app.schemas.transaction_schema import (
    TrainingResponse,
    CategoryTransactionsResponse,
    DeleteAllTransactionsAdminResponse,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/train-model",
    status_code=status.HTTP_200_OK,
    summary="Train categorization model (Admin Only)",
    description="Train a new transaction categorization model from labeled CSV file. CSV must include: description, amount, transaction_type, date, and category (or label) columns.",
)
async def train_model(
    file: UploadFile = File(...),
    version: str = Query("latest", description="Model version identifier"),
    transaction_service: TransactionService = Depends(get_transaction_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
):
    """Train a new categorization model from CSV file (Admin Only)"""
    try:
        # Read uploaded CSV file
        contents = await file.read()
        
        # Parse CSV for training (validates minimum 2 categories)
        training_data, labels = transaction_service.parse_csv_for_training(contents)
        
        result = categorization_service.train_model(
            training_data,
            labels,
            save_model=True,
            version=version,
        )
        
        resp = ResponseBody(
            message="Model trained successfully",
            errors=[],
            data=TrainingResponse(**result).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except ValueError as e:
        raise InternalServerErrorException(
            message=f"Invalid training data: {str(e)}", errors=[str(e)]
        )
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to train model", errors=[str(e)]
        )


@router.post(
    "/retrain-model",
    status_code=status.HTTP_200_OK,
    summary="Retrain model from labeled transactions (Admin Only)",
    description="Retrain the global or user-specific model from labeled transactions in the database.",
)
async def retrain_model(
    min_samples: int = Query(50, description="Minimum number of labeled transactions required"),
    service: TransactionService = Depends(get_transaction_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
):
    """Retrain model from database transactions (Admin Only)"""
    try:
        # Get labeled transactions
        training_data = service.get_labeled_transactions(user_id=None)

        if len(training_data) < min_samples:
            raise ValueError(
                f"Not enough labeled transactions. Need at least {min_samples}, got {len(training_data)}"
            )

        # Prepare training data
        labels = [txn.get("category") for txn in training_data]
        training_list = [
            {
                "description": txn.get("description", ""),
                "amount": float(txn.get("amount", 0)),
                "transaction_type": txn.get("transaction_type", "UNKNOWN"),
                "currency": txn.get("currency", "INR"),
                "timestamp": txn.get("date") or txn.get("timestamp", ""),
                "label": txn.get("category")
            }
            for txn in training_data
        ]

        # Train model
        result = categorization_service.train_model(
            training_list, labels, save_model=True, version="latest"
        )

        resp = ResponseBody(
            message="Model retrained successfully",
            errors=[],
            data=TrainingResponse(**result).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to retrain model", errors=[str(e)]
        )


@router.get(
    "/category-transactions",
    status_code=status.HTTP_200_OK,
    summary="Get transactions by category (Admin Only)",
    description="Get all transactions for a specific category in a given month and year.",
)
async def get_category_transactions(
    year: int = Query(..., description="Year (e.g., 2024)", ge=2000, le=2100),
    month: int = Query(..., description="Month (1-12)", ge=1, le=12),
    category: str = Query(..., description="Category name"),
    user_id: Optional[str] = Query(None, description="Optional user ID to filter by"),
    service: TransactionService = Depends(get_transaction_service)
):
    """Get all transactions for a specific category in a month and year (Admin Only)"""
    try:
        # Get transactions by category
        transactions = service.get_transactions_by_category(year, month, category, user_id)

        response_data = CategoryTransactionsResponse(
            transactions=transactions,
            count=len(transactions),
            category=category,
            year=year,
            month=month,
        )

        resp = ResponseBody(
            message="Category transactions retrieved successfully",
            errors=[],
            data=response_data.model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get category transactions", errors=[str(e)]
        )


@router.post(
    "/upload-merchant-mappings",
    status_code=status.HTTP_200_OK,
    summary="Upload merchant mappings CSV (Admin Only)",
    description="Upload a CSV file containing merchant to category mappings. CSV must have 'Category' and 'Merchants' columns. Merchants can be a Python list string or comma-separated values.",
)
async def upload_merchant_mappings(
    request: Request,
    file: UploadFile = File(...),
    replace_existing: bool = Query(False, description="Replace existing mappings (default: append)"),
    service: TransactionService = Depends(get_transaction_service)
):
    """Upload merchant mappings CSV to MongoDB collection merchant_knowledge_base (Admin Only)"""
    try:
        # Get database connection
        db = get_db_from_request(request)
        
        # Read uploaded CSV file
        contents = await file.read()
        
        # Upload via service
        result = service.upload_merchant_mappings(
            db=db,
            file_content=contents,
            replace_existing=replace_existing
        )
        
        resp = ResponseBody(
            message="Merchant mappings uploaded successfully",
            errors=[],
            data=result,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except ValueError as e:
        raise BadRequestException(
            message=str(e),
            errors=[str(e)]
        )
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to upload merchant mappings", errors=[str(e)]
        )


@router.post(
    "/upload-merchant-corrections",
    status_code=status.HTTP_200_OK,
    summary="Upload merchant corrections CSV (Admin Only)",
    description="Upload a CSV file containing merchant name corrections. CSV must have 'raw_description', 'canonical_merchant', and 'canonical_category' columns.",
)
async def upload_merchant_corrections(
    request: Request,
    file: UploadFile = File(...),
    replace_existing: bool = Query(False, description="Replace existing corrections (default: append)"),
    service: TransactionService = Depends(get_transaction_service)
):
    """Upload merchant corrections CSV to MongoDB collection merchant_corrections (Admin Only)"""
    try:
        # Get database connection
        db = get_db_from_request(request)
        
        # Read uploaded CSV file
        contents = await file.read()
        
        # Upload via service
        result = service.upload_merchant_corrections(
            db=db,
            file_content=contents,
            replace_existing=replace_existing
        )
        
        resp = ResponseBody(
            message="Merchant corrections uploaded successfully",
            errors=[],
            data=result,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except ValueError as e:
        raise BadRequestException(
            message=str(e),
            errors=[str(e)]
        )
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to upload merchant corrections", errors=[str(e)]
        )


@router.delete(
    "/transactions/all",
    status_code=status.HTTP_200_OK,
    summary="Delete all transactions in database (Admin Only)",
    description="Delete all transaction records from the database. This action cannot be undone and affects all users.",
)
async def delete_all_transactions(
    service: TransactionService = Depends(get_transaction_service)
):
    """Delete all transactions in the database (Admin Only)"""
    try:
        # Delete all transactions
        result = service.delete_all_transactions()

        response_data = DeleteAllTransactionsAdminResponse(**result)

        resp = ResponseBody(
            message=f"Successfully deleted {result['deleted_count']} transaction(s) from database",
            errors=[],
            data=response_data.model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to delete all transactions", errors=[str(e)]
        )

