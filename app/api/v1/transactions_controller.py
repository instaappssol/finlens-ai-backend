from fastapi import APIRouter, Request, UploadFile, File, status, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.core.dependencies import (
    get_transaction_service,
    get_categorization_service
)
from app.core.exceptions import InternalServerErrorException, ResponseBody
from app.services.transactions_service import TransactionService
from app.services.categorization_service import CategorizationService
from app.schemas.transaction_schema import (
    TransactionPredictionRequest,
    TransactionPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TrainingResponse,
    ModelInfoResponse,
    UploadTransactionsResponse,
    CategorizationStats,
    TrainingDataStats,
)

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post(
    "/upload-transactions",
    status_code=status.HTTP_200_OK,
    summary="Upload transactions CSV",
    description="Upload a CSV file containing transactions. Transactions will be auto-categorized if model is available.",
)
async def upload_transactions(
    request: Request,
    file: UploadFile = File(...),
    service: TransactionService = Depends(get_transaction_service)
):
    """Accepts a CSV file, auto-categorizes transactions, and inserts into DB."""
    try:
        # Get user_id from JWT token (set by middleware)
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')

        # Read uploaded file
        contents = await file.read()

        # Parse, categorize, and insert
        inserted_count, sample, stats = service.parse_csv_and_insert(
            contents, 
            user_id=user_id,
            auto_categorize=True
        )

        # Check if we should trigger model retraining
        training_stats = service.get_training_data_stats(user_id=None)  # Global stats
        
        # Build response data
        response_data = {
            "inserted_count": inserted_count,
            "sample": sample,
        }
        
        # Add categorization stats if available
        if stats:
            response_data["categorization_stats"] = CategorizationStats(**stats).model_dump()
            response_data["training_data_available"] = training_stats["total_labeled"]
            response_data["can_retrain_global"] = training_stats["can_train_global"]

        resp = ResponseBody(
            message="CSV processed successfully",
            errors=[],
            data=UploadTransactionsResponse(**response_data).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to process CSV", errors=[str(e)]
        )


@router.post(
    "/categorize",
    status_code=status.HTTP_200_OK,
    summary="Categorize a transaction",
    description="Predict the category for a single transaction using ML model.",
)
async def categorize_transaction(
    request: Request,
    transaction: TransactionPredictionRequest,
    service: CategorizationService = Depends(get_categorization_service)
):
    """Predict category for a single transaction"""
    try:
        result = service.predict(transaction.model_dump())
        
        resp = ResponseBody(
            message="Category predicted successfully",
            errors=[],
            data=TransactionPredictionResponse(**result).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to categorize transaction", errors=[str(e)]
        )


@router.post(
    "/categorize-batch",
    status_code=status.HTTP_200_OK,
    summary="Categorize multiple transactions",
    description="Predict categories for multiple transactions in batch.",
)
async def categorize_transactions_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    service: CategorizationService = Depends(get_categorization_service)
):
    """Predict categories for multiple transactions"""
    try:
        transactions = [txn.model_dump() for txn in batch_request.transactions]
        results = service.predict_batch(transactions)
        
        resp = ResponseBody(
            message="Categories predicted successfully",
            errors=[],
            data=BatchPredictionResponse(
                predictions=[TransactionPredictionResponse(**r) for r in results],
                count=len(results)
            ).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to categorize transactions", errors=[str(e)]
        )


@router.post(
    "/train-model",
    status_code=status.HTTP_200_OK,
    summary="Train categorization model",
    description="Train a new transaction categorization model from labeled CSV file. CSV must include: description, amount, transaction_type, date, and category (or label) columns.",
)
async def train_model(
    request: Request,
    file: UploadFile = File(...),
    version: str = Query("latest", description="Model version identifier"),
    transaction_service: TransactionService = Depends(get_transaction_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
):
    """Train a new categorization model from CSV file"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Parse CSV for training (validates minimum 2 categories)
        training_data, labels = transaction_service.parse_csv_for_training(contents)
        
        # Train model
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
    summary="Retrain model from labeled transactions",
    description="Retrain the global or user-specific model from labeled transactions in the database.",
)
async def retrain_model(
    request: Request,
    model_type: str = Query("global", description="Model type: 'global' or 'user'"),
    min_samples: int = Query(50, description="Minimum number of labeled transactions required"),
    service: TransactionService = Depends(get_transaction_service),
    categorization_service: CategorizationService = Depends(get_categorization_service)
):
    """Retrain model from database transactions"""
    try:
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')

        # Get labeled transactions
        if model_type == "user" and user_id:
            training_data = service.get_labeled_transactions(user_id=user_id)
        else:
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
        if model_type == "user" and user_id:
            result = categorization_service.train_user_model(
                user_id, training_list, labels, save_model=True
            )
        else:
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
    "/training-stats",
    status_code=status.HTTP_200_OK,
    summary="Get training data statistics",
    description="Get statistics about available labeled transactions for training.",
)
async def get_training_stats(
    request: Request,
    user_id: Optional[str] = None,
    service: TransactionService = Depends(get_transaction_service)
):
    """Get statistics about training data"""
    try:
        # If user_id not provided, try to get from JWT token
        if not user_id and hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')
        
        stats = service.get_training_data_stats(user_id=user_id)
        
        resp = ResponseBody(
            message="Training statistics retrieved successfully",
            errors=[],
            data=TrainingDataStats(**stats).model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get training stats", errors=[str(e)]
        )


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Get metadata about the current categorization model.",
)
async def get_model_info(
    request: Request,
    service: CategorizationService = Depends(get_categorization_service)
):
    """Get information about the loaded model"""
    try:
        info = service.get_model_info()
        
        # Format model info response
        model_info = ModelInfoResponse(
            model_name="transaction_categorizer",
            current_version=info.get("current"),
            versions=info,
            loaded=service.model_loaded,
            error=info.get("error")
        )
        
        resp = ResponseBody(
            message="Model information retrieved successfully",
            errors=[],
            data=model_info.model_dump(exclude_none=True),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())
    
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get model info", errors=[str(e)]
        )
