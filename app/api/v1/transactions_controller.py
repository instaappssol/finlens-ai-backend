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
    UploadTransactionsResponse,
    CategorizationStats,
    AnalyticsSummaryResponse,
    CategoryBreakdown,
    UniqueCategoriesResponse,
    UpdateCategoryRequest,
    UpdateCategoryResponse,
    CategoryTransactionsResponse,
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

        # Build response data
        response_data = {
            "inserted_count": inserted_count,
            "sample": sample,
        }
        
        # Add categorization stats if available
        if stats:
            response_data["categorization_stats"] = CategorizationStats(**stats).model_dump()

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


@router.get(
    "/analytics",
    status_code=status.HTTP_200_OK,
    summary="Get transaction analytics summary",
    description="Get analytics summary (inflows, outflows, category breakdowns) for a specific month and year.",
)
async def get_analytics_summary(
    request: Request,
    year: int = Query(..., description="Year (e.g., 2024)", ge=2000, le=2100),
    month: int = Query(..., description="Month (1-12)", ge=1, le=12),
    service: TransactionService = Depends(get_transaction_service)
):
    """Get analytics summary for a specific month and year"""
    try:
        # Get user_id from JWT token (set by middleware) - REQUIRED
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')
        
        if not user_id:
            raise BadRequestException(
                message="User authentication required",
                errors=["User ID not found in token"]
            )

        # Get analytics summary (only for this user)
        analytics_data = service.get_analytics_summary(year, month, user_id)

        # Convert category breakdowns to proper format
        inflows_by_category = [
            CategoryBreakdown(**item) for item in analytics_data["inflows_by_category"]
        ]
        outflows_by_category = [
            CategoryBreakdown(**item) for item in analytics_data["outflows_by_category"]
        ]

        response_data = AnalyticsSummaryResponse(
            total_inflows=analytics_data["total_inflows"],
            total_outflows=analytics_data["total_outflows"],
            diff_percentage=analytics_data["diff_percentage"],
            inflows_by_category=inflows_by_category,
            outflows_by_category=outflows_by_category,
        )

        resp = ResponseBody(
            message="Analytics summary retrieved successfully",
            errors=[],
            data=response_data.model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get analytics summary", errors=[str(e)]
        )


@router.get(
    "/categories",
    status_code=status.HTTP_200_OK,
    summary="Get unique categories from merchant knowledge base",
    description="Get a list of all unique categories available in the merchant knowledge base.",
)
async def get_unique_categories(
    request: Request,
    categorization_service: CategorizationService = Depends(get_categorization_service)
):
    """Get unique categories from the merchant knowledge base"""
    try:
        # Get unique categories from merchant knowledge base
        categories = categorization_service.get_unique_categories_from_kb()
        
        response_data = UniqueCategoriesResponse(
            categories=categories,
            count=len(categories)
        )

        resp = ResponseBody(
            message="Categories retrieved successfully",
            errors=[],
            data=response_data.model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get categories", errors=[str(e)]
        )


@router.post(
    "/{transaction_id}/feedback",
    status_code=status.HTTP_200_OK,
    summary="Submit category feedback",
    description="Submit feedback about a transaction's category. This feedback will be used for future categorizations of similar transactions. The transaction itself is not updated.",
)
async def submit_category_feedback(
    request: Request,
    transaction_id: str,
    update_request: UpdateCategoryRequest,
    service: TransactionService = Depends(get_transaction_service)
):
    """Store user feedback for future categorizations (does not update transaction)"""
    try:
        # Get user_id from JWT token (set by middleware)
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')
        
        if not user_id:
            raise BadRequestException(
                message="User authentication required",
                errors=["User ID not found in token"]
            )

        # Store user feedback
        result = service.store_user_feedback(
            transaction_id=transaction_id,
            category=update_request.category,
            user_id=user_id
        )

        response_data = UpdateCategoryResponse(**result)

        resp = ResponseBody(
            message="Feedback stored successfully. This will be used for future categorizations.",
            errors=[],
            data=response_data.model_dump(),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())

    except ValueError as e:
        raise BadRequestException(
            message="Failed to store feedback",
            errors=[str(e)]
        )
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to store feedback", errors=[str(e)]
        )

@router.get(
    "/category-transactions",
    status_code=status.HTTP_200_OK,
    summary="Get transactions by category",
    description="Get all transactions for a specific category in a given month and year. Optionally filter by transaction type (DEBIT/CREDIT).",
)
async def get_category_transactions(
    request: Request,
    year: int = Query(..., description="Year (e.g., 2024)", ge=2000, le=2100),
    month: int = Query(..., description="Month (1-12)", ge=1, le=12),
    category: str = Query(..., description="Category name"),
    transaction_type: Optional[str] = Query(None, description="Transaction type (DEBIT/CREDIT). Optional filter."),
    service: TransactionService = Depends(get_transaction_service)
):
    """Get all transactions for a specific category in a month and year"""
    try:
        # Get user_id from JWT token (set by middleware)
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = request.state.user.get('user_id') or request.state.user.get('sub')
        
        if not user_id:
            raise BadRequestException(
                message="User authentication required",
                errors=["User ID not found in token"]
            )

        # Get transactions by category
        transactions = service.get_transactions_by_category(
            year=year, 
            month=month, 
            category=category, 
            user_id=user_id,
            transaction_type=transaction_type
        )

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

    except BadRequestException:
        raise
    except Exception as e:
        raise InternalServerErrorException(
            message="Failed to get category transactions", errors=[str(e)]
        )
