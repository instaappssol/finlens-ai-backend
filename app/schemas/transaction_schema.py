from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


class TrainingTransaction(BaseModel):
    """Schema for a single training transaction (must include label)"""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    description: str = Field(..., description="Transaction description")
    amount: float = Field(..., description="Transaction amount", gt=0)
    transaction_type: Optional[str] = Field(None, description="Type of transaction")
    currency: Optional[str] = Field(None, description="Currency code")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")
    label: str = Field(..., description="Category label for training", min_length=1)

    @field_validator('label')
    @classmethod
    def validate_label(cls, v):
        """Validate that label is not empty"""
        if not v or not v.strip():
            raise ValueError("Label cannot be empty")
        return v.strip()


class TrainingDataRequest(BaseModel):
    """Schema for model training request"""
    transactions: List[Dict[str, Any]] = Field(..., description="List of labeled transactions (must include 'label' field)", min_length=1)
    version: Optional[str] = Field("latest", description="Model version identifier")

    @field_validator('transactions')
    @classmethod
    def validate_transactions(cls, v):
        """Validate that transactions have labels"""
        if not v or len(v) == 0:
            raise ValueError("At least one transaction must be provided")
        
        for i, txn in enumerate(v):
            if not isinstance(txn, dict):
                raise ValueError(f"Transaction at index {i} must be a dictionary")
            if "label" not in txn or not txn.get("label"):
                raise ValueError(f"Transaction at index {i} must include a 'label' field")
            if "description" not in txn:
                raise ValueError(f"Transaction at index {i} must include a 'description' field")
            if "amount" not in txn:
                raise ValueError(f"Transaction at index {i} must include an 'amount' field")
        
        return v


class TrainingResponse(BaseModel):
    """Schema for model training response"""
    status: str = Field(..., description="Training status")
    model_version: str = Field(..., description="Version of the trained model")
    training_samples: int = Field(..., description="Number of training samples used")
    categories: List[str] = Field(..., description="List of categories in the training data")

    class Config:
        from_attributes = True


class CategorizationStats(BaseModel):
    """Schema for categorization statistics"""
    total_inserted: int = Field(..., description="Total transactions inserted")
    auto_categorized: int = Field(..., description="Number of transactions auto-categorized")
    failed_categorizations: int = Field(..., description="Number of failed categorizations")
    already_categorized: int = Field(..., description="Number of transactions that already had categories")

    class Config:
        from_attributes = True


class UploadTransactionsResponse(BaseModel):
    """Schema for CSV upload response"""
    inserted_count: int = Field(..., description="Number of transactions inserted")
    sample: List[Dict[str, Any]] = Field(..., description="Sample of inserted transactions")
    categorization_stats: Optional[CategorizationStats] = Field(None, description="Categorization statistics")

    class Config:
        from_attributes = True


class CategoryBreakdown(BaseModel):
    """Schema for category breakdown in analytics"""
    category: str = Field(..., description="Category name")
    amount: float = Field(..., description="Total amount for this category")
    count: int = Field(..., description="Number of transactions in this category")
    percentage: float = Field(..., description="Percentage of total inflows/outflows this category represents")

    class Config:
        from_attributes = True


class AnalyticsSummaryResponse(BaseModel):
    """Schema for analytics summary response"""
    total_inflows: float = Field(..., description="Total inflow amount")
    total_outflows: float = Field(..., description="Total outflow amount")
    diff_percentage: float = Field(..., description="Difference percentage (positive = more inflows, negative = more outflows)")
    inflows_by_category: List[CategoryBreakdown] = Field(..., description="Inflows aggregated by category")
    outflows_by_category: List[CategoryBreakdown] = Field(..., description="Outflows aggregated by category")

    class Config:
        from_attributes = True


class CategoryTransactionsResponse(BaseModel):
    """Schema for category transactions response"""
    transactions: List[Dict[str, Any]] = Field(..., description="List of transactions")
    count: int = Field(..., description="Number of transactions")
    category: str = Field(..., description="Category name")
    year: int = Field(..., description="Year")
    month: int = Field(..., description="Month")

    class Config:
        from_attributes = True


class UniqueCategoriesResponse(BaseModel):
    """Schema for unique categories response"""
    categories: List[str] = Field(..., description="List of unique categories from merchant knowledge base")
    count: int = Field(..., description="Number of unique categories")

    class Config:
        from_attributes = True


class UpdateCategoryRequest(BaseModel):
    """Schema for submitting category feedback"""
    category: str = Field(..., description="Category feedback from user", min_length=1)

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        """Validate that category is not empty"""
        if not v or not v.strip():
            raise ValueError("Category cannot be empty")
        return v.strip()


class UpdateCategoryResponse(BaseModel):
    """Schema for feedback response"""
    transaction_id: str = Field(..., description="Transaction ID that feedback was provided for")
    category: str = Field(..., description="Category feedback that was stored")
    status: str = Field(..., description="Feedback status")
    message: str = Field(..., description="Response message")

    class Config:
        from_attributes = True


class DeleteTransactionResponse(BaseModel):
    """Schema for delete transaction response"""
    transaction_id: str = Field(..., description="Transaction ID that was deleted")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Response message")

    class Config:
        from_attributes = True


class DeleteAllTransactionsResponse(BaseModel):
    """Schema for delete all transactions response"""
    user_id: str = Field(..., description="User ID whose transactions were deleted")
    deleted_count: int = Field(..., description="Number of transactions deleted")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Response message")

    class Config:
        from_attributes = True


class DeleteAllTransactionsAdminResponse(BaseModel):
    """Schema for admin delete all transactions response"""
    deleted_count: int = Field(..., description="Number of transactions deleted")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Response message")

    class Config:
        from_attributes = True

