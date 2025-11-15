from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime


class TransactionPredictionRequest(BaseModel):
    """Schema for transaction categorization request"""
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    description: str = Field(..., description="Transaction description", min_length=1)
    amount: float = Field(..., description="Transaction amount", gt=0)
    transaction_type: Optional[str] = Field(None, description="Type of transaction (e.g., CARD, P2P_TRANSFER)")
    currency: Optional[str] = Field(None, description="Currency code (e.g., INR, USD)")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp in ISO format")

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """Validate that description is not empty"""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate that amount is positive"""
        if v <= 0:
            raise ValueError("Amount must be greater than 0")
        return v


class TransactionPredictionResponse(BaseModel):
    """Schema for transaction categorization response"""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    category: str = Field(..., description="Predicted category")
    confidence_score: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        from_attributes = True


class BatchPredictionRequest(BaseModel):
    """Schema for batch transaction categorization request"""
    transactions: List[TransactionPredictionRequest] = Field(..., description="List of transactions to categorize", min_length=1)

    @field_validator('transactions')
    @classmethod
    def validate_transactions(cls, v):
        """Validate that at least one transaction is provided"""
        if not v or len(v) == 0:
            raise ValueError("At least one transaction must be provided")
        return v


class BatchPredictionResponse(BaseModel):
    """Schema for batch transaction categorization response"""
    predictions: List[TransactionPredictionResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")

    class Config:
        from_attributes = True


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


class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    model_name: Optional[str] = Field(None, description="Name of the model")
    current_version: Optional[str] = Field(None, description="Current active version")
    versions: Optional[Dict[str, Any]] = Field(None, description="Available versions and metadata")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    error: Optional[str] = Field(None, description="Error message if any")

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


class TrainingDataStats(BaseModel):
    """Schema for training data statistics"""
    total_labeled: int = Field(..., description="Total labeled transactions available")
    categories: Dict[str, int] = Field(..., description="Count of transactions per category")
    min_samples_per_category: int = Field(..., description="Minimum samples in any category")
    can_train_global: bool = Field(..., description="Whether global model can be trained")
    can_train_user: bool = Field(..., description="Whether user-specific model can be trained")

    class Config:
        from_attributes = True


class UploadTransactionsResponse(BaseModel):
    """Schema for CSV upload response"""
    inserted_count: int = Field(..., description="Number of transactions inserted")
    sample: List[Dict[str, Any]] = Field(..., description="Sample of inserted transactions")
    categorization_stats: Optional[CategorizationStats] = Field(None, description="Categorization statistics")
    training_data_available: Optional[int] = Field(None, description="Number of labeled transactions available for training")
    can_retrain_global: Optional[bool] = Field(None, description="Whether global model can be retrained")

    class Config:
        from_attributes = True

