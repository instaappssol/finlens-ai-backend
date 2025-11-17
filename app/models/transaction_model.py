from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Transaction model for MongoDB"""

    id: Optional[str] = Field(None, alias="_id")
    user_id: Optional[str] = Field(
        None, description="User ID who uploaded the transaction"
    )
    amount: float = Field(..., description="Transaction amount")
    description: str = Field(..., description="Transaction description")
    category: Optional[str] = Field(None, description="Transaction category")
    date: Optional[str] = Field(None, description="Transaction date")
    transaction_type: Optional[str] = Field(
        None, 
        description="Transaction type: DEBIT (outflow/expense) or CREDIT (inflow/income)"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="XAI explanation for category prediction (includes top factors, confidence, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        populate_by_name = True
