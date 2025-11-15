from datetime import datetime
from typing import Optional
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        populate_by_name = True
