from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model for MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    email: EmailStr = Field(..., description="User's email address")
    mobile_number: str = Field(..., description="User's mobile number")
    password_hash: str = Field(..., description="Hashed password")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    class Config:
        from_attributes = True
        populate_by_name = True
