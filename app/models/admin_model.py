from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Admin(BaseModel):
    """Admin model for MongoDB"""

    id: Optional[str] = Field(None, alias="_id")
    name: str = Field(..., description="Admin's full name")
    email: EmailStr = Field(..., description="Admin's email address")
    password_hash: str = Field(..., description="Hashed password")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)

    class Config:
        from_attributes = True
        populate_by_name = True

