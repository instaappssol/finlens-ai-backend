from pydantic import BaseModel, EmailStr, Field, field_validator
import re
from app.core.exceptions import UnprocessableEntityException


class AdminSignupRequest(BaseModel):
    """Schema for admin signup request"""
    name: str = Field(..., description="Admin's full name", min_length=1, max_length=100)
    email: EmailStr = Field(..., description="Admin's email address")
    password: str = Field(..., description="Admin's password", min_length=6)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate that password meets security requirements"""
        errors = []
        if len(v) < 6:
            errors.append('Password must be at least 6 characters long')
        if not re.search(r'[A-Z]', v):
            errors.append('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            errors.append('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            errors.append('Password must contain at least one digit')
        
        if errors:
            raise UnprocessableEntityException('Invalid password', errors)
        return v


class AdminSignupResponse(BaseModel):
    """Schema for admin signup response"""
    id: str = Field(..., description="Admin ID")
    name: str = Field(..., description="Admin's full name")
    email: str = Field(..., description="Admin's email address")
    message: str = Field(..., description="Success message")

    class Config:
        from_attributes = True


class AdminLoginRequest(BaseModel):
    """Schema for admin login request"""
    email: EmailStr = Field(..., description="Admin's email address")
    password: str = Field(..., description="Admin's password")

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate that password is not empty"""
        if not v or not v.strip():
            raise UnprocessableEntityException(
                'Invalid password',
                ['Password cannot be empty']
            )
        return v


class AdminLoginResponse(BaseModel):
    """Schema for admin login response"""
    id: str = Field(..., description="Admin ID")
    name: str = Field(..., description="Admin's full name")
    email: str = Field(..., description="Admin's email address")
    token: str = Field(..., description="JWT access token")

    class Config:
        from_attributes = True

