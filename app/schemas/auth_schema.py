from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class SignupRequest(BaseModel):
    """Schema for user signup request"""
    email: EmailStr = Field(..., description="User's email address")
    mobile_number: str = Field(..., description="User's mobile number", min_length=10, max_length=15)
    password: str = Field(..., description="User's password", min_length=6)

    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        """Validate that mobile number contains only digits"""
        if not re.match(r'^\d{10,15}$', v):
            raise ValueError('Mobile number must contain only digits and be between 10-15 characters')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate that password meets security requirements"""
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        return v


class SignupResponse(BaseModel):
    """Schema for signup response"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User's email address")
    message: str = Field(..., description="Success message")

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Schema for user login request"""
    credential: str = Field(..., description="Email or mobile number")
    password: str = Field(..., description="User's password")


class LoginResponse(BaseModel):
    """Schema for login response"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User's email address")
    mobile_number: str = Field(..., description="User's mobile number")
    message: str = Field(..., description="Success message")

    class Config:
        from_attributes = True
