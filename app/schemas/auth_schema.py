from pydantic import BaseModel, EmailStr, Field, field_validator
import re
from app.core.exceptions import UnprocessableEntityException


class SignupRequest(BaseModel):
    """Schema for user signup request"""
    email: EmailStr = Field(..., description="User's email address")
    mobile_number: str = Field(default="9999999999", description="User's mobile number", min_length=10, max_length=10)
    password: str = Field(..., description="User's password", min_length=6)

    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        """Validate that mobile number contains only digits"""
        if not re.match(r'^\d{10,15}$', v):
            raise UnprocessableEntityException(
                'Invalid mobile number',
                ['Mobile number must contain only digits and be between 10-15 characters']
            )
        return v

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

    @field_validator('credential')
    @classmethod
    def validate_credential(cls, v):
        """Validate that credential is not empty"""
        if not v or not v.strip():
            raise UnprocessableEntityException(
                'Invalid credential',
                ['Email or mobile number cannot be empty']
            )
        return v

    @field_validator('password')
    @classmethod
    def validate_login_password(cls, v):
        """Validate that password is not empty"""
        if not v or not v.strip():
            raise UnprocessableEntityException(
                'Invalid password',
                ['Password cannot be empty']
            )
        return v


class LoginResponse(BaseModel):
    """Schema for login response"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User's email address")
    mobile_number: str = Field(..., description="User's mobile number")
    token: str = Field(..., description="JWT access token")

    class Config:
        from_attributes = True
