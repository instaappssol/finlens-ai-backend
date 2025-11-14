"""
Common response models and exceptions for API
"""
from typing import Any, Optional, List
from pydantic import BaseModel, Field


class ResponseBody(BaseModel):
    """Common API response structure"""
    message: str = Field(..., description="Response message")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    data: Optional[Any] = Field(default=None, description="Response data")

    class Config:
        from_attributes = True


class UnprocessableEntityException(Exception):
    """Exception for unprocessable entity (422)"""
    def __init__(self, message: str, errors: List[str] = []):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)


class BadRequestException(Exception):
    """Exception for bad request (400)"""
    def __init__(self, message: str, errors: List[str] = []):
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)


class InternalServerErrorException(Exception):
    """Exception for internal server error (500)"""
    def __init__(self, message: str, errors: List[str] = []):
        print(message)
        self.message = "Sorry something went wrong"
        self.errors = errors or []
        super().__init__(self.message)
