import hashlib
from datetime import datetime
from typing import Optional

from app.repositories.user_repository import UserRepository


class AuthService:
    """Service for authentication operations following clean architecture"""

    def __init__(self, user_repository: UserRepository):
        """
        Initialize authentication service.

        Args:
            user_repository: Repository for user data access
        """
        self.user_repository = user_repository

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def signup(self, email: str, mobile_number: str, password: str) -> dict:
        """
        Register a new user
        
        Args:
            email: User's email address
            mobile_number: User's mobile number
            password: User's password (will be hashed)
            
        Returns:
            dict: Created user data
            
        Raises:
            ValueError: If email or mobile number already exists
        """
        # Check if email already exists
        existing_email = self.user_repository.find_by_email(email.lower())
        if existing_email:
            raise ValueError(f"Email '{email}' is already registered")

        # Check if mobile number already exists
        existing_mobile = self.user_repository.find_by_mobile(mobile_number)
        if existing_mobile:
            raise ValueError(f"Mobile number '{mobile_number}' is already registered")

        # Hash password
        password_hash = self.hash_password(password)

        # Create user document
        user_doc = {
            'email': email.lower(),
            'mobile_number': mobile_number,
            'password_hash': password_hash,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'is_active': True
        }

        try:
            return self.user_repository.create(user_doc)
        except ValueError as e:
            # Repository may raise ValueError for duplicates
            raise ValueError(f"User with this email or mobile number already exists: {str(e)}")

    def login(self, credential: str, password: str) -> dict:
        """
        Authenticate user by email or mobile number
        
        Args:
            credential: Email or mobile number
            password: User's password
            
        Returns:
            dict: User data if authentication successful
            
        Raises:
            ValueError: If user not found or password is incorrect
        """
        # Find user by email or mobile number
        user = self.user_repository.find_by_credential(credential)

        if not user:
            raise ValueError("Invalid email/mobile number or password")

        # Verify password
        if not self.verify_password(password, user['password_hash']):
            raise ValueError("Invalid email/mobile number or password")

        # Check if user is active
        if not user.get('is_active', True):
            raise ValueError("User account is inactive")

        return user

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email"""
        return self.user_repository.find_by_email(email)

    def get_user_by_mobile(self, mobile_number: str) -> Optional[dict]:
        """Get user by mobile number"""
        return self.user_repository.find_by_mobile(mobile_number)

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        return self.user_repository.find_by_id(user_id)

    def update_user(self, user_id: str, update_data: dict) -> Optional[dict]:
        """Update user data"""
        update_data['updated_at'] = datetime.utcnow()
        return self.user_repository.update(user_id, update_data)

    def delete_user(self, user_id: str) -> bool:
        """Delete user account"""
        return self.user_repository.delete(user_id)
