import hashlib
from datetime import datetime
from typing import Optional

from app.repositories.admin_repository import AdminRepository


class AdminAuthService:
    """Service for admin authentication operations following clean architecture"""

    def __init__(self, admin_repository: AdminRepository):
        """
        Initialize admin authentication service.

        Args:
            admin_repository: Repository for admin data access
        """
        self.admin_repository = admin_repository

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def signup(self, name: str, email: str, password: str) -> dict:
        """
        Register a new admin
        
        Args:
            name: Admin's full name
            email: Admin's email address
            password: Admin's password (will be hashed)
            
        Returns:
            dict: Created admin data
            
        Raises:
            ValueError: If email already exists
        """
        # Check if email already exists
        existing_email = self.admin_repository.find_by_email(email.lower())
        if existing_email:
            raise ValueError(f"Email '{email}' is already registered")

        # Hash password
        password_hash = self.hash_password(password)

        # Create admin document
        admin_doc = {
            'name': name.strip(),
            'email': email.lower(),
            'password_hash': password_hash,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'is_active': True
        }

        try:
            return self.admin_repository.create(admin_doc)
        except ValueError as e:
            # Repository may raise ValueError for duplicates
            raise ValueError(f"Admin with this email already exists: {str(e)}")

    def login(self, email: str, password: str) -> dict:
        """
        Authenticate admin by email
        
        Args:
            email: Admin's email address
            password: Admin's password
            
        Returns:
            dict: Admin data if authentication successful
            
        Raises:
            ValueError: If admin not found or password is incorrect
        """
        # Find admin by email
        admin = self.admin_repository.find_by_email(email.lower())

        if not admin:
            raise ValueError("Invalid email or password")

        # Verify password
        if not self.verify_password(password, admin['password_hash']):
            raise ValueError("Invalid email or password")

        # Check if admin is active
        if not admin.get('is_active', True):
            raise ValueError("Admin account is inactive")

        return admin

    def get_admin_by_email(self, email: str) -> Optional[dict]:
        """Get admin by email"""
        return self.admin_repository.find_by_email(email)

    def get_admin_by_id(self, admin_id: str) -> Optional[dict]:
        """Get admin by ID"""
        return self.admin_repository.find_by_id(admin_id)

    def update_admin(self, admin_id: str, update_data: dict) -> Optional[dict]:
        """Update admin data"""
        update_data['updated_at'] = datetime.utcnow()
        return self.admin_repository.update(admin_id, update_data)

    def delete_admin(self, admin_id: str) -> bool:
        """Delete admin account"""
        return self.admin_repository.delete(admin_id)

