import hashlib
from datetime import datetime
from typing import Optional
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from bson import ObjectId


class AuthService:
    """Service for authentication operations with MongoDB"""

    def __init__(self, db: Database):
        self.db = db
        self.users_collection = db['users']
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            self.users_collection.create_index('email', unique=True)
            self.users_collection.create_index('mobile_number', unique=True)
        except Exception as e:
            print(f"Index creation warning: {e}")

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
        existing_email = self.users_collection.find_one({'email': email.lower()})
        if existing_email:
            raise ValueError(f"Email '{email}' is already registered")

        # Check if mobile number already exists
        existing_mobile = self.users_collection.find_one({'mobile_number': mobile_number})
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
            result = self.users_collection.insert_one(user_doc)
            user_doc['_id'] = result.inserted_id
            return user_doc
        except DuplicateKeyError as e:
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
        user = self.users_collection.find_one({
            '$or': [
                {'email': credential.lower()},
                {'mobile_number': credential}
            ]
        })

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
        return self.users_collection.find_one({'email': email.lower()})

    def get_user_by_mobile(self, mobile_number: str) -> Optional[dict]:
        """Get user by mobile number"""
        return self.users_collection.find_one({'mobile_number': mobile_number})

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        try:
            return self.users_collection.find_one({'_id': ObjectId(user_id)})
        except Exception:
            return None

    def update_user(self, user_id: str, update_data: dict) -> dict:
        """Update user data"""
        update_data['updated_at'] = datetime.utcnow()
        result = self.users_collection.find_one_and_update(
            {'_id': ObjectId(user_id)},
            {'$set': update_data},
            return_document=True
        )
        return result

    def delete_user(self, user_id: str) -> bool:
        """Delete user account"""
        result = self.users_collection.delete_one({'_id': ObjectId(user_id)})
        return result.deleted_count > 0
