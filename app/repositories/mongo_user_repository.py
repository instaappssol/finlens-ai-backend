"""MongoDB implementation of UserRepository"""

from typing import Optional, Dict, Any
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from app.repositories.user_repository import UserRepository


class MongoUserRepository(UserRepository):
    """MongoDB implementation of UserRepository"""

    def __init__(self, db: Database):
        """
        Initialize MongoDB user repository.

        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db["users"]
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            self.collection.create_index("email", unique=True)
            self.collection.create_index("mobile_number", unique=True)
        except Exception as e:
            print(f"Index creation warning: {e}")

    def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in MongoDB"""
        try:
            result = self.collection.insert_one(user_data)
            user_data["_id"] = result.inserted_id
            return user_data
        except DuplicateKeyError as e:
            raise ValueError(f"User with this email or mobile number already exists: {str(e)}") from e

    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user by email address"""
        return self.collection.find_one({"email": email.lower()})

    def find_by_mobile(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """Find user by mobile number"""
        return self.collection.find_one({"mobile_number": mobile_number})

    def find_by_credential(self, credential: str) -> Optional[Dict[str, Any]]:
        """Find user by email or mobile number"""
        return self.collection.find_one({
            "$or": [
                {"email": credential.lower()},
                {"mobile_number": credential}
            ]
        })

    def find_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find user by ID"""
        try:
            return self.collection.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    def update(self, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user data"""
        try:
            result = self.collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {"$set": update_data},
                return_document=True
            )
            return result
        except Exception:
            return None

    def delete(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(user_id)})
            return result.deleted_count > 0
        except Exception:
            return False

