"""MongoDB implementation of AdminRepository"""

from typing import Optional, Dict, Any
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from app.repositories.admin_repository import AdminRepository


class MongoAdminRepository(AdminRepository):
    """MongoDB implementation of AdminRepository"""

    def __init__(self, db: Database):
        """
        Initialize MongoDB admin repository.

        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db["admins"]
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            self.collection.create_index("email", unique=True)
        except Exception as e:
            print(f"Index creation warning: {e}")

    def create(self, admin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new admin in MongoDB"""
        try:
            result = self.collection.insert_one(admin_data)
            admin_data["_id"] = result.inserted_id
            return admin_data
        except DuplicateKeyError as e:
            raise ValueError(f"Admin with this email already exists: {str(e)}") from e

    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find admin by email address"""
        return self.collection.find_one({"email": email.lower()})

    def find_by_id(self, admin_id: str) -> Optional[Dict[str, Any]]:
        """Find admin by ID"""
        try:
            return self.collection.find_one({"_id": ObjectId(admin_id)})
        except Exception:
            return None

    def update(self, admin_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update admin data"""
        try:
            result = self.collection.find_one_and_update(
                {"_id": ObjectId(admin_id)},
                {"$set": update_data},
                return_document=True
            )
            return result
        except Exception:
            return None

    def delete(self, admin_id: str) -> bool:
        """Delete an admin"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(admin_id)})
            return result.deleted_count > 0
        except Exception:
            return False

