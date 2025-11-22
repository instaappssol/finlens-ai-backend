"""Admin repository interface following clean architecture"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AdminRepository(ABC):
    """Abstract repository interface for admin data access"""

    @abstractmethod
    def create(self, admin_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new admin.

        Args:
            admin_data: Dictionary containing admin data

        Returns:
            Created admin dictionary with _id
        """
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Find admin by email address.

        Args:
            email: Admin's email address

        Returns:
            Admin dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def find_by_id(self, admin_id: str) -> Optional[Dict[str, Any]]:
        """
        Find admin by ID.

        Args:
            admin_id: Admin's ID

        Returns:
            Admin dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, admin_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update admin data.

        Args:
            admin_id: Admin's ID
            update_data: Dictionary with fields to update

        Returns:
            Updated admin dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, admin_id: str) -> bool:
        """
        Delete an admin.

        Args:
            admin_id: Admin's ID

        Returns:
            True if deleted, False otherwise
        """
        pass

