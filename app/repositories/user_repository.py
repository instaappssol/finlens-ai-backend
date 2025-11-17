"""User repository interface following clean architecture"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class UserRepository(ABC):
    """Abstract repository interface for user data access"""

    @abstractmethod
    def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            user_data: Dictionary containing user data

        Returns:
            Created user dictionary with _id
        """
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Find user by email address.

        Args:
            email: User's email address

        Returns:
            User dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def find_by_mobile(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """
        Find user by mobile number.

        Args:
            mobile_number: User's mobile number

        Returns:
            User dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def find_by_credential(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Find user by email or mobile number.

        Args:
            credential: Email address or mobile number

        Returns:
            User dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Find user by ID.

        Args:
            user_id: User's ID

        Returns:
            User dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, user_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update user data.

        Args:
            user_id: User's ID
            update_data: Dictionary with fields to update

        Returns:
            Updated user dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        """
        Delete a user.

        Args:
            user_id: User's ID

        Returns:
            True if deleted, False otherwise
        """
        pass

