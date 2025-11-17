"""Transaction repository interface following clean architecture"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class TransactionRepository(ABC):
    """Abstract repository interface for transaction data access"""

    @abstractmethod
    def insert_many(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple transactions into the database.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of inserted document IDs
        """
        pass

    @abstractmethod
    def find_labeled_transactions(
        self,
        user_id: Optional[str] = None,
        exclude_auto_categorized: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find transactions that have category labels.

        Args:
            user_id: Optional user ID to filter by
            exclude_auto_categorized: If True, exclude auto-categorized transactions

        Returns:
            List of labeled transactions
        """
        pass

    @abstractmethod
    def get_training_data_stats(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about available training data.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with training data statistics
        """
        pass

    @abstractmethod
    def get_transactions_by_month_year(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all transactions for a specific month and year.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            user_id: Optional user ID to filter by

        Returns:
            List of transactions
        """
        pass

    @abstractmethod
    def get_transactions_by_category(
        self,
        year: int,
        month: int,
        category: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all transactions for a specific category in a month and year.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            category: Category name
            user_id: Optional user ID to filter by

        Returns:
            List of transactions
        """
        pass

    @abstractmethod
    def get_analytics_summary(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get analytics summary (inflows, outflows, category breakdowns) for a month.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with analytics data including:
            - total_inflows: Total inflow amount
            - total_outflows: Total outflow amount
            - diff_percentage: Difference percentage
            - inflows_by_category: List of {category, amount, count}
            - outflows_by_category: List of {category, amount, count}
        """
        pass

