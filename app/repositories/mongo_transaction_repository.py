"""MongoDB implementation of TransactionRepository"""

from typing import List, Dict, Optional, Any
from collections import Counter
from pymongo.database import Database
from bson import ObjectId

from app.repositories.transaction_repository import TransactionRepository


class MongoTransactionRepository(TransactionRepository):
    """MongoDB implementation of TransactionRepository"""

    def __init__(self, db: Database):
        """
        Initialize MongoDB transaction repository.

        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db["transactions"]

    def insert_many(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple transactions into MongoDB"""
        if not transactions:
            return []

        result = self.collection.insert_many(transactions)
        return [str(id) for id in result.inserted_ids]

    def find_labeled_transactions(
        self,
        user_id: Optional[str] = None,
        exclude_auto_categorized: bool = True
    ) -> List[Dict[str, Any]]:
        """Find transactions that have category labels"""
        query = {
            "category": {"$exists": True, "$ne": None, "$ne": ""},
        }

        if exclude_auto_categorized:
            query["category_source"] = {"$ne": "auto_ml"}

        if user_id:
            query["user_id"] = user_id

        transactions = list(self.collection.find(query))
        return transactions

    def get_training_data_stats(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about available training data"""
        labeled = self.find_labeled_transactions(user_id)

        if not labeled:
            return {
                "total_labeled": 0,
                "categories": {},
                "can_train_global": False,
                "can_train_user": False,
            }

        # Count by category
        category_counts = {}
        for txn in labeled:
            cat = txn.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        total = len(labeled)
        min_samples_per_category = min(category_counts.values()) if category_counts else 0

        return {
            "total_labeled": total,
            "categories": category_counts,
            "min_samples_per_category": min_samples_per_category,
            "can_train_global": total >= 50 and min_samples_per_category >= 2,
            "can_train_user": user_id is not None and total >= 20 and min_samples_per_category >= 2,
        }

