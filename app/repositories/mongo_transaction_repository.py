"""MongoDB implementation of TransactionRepository"""

from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime
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

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            try:
                # Try ISO format first
                if 'T' in date_value:
                    return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                else:
                    # Try YYYY-MM-DD format
                    return datetime.strptime(date_value, "%Y-%m-%d")
            except (ValueError, AttributeError):
                return None
        
        return None

    def _is_in_month_year(self, date_value: Any, year: int, month: int) -> bool:
        """Check if date falls in the specified month and year"""
        parsed_date = self._parse_date(date_value)
        if parsed_date is None:
            return False
        return parsed_date.year == year and parsed_date.month == month

    def _is_inflow(self, transaction: Dict[str, Any]) -> bool:
        """
        Determine if transaction is an inflow (CREDIT) or outflow (DEBIT).
        
        Uses transaction_type field which has values: DEBIT or CREDIT
        
        Note: CREDIT = money coming in (inflow/income)
              DEBIT = money going out (outflow/expense)
        """
        # Check transaction_type field for DEBIT/CREDIT values
        transaction_type = transaction.get("transaction_type", "")
        if transaction_type:
            transaction_type_upper = str(transaction_type).upper()
            if transaction_type_upper == "CREDIT":
                return True  # CREDIT = inflow (money coming in)
            if transaction_type_upper == "DEBIT":
                return False  # DEBIT = outflow (expense)
        
        # Fallback: If transaction_type is not DEBIT/CREDIT, default to DEBIT (expense)
        return False

    def get_transactions_by_month_year(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all transactions for a specific month and year"""
        query = {}
        
        if user_id:
            query["user_id"] = user_id

        # Get all transactions (we'll filter by date in Python since date formats may vary)
        all_transactions = list(self.collection.find(query))
        
        # Filter by month and year
        filtered = []
        for txn in all_transactions:
            date_value = txn.get("date") or txn.get("timestamp") or txn.get("created_at")
            if self._is_in_month_year(date_value, year, month):
                filtered.append(txn)
        
        return filtered

    def get_transactions_by_category(
        self,
        year: int,
        month: int,
        category: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all transactions for a specific category in a month and year"""
        query = {"category": category}
        
        if user_id:
            query["user_id"] = user_id

        # Get all transactions with this category
        all_transactions = list(self.collection.find(query))
        
        # Filter by month and year
        filtered = []
        for txn in all_transactions:
            date_value = txn.get("date") or txn.get("timestamp") or txn.get("created_at")
            if self._is_in_month_year(date_value, year, month):
                filtered.append(txn)
        
        return filtered

    def get_analytics_summary(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get analytics summary (inflows, outflows, category breakdowns) for a month"""
        transactions = self.get_transactions_by_month_year(year, month, user_id)
        
        total_inflows = 0.0
        total_outflows = 0.0
        inflows_by_category = defaultdict(lambda: {"amount": 0.0, "count": 0})
        outflows_by_category = defaultdict(lambda: {"amount": 0.0, "count": 0})
        
        for txn in transactions:
            amount = txn.get("amount", 0)
            try:
                amount_float = float(amount)
            except (ValueError, TypeError):
                continue
            
            category = txn.get("category", "Uncategorized")
            is_inflow = self._is_inflow(txn)
            
            # Use absolute value for calculations
            abs_amount = abs(amount_float)
            
            if is_inflow:
                total_inflows += abs_amount
                inflows_by_category[category]["amount"] += abs_amount
                inflows_by_category[category]["count"] += 1
            else:
                total_outflows += abs_amount
                outflows_by_category[category]["amount"] += abs_amount
                outflows_by_category[category]["count"] += 1
        
        # Calculate difference percentage
        total = total_inflows + total_outflows
        if total > 0:
            diff_percentage = ((total_inflows - total_outflows) / total) * 100
        else:
            diff_percentage = 0.0
        
        # Convert category dicts to lists with percentage calculations
        inflows_list = []
        for cat, data in inflows_by_category.items():
            percentage = (data["amount"] / total_inflows * 100) if total_inflows > 0 else 0.0
            inflows_list.append({
                "category": cat,
                "amount": round(data["amount"], 2),
                "count": data["count"],
                "percentage": round(percentage, 2)
            })
        
        outflows_list = []
        for cat, data in outflows_by_category.items():
            percentage = (data["amount"] / total_outflows * 100) if total_outflows > 0 else 0.0
            outflows_list.append({
                "category": cat,
                "amount": round(data["amount"], 2),
                "count": data["count"],
                "percentage": round(percentage, 2)
            })
        
        # Sort by amount descending
        inflows_list.sort(key=lambda x: x["amount"], reverse=True)
        outflows_list.sort(key=lambda x: x["amount"], reverse=True)
        
        return {
            "total_inflows": round(total_inflows, 2),
            "total_outflows": round(total_outflows, 2),
            "diff_percentage": round(diff_percentage, 2),
            "inflows_by_category": inflows_list,
            "outflows_by_category": outflows_list,
        }

    def store_user_feedback(
        self,
        transaction_id: str,
        category: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Store user feedback for future categorizations (does not update transaction)"""
        try:
            # Validate transaction_id
            if not ObjectId.is_valid(transaction_id):
                return False

            # Get transaction to extract description for feedback
            transaction = self.collection.find_one({"_id": ObjectId(transaction_id)})
            if not transaction:
                return False

            description = transaction.get("description", "")
            if not description:
                return False  # Need description to store feedback

            # Store user feedback in user_feedback collection
            feedback_collection = self.db["user_feedback"]
            
            # Normalize description for matching (same as merchant corrections)
            normalized_desc = self._normalize_description(description)
            
            # Check if feedback already exists
            existing_feedback = feedback_collection.find_one({
                "normalized_description": normalized_desc,
                "user_id": user_id
            })

            feedback_doc = {
                "normalized_description": normalized_desc,
                "original_description": description,
                "category": category,
                "user_id": user_id,
                "transaction_id": transaction_id,  # Store reference to original transaction
                "updated_at": datetime.now()
            }

            if existing_feedback:
                # Update existing feedback
                feedback_collection.update_one(
                    {"_id": existing_feedback["_id"]},
                    {"$set": feedback_doc}
                )
            else:
                # Insert new feedback
                feedback_doc["created_at"] = datetime.now()
                feedback_collection.insert_one(feedback_doc)

            return True

        except Exception as e:
            print(f"Error storing user feedback: {e}")
            return False

    def get_user_feedback(
        self,
        description: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Get user feedback category for a transaction description"""
        try:
            if not description:
                return None

            feedback_collection = self.db["user_feedback"]
            normalized_desc = self._normalize_description(description)

            # First try to find user-specific feedback
            if user_id:
                feedback = feedback_collection.find_one({
                    "normalized_description": normalized_desc,
                    "user_id": user_id
                })
                if feedback:
                    return feedback.get("category")

            # Fallback to global feedback (user_id is None)
            feedback = feedback_collection.find_one({
                "normalized_description": normalized_desc,
                "user_id": None
            })
            if feedback:
                return feedback.get("category")

            return None

        except Exception as e:
            print(f"Error getting user feedback: {e}")
            return None

    @staticmethod
    def _normalize_description(description: str) -> str:
        """Normalize description for matching (same logic as merchant corrections)"""
        import re
        if not description:
            return ""
        # Convert to lowercase, remove special chars, normalize spaces
        normalized = description.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

