import csv
import io
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from pymongo.database import Database
from bson import ObjectId

from app.models.transaction_model import Transaction


def serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert non-serializable objects to JSON-serializable formats.
    Handles datetime, ObjectId, and other types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


class TransactionService:
    """Service for transaction operations with MongoDB"""

    def __init__(self, db: Database, categorization_service=None):
        self.db = db
        self.transactions_collection = db["transactions"]
        self.categorization_service = categorization_service

    def parse_csv_and_insert(
        self, 
        file_content: bytes, 
        user_id: Optional[str] = None,
        auto_categorize: bool = True
    ) -> Tuple[int, List[Dict], Dict]:
        """
        Parse CSV file content, auto-categorize transactions, and insert into DB.

        Args:
            file_content: Raw bytes of uploaded CSV file
            user_id: User ID who uploaded the transactions
            auto_categorize: Whether to automatically categorize transactions

        Returns:
            Tuple of (inserted_count, sample_rows, categorization_stats)
        """
        try:
            # Decode and parse CSV
            text_stream = io.StringIO(file_content.decode("utf-8"))
            reader = csv.DictReader(text_stream)

            rows = []
            categorized_count = 0
            failed_categorizations = 0
            already_categorized = 0

            for row in reader:
                # Add user_id and timestamps
                if user_id:
                    row["user_id"] = user_id
                row["created_at"] = datetime.now()
                row["updated_at"] = datetime.now()

                # Check if category already exists
                has_category = row.get("category") and row.get("category").strip()

                # Auto-categorize if enabled and model available
                if auto_categorize and self.categorization_service and not has_category:
                    try:
                        # Check if model is loaded
                        if not self.categorization_service.model_loaded:
                            # Try to load model
                            if not self.categorization_service.load_model():
                                raise RuntimeError("Model not available. Please train a model first.")
                        
                        # Get amount, handling string or number
                        amount = row.get("amount", 0)
                        try:
                            amount = float(amount)
                        except (ValueError, TypeError):
                            amount = 0.0

                        prediction = self.categorization_service.predict({
                            "description": row.get("description", ""),
                            "amount": amount,
                            "transaction_type": row.get("transaction_type"),
                            "currency": row.get("currency", "INR"),
                            "timestamp": row.get("date") or row.get("timestamp", ""),
                        })
                        row["category"] = prediction["category"]
                        row["category_confidence"] = prediction["confidence_score"]
                        row["category_source"] = "auto_ml"
                        categorized_count += 1
                    except Exception as e:
                        # Log error but don't fail the upload
                        error_msg = str(e)
                        print(f"Failed to categorize transaction '{row.get('description', 'unknown')}': {error_msg}")
                        failed_categorizations += 1
                        row["category_source"] = "failed"
                        row["categorization_error"] = error_msg
                elif has_category:
                    already_categorized += 1
                    row["category_source"] = "manual"

                rows.append(row)

            if not rows:
                return 0, [], {}

            # Insert into MongoDB
            result = self.transactions_collection.insert_many(rows)
            inserted_count = len(result.inserted_ids)

            # Return sample and stats (serialize datetime objects)
            sample = rows[:3]
            # Convert datetime objects to ISO format strings for JSON serialization
            sample = [serialize_for_json(row) for row in sample]
            
            stats = {
                "total_inserted": inserted_count,
                "auto_categorized": categorized_count,
                "failed_categorizations": failed_categorizations,
                "already_categorized": already_categorized
            }

            return inserted_count, sample, stats

        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")

    def get_labeled_transactions(
        self, 
        user_id: Optional[str] = None,
        min_samples: int = 10
    ) -> List[Dict]:
        """
        Get transactions that have category labels (for training).

        Args:
            user_id: If provided, get only this user's transactions
            min_samples: Minimum samples required per category

        Returns:
            List of labeled transactions
        """
        query = {
            "category": {"$exists": True, "$ne": None, "$ne": ""},
            "category_source": {"$ne": "auto_ml"}  # Only user-labeled or manual
        }
        
        if user_id:
            query["user_id"] = user_id

        transactions = list(self.transactions_collection.find(query))
        return transactions

    def parse_csv_for_training(self, file_content: bytes) -> Tuple[List[Dict], List[str]]:
        """
        Parse CSV file content for training data.
        
        Args:
            file_content: Raw bytes of uploaded CSV file
            
        Returns:
            Tuple of (training_transactions, labels)
            
        Raises:
            ValueError: If CSV parsing fails or required fields are missing
        """
        try:
            # Decode and parse CSV (try comma first, then tab)
            text_content = file_content.decode("utf-8")
            
            # Detect delimiter (comma or tab)
            first_line = text_content.split('\n')[0] if '\n' in text_content else text_content
            delimiter = '\t' if '\t' in first_line else ','
            
            text_stream = io.StringIO(text_content)
            reader = csv.DictReader(text_stream, delimiter=delimiter)
            
            training_data = []
            labels = []
            
            for row in reader:
                # Check for required fields
                if not row.get("description"):
                    raise ValueError("CSV must include 'description' field")
                if not row.get("amount"):
                    raise ValueError("CSV must include 'amount' field")
                
                # Check for label/category field (either works)
                label = row.get("label") or row.get("category")
                if not label or not label.strip():
                    raise ValueError(f"CSV must include 'label' or 'category' field. Row: {row.get('description', 'unknown')}")
                
                # Convert amount to float
                try:
                    amount = float(row.get("amount", 0))
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid amount value: {row.get('amount')}")
                
                # Build training transaction dict
                training_txn = {
                    "description": row.get("description", ""),
                    "amount": amount,
                    "transaction_type": row.get("transaction_type", "UNKNOWN"),
                    "currency": row.get("currency", "INR"),
                    "timestamp": row.get("date") or row.get("timestamp", ""),
                    "label": label.strip()
                }
                
                training_data.append(training_txn)
                labels.append(label.strip())
            
            if not training_data:
                raise ValueError("CSV file is empty or has no valid rows")
            
            # Validate that we have at least 2 different categories
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                categories_found = ", ".join(unique_labels) if unique_labels else "none"
                raise ValueError(
                    f"Training requires at least 2 different categories. "
                    f"Found only {len(unique_labels)} category/categories: {categories_found}. "
                    f"Please add transactions with different categories."
                )
            
            # Check minimum samples per category (at least 1 per category, ideally 2+)
            label_counts = Counter(labels)
            min_count = min(label_counts.values())
            if min_count < 1:
                raise ValueError("Each category must have at least 1 sample")
            
            return training_data, labels
            
        except Exception as e:
            raise ValueError(f"Failed to parse CSV for training: {str(e)}")

    def get_training_data_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get statistics about available training data"""
        labeled = self.get_labeled_transactions(user_id)
        
        if not labeled:
            return {
                "total_labeled": 0,
                "categories": {},
                "can_train_global": False,
                "can_train_user": False
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
            "can_train_user": user_id and total >= 20 and min_samples_per_category >= 2
        }
