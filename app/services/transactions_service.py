import csv
import io
import ast
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from bson import ObjectId
import pandas as pd

from app.repositories.transaction_repository import TransactionRepository


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
    """Service for transaction operations following clean architecture"""

    def __init__(
        self,
        transaction_repository: TransactionRepository,
        categorization_service=None
    ):
        """
        Initialize transaction service.

        Args:
            transaction_repository: Repository for transaction data access
            categorization_service: Optional service for transaction categorization
        """
        self.transaction_repository = transaction_repository
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

                # Ensure transaction_type is set to DEBIT or CREDIT if not provided
                # Default to DEBIT (expense) if not specified
                if not row.get("transaction_type") or row.get("transaction_type").upper() not in ["DEBIT", "CREDIT"]:
                    # If transaction_type exists but is not DEBIT/CREDIT, keep it as is
                    # Otherwise default to DEBIT
                    if not row.get("transaction_type"):
                        row["transaction_type"] = "DEBIT"

                # Check if category already exists
                has_category = row.get("category") and row.get("category").strip()

                # Auto-categorize if enabled and model available
                if auto_categorize and self.categorization_service and not has_category:
                    try:
                        # First, check for user feedback (corrections)
                        description = row.get("description", "")
                        feedback_category = self.transaction_repository.get_user_feedback(
                            description=description,
                            user_id=user_id
                        )
                        
                        # Get amount, handling string or number (needed for both feedback and model)
                        amount = row.get("amount", 0)
                        try:
                            amount = float(amount)
                        except (ValueError, TypeError):
                            amount = 0.0

                        # Prepare transaction_data for explanation (needed in both cases)
                        transaction_data = {
                            "description": description,
                            "amount": amount,
                        }

                        if feedback_category:
                            # Use user feedback category
                            row["category"] = feedback_category
                            row["category_source"] = "user_feedback"
                            categorized_count += 1
                        else:
                            # No feedback found, use model prediction
                            # Check if model is loaded
                            if not self.categorization_service.model_loaded:
                                # Try to load model
                                if not self.categorization_service.load_model():
                                    raise RuntimeError("Model not available. Please train a model first.")
                            
                            # Get prediction
                            prediction = self.categorization_service.predict(transaction_data)
                            row["category"] = prediction["category"]
                        
                        # Get XAI explanation (only if model is available)
                        try:
                            if self.categorization_service.model_loaded:
                                explanation = self.categorization_service.explain_prediction(transaction_data)
                            else:
                                # If no model, create a simple explanation
                                explanation = {
                                    "transaction_id": row.get("transaction_id"),
                                    "predicted_category": row.get("category"),
                                    "confidence_score": 1.0 if feedback_category else 0.0,
                                    "top_factors": [
                                        {"feature": "user_feedback", "contribution": 1.0} if feedback_category else {}
                                    ],
                                    "model_version": "user_feedback" if feedback_category else "unknown",
                                    "normalized_merchant": description,
                                    "kb_category": feedback_category if feedback_category else None,
                                    "merchant_confidence": 1.0 if feedback_category else 0.0,
                                }
                            row["explanation"] = explanation
                        except Exception as explain_error:
                            # Log but don't fail if explanation fails
                            # Still save error information in explanation
                            error_msg = str(explain_error)
                            print(f"Failed to get explanation for transaction '{row.get('description', 'unknown')}': {error_msg}")
                            row["explanation"] = {
                                "transaction_id": row.get("transaction_id"),
                                "predicted_category": row.get("category"),
                                "confidence_score": 0.0,
                                "top_factors": [],
                                "model_version": "unknown",
                                "error": f"Exception during explanation: {error_msg}",
                            }
                        
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

            # Insert into database via repository
            inserted_ids = self.transaction_repository.insert_many(rows)
            inserted_count = len(inserted_ids)

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
            min_samples: Minimum samples required per category (not used in query, kept for compatibility)

        Returns:
            List of labeled transactions
        """
        return self.transaction_repository.find_labeled_transactions(
            user_id=user_id,
            exclude_auto_categorized=True
        )

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

    def get_analytics_summary(
        self,
        year: int,
        month: int,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Get analytics summary for a specific month and year.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with analytics data
        """
        return self.transaction_repository.get_analytics_summary(year, month, user_id)

    def get_transactions_by_category(
        self,
        year: int,
        month: int,
        category: str,
        user_id: Optional[str] = None
    ) -> List[Dict]:
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
        transactions = self.transaction_repository.get_transactions_by_category(
            year, month, category, user_id
        )
        # Serialize for JSON response
        return [serialize_for_json(txn) for txn in transactions]

    def upload_merchant_mappings(
        self,
        db,
        file_content: bytes,
        replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Upload merchant mappings CSV to MongoDB collection merchant_knowledge_base.
        
        Args:
            db: MongoDB database connection
            file_content: Raw bytes of uploaded CSV file
            replace_existing: Whether to replace existing mappings (default: append)
            
        Returns:
            Dictionary with upload statistics
            
        Raises:
            ValueError: If CSV parsing fails or required columns are missing
        """
        collection = db["merchant_knowledge_base"]
        
        # Read and parse CSV
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Validate required columns
        if "Category" not in df.columns or "Merchants" not in df.columns:
            raise ValueError("CSV must contain 'Category' and 'Merchants' columns")
        
        # Replace existing if requested
        if replace_existing:
            collection.delete_many({})
        
        inserted_count = 0
        
        # Parse each row
        for _, row in df.iterrows():
            category = str(row["Category"]).strip()
            raw_merchants = row["Merchants"]
            
            if pd.isna(raw_merchants) or not category:
                continue
            
            # Parse merchants list (can be Python list string or comma-separated)
            if isinstance(raw_merchants, str) and raw_merchants.strip().startswith("["):
                # Python-list-like string
                try:
                    merchant_list = ast.literal_eval(raw_merchants)
                except Exception:
                    merchant_list = [m.strip().strip('"').strip("'") for m in raw_merchants.split(",") if m.strip()]
            elif isinstance(raw_merchants, str):
                merchant_list = [m.strip().strip('"').strip("'") for m in raw_merchants.split(",") if m.strip()]
            else:
                merchant_list = [str(raw_merchants).strip()]
            
            # Remove empty strings
            merchant_list = [m for m in merchant_list if m]
            
            if not merchant_list:
                continue
            
            # Main merchant is the first one
            main_merchant = merchant_list[0]
            aliases = merchant_list[1:] if len(merchant_list) > 1 else []
            
            # Check if merchant already exists
            existing = collection.find_one({"merchant": main_merchant, "category": category})
            
            if existing:
                # Update with aliases
                collection.update_one(
                    {"merchant": main_merchant, "category": category},
                    {"$set": {"merchants": aliases}}
                )
            else:
                # Insert new document
                collection.insert_one({
                    "merchant": main_merchant,
                    "category": category,
                    "merchants": aliases
                })
                inserted_count += 1
        
        return {
            "inserted_count": inserted_count,
            "total_rows_processed": len(df),
            "collection": "merchant_knowledge_base"
        }

    def upload_merchant_corrections(
        self,
        db,
        file_content: bytes,
        replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Upload merchant corrections CSV to MongoDB collection merchant_corrections.
        
        Args:
            db: MongoDB database connection
            file_content: Raw bytes of uploaded CSV file
            replace_existing: Whether to replace existing corrections (default: append)
            
        Returns:
            Dictionary with upload statistics
            
        Raises:
            ValueError: If CSV parsing fails or required columns are missing
        """
        collection = db["merchant_corrections"]
        
        # Read and parse CSV
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Validate required columns
        required_cols = {"raw_description", "canonical_merchant", "canonical_category"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
        
        # Replace existing if requested
        if replace_existing:
            collection.delete_many({})
        
        inserted_count = 0
        
        # Parse each row
        for _, row in df.iterrows():
            raw_desc = str(row["raw_description"]).strip()
            canon_merchant = str(row["canonical_merchant"]).strip()
            canon_category = str(row["canonical_category"]).strip()
            
            if not raw_desc or not canon_merchant or not canon_category:
                continue
            
            # Check if correction already exists
            existing = collection.find_one({"raw_description": raw_desc})
            
            if existing:
                # Update existing correction
                collection.update_one(
                    {"raw_description": raw_desc},
                    {
                        "$set": {
                            "canonical_merchant": canon_merchant,
                            "canonical_category": canon_category
                        }
                    }
                )
            else:
                # Insert new correction
                collection.insert_one({
                    "raw_description": raw_desc,
                    "canonical_merchant": canon_merchant,
                    "canonical_category": canon_category
                })
                inserted_count += 1
        
        return {
            "inserted_count": inserted_count,
            "total_rows_processed": len(df),
            "collection": "merchant_corrections"
        }

    def store_user_feedback(
        self,
        transaction_id: str,
        category: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store user feedback for future categorizations (does not update transaction).

        Args:
            transaction_id: Transaction ID to get description from
            category: Category feedback from user
            user_id: User ID who provided the feedback

        Returns:
            Dictionary with feedback result
        """
        if not category or not category.strip():
            raise ValueError("Category cannot be empty")

        success = self.transaction_repository.store_user_feedback(
            transaction_id=transaction_id,
            category=category.strip(),
            user_id=user_id
        )

        if not success:
            raise ValueError(f"Transaction with ID {transaction_id} not found or feedback storage failed")

        return {
            "transaction_id": transaction_id,
            "category": category.strip(),
            "status": "success",
            "message": "Feedback stored successfully. This will be used for future categorizations."
        }
