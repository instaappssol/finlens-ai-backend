"""
Transaction Categorization Service

Wraps the transaction categorization model for use in the API.
Handles model loading, prediction, and training.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from app.services.model_manager import ModelManager

# Import the ML model from app.ml
from app.ml.transaction_categorization_engine import (
    TransactionCategorizationEngine,
    MerchantKnowledgeBase,
)


class CategorizationService:
    """Service for transaction categorization using ML models"""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        db=None,
    ):
        """
        Initialize categorization service.

        Args:
            model_manager: Optional ModelManager instance
            db: MongoDB database connection for GridFS storage (required)
        """
        if model_manager is None:
            if db is None:
                raise ValueError("MongoDB database connection is required for GridFS storage")
            self.model_manager = ModelManager(
                db=db, use_gridfs=True
            )
        else:
            self.model_manager = model_manager
        self.model: Optional[TransactionCategorizationEngine] = None
        self.model_name = "transaction_categorizer"
        self.model_loaded = False
        self.db = db  # Store DB reference for loading merchant KB
        
        # Create default empty merchant KB (will be loaded from MongoDB if available)
        self.default_merchant_kb = MerchantKnowledgeBase(
            merchants=[],
            corrections=None
        )
        
        # Try to load merchant KB from MongoDB
        if db is not None:
            try:
                self.default_merchant_kb = MerchantKnowledgeBase.from_mongodb(
                    db, collection_name="merchant_knowledge_base"
                )
            except Exception as e:
                print(f"[WARN] Could not load merchant KB from MongoDB: {e}. Using empty KB.")

    def load_model(self, version: Optional[str] = None) -> bool:
        """
        Load the categorization model from GridFS.

        Args:
            version: Model version to load (default: latest)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load model from GridFS (joblib serialized engine)
            self.model = self.model_manager.load_model(
                self.model_name, version=version, use_joblib=True
            )
            
            if isinstance(self.model, TransactionCategorizationEngine):
                self.model_loaded = True
                return True
            else:
                raise ValueError(f"Unexpected model data type: {type(self.model)}")
        except FileNotFoundError as e:
            print(
                f"[INFO] Model file not found: {e}. Train a model first using /admin/train-model endpoint."
            )
            self.model_loaded = False
            return False
        except ValueError as e:
            # Model not found in metadata - this is expected if no model has been trained yet
            if "not found in metadata" in str(e):
                print(f"[INFO] No model found. Train a model first using /admin/train-model endpoint.")
            else:
                print(f"[WARN] Error loading model: {e}")
            self.model_loaded = False
            return False
        except Exception as e:
            print(f"[WARN] Error loading model: {e}")
            self.model_loaded = False
            return False

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict category for a single transaction.

        Args:
            transaction: Transaction dict with fields:
                - transaction_id (optional)
                - description (required)
                - amount (required)
                - transaction_type (optional)
                - currency (optional)
                - timestamp (optional)

        Returns:
            Prediction result dict with category, confidence, etc.
        """
        if not self.model_loaded:
            # Try to load model if not loaded
            if not self.load_model():
                raise RuntimeError(
                    "Model not loaded. Train a model first or ensure model file exists."
                )

        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Ensure required fields
        if "description" not in transaction:
            raise ValueError("Transaction must have 'description' field")

        if "amount" not in transaction:
            raise ValueError("Transaction must have 'amount' field")

        # Set defaults
        transaction.setdefault("transaction_type", transaction.get("type", "UNKNOWN"))
        transaction.setdefault("timestamp", transaction.get("timestamp", ""))
        
        # Convert to format expected by engine
        engine_txn = {
            "description": transaction["description"],
            "amount": float(transaction["amount"]),
            "timestamp": transaction.get("timestamp", ""),
            "type": transaction.get("transaction_type", "UNKNOWN"),
        }
        
        # Predict using batch method (engine uses predict_batch)
        results = self.model.predict_batch([engine_txn])
        result = results[0]

        return {
            "transaction_id": transaction.get("transaction_id"),
            "category": result["predicted_category"],
            "confidence_score": result["prediction_confidence"],
            "model_version": "v1.0.0",
        }

    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple transactions.

        Args:
            transactions: List of transaction dicts

        Returns:
            List of prediction results
        """
        return [self.predict(txn) for txn in transactions]

    def train_model(
        self,
        training_data: List[Dict[str, Any]],
        labels: List[str],
        save_model: bool = True,
        version: str = "latest",
        merchant_kb: Optional[MerchantKnowledgeBase] = None,
    ) -> Dict[str, Any]:
        """
        Train a new categorization model from data.

        Args:
            training_data: List of transaction dicts with 'label' field
            labels: List of category labels (can be extracted from training_data)
            save_model: Whether to save the trained model
            version: Version string for the saved model
            merchant_kb: Optional MerchantKnowledgeBase (uses default empty if not provided)

        Returns:
            Training results dict with metrics
        """
        # Create DataFrame
        df = pd.DataFrame(training_data)

        # Ensure labels are in DataFrame
        if "label" not in df.columns and labels:
            df["label"] = labels
        elif "label" not in df.columns:
            raise ValueError(
                "Training data must include 'label' field or provide labels list"
            )

        # Use provided merchant KB, try to load from MongoDB, or use default empty one
        if merchant_kb is not None:
            kb = merchant_kb
        elif self.db is not None:
            try:
                kb = MerchantKnowledgeBase.from_mongodb(
                    self.db, collection_name="merchant_knowledge_base"
                )
            except Exception as e:
                print(f"[WARN] Could not load merchant KB from MongoDB: {e}. Using default.")
                kb = self.default_merchant_kb
        else:
            kb = self.default_merchant_kb

        # Prepare DataFrame for engine (map columns)
        engine_df = df.copy()
        engine_df["category"] = engine_df["label"]  # Engine expects 'category' column
        if "timestamp" not in engine_df.columns:
            engine_df["timestamp"] = ""
        if "type" not in engine_df.columns:
            engine_df["type"] = engine_df.get("transaction_type", "UNKNOWN")

        # Train model using engine
        self.model = TransactionCategorizationEngine.from_training_data(
            df=engine_df,
            merchant_kb=kb,
            label_col="category",
            description_col="description",
            amount_col="amount",
            timestamp_col="timestamp" if "timestamp" in engine_df.columns else None,
            type_col="type" if "type" in engine_df.columns else None,
            num_epochs=10,
            batch_size=32,
        )

        # Save model if requested
        if save_model:
            metadata = {
                "training_samples": len(training_data),
                "categories": df["label"].unique().tolist(),
                "training_date": pd.Timestamp.now().isoformat(),
            }
            
            # Save engine directly to GridFS using joblib
            self.model_manager.save_model(
                self.model,
                self.model_name,
                version=version,
                metadata=metadata,
                use_joblib=True,  # Use joblib (handles threading objects better)
            )

        self.model_loaded = True

        return {
            "status": "success",
            "model_version": version,
            "training_samples": len(training_data),
            "categories": df["label"].unique().tolist(),
        }

    def explain_prediction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get explanation for a prediction (feature importance, etc.).

        Args:
            transaction: Transaction dict

        Returns:
            Explanation dict with top factors
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # Get prediction first
        pred_result = self.predict(transaction)
        
        # Convert to engine format
        engine_txn = {
            "description": transaction.get("description", ""),
            "amount": float(transaction.get("amount", 0)),
            "timestamp": transaction.get("timestamp", ""),
            "type": transaction.get("transaction_type", "UNKNOWN"),
        }
        
        # Get detailed prediction from engine
        results = self.model.predict_batch([engine_txn])
        result = results[0]
        
        # Return explanation with available info
        return {
            "transaction_id": transaction.get("transaction_id"),
            "predicted_category": pred_result["category"],
            "confidence_score": pred_result["confidence_score"],
            "top_factors": [
                {"feature": "normalized_merchant", "contribution": result.get("merchant_confidence", 0.0)},
                {"feature": "kb_category", "contribution": 0.8 if result.get("kb_category") else 0.0},
            ],
            "model_version": pred_result["model_version"],
            "normalized_merchant": result.get("normalized_merchant"),
            "kb_category": result.get("kb_category"),
            "merchant_confidence": result.get("merchant_confidence"),
        }

    def load_user_model(self, user_id: str, version: Optional[str] = None) -> bool:
        """
        Load user-specific model if available, fallback to global model.

        Args:
            user_id: User ID for user-specific model
            version: Model version to load

        Returns:
            True if loaded successfully, False otherwise
        """
        user_model_name = f"transaction_categorizer_user_{user_id}"

        try:
            # Try to load user-specific model
            self.model = self.model_manager.load_model(
                user_model_name, version=version, use_joblib=True
            )
            
            if isinstance(self.model, TransactionCategorizationEngine):
                self.model_loaded = True
                self.model_name = user_model_name
                return True
            else:
                raise ValueError(f"Unexpected model data type: {type(self.model)}")
        except Exception:
            # Fallback to global model
            return self.load_model()

    def train_user_model(
        self,
        user_id: str,
        training_data: List[Dict[str, Any]],
        labels: List[str],
        save_model: bool = True,
        version: str = "latest",
        merchant_kb: Optional[MerchantKnowledgeBase] = None,
    ) -> Dict[str, Any]:
        """
        Train a user-specific model.

        Args:
            user_id: User ID for user-specific model
            training_data: List of transaction dicts with 'label' field
            labels: List of category labels
            save_model: Whether to save the trained model
            version: Version string for the saved model
            merchant_kb: Optional MerchantKnowledgeBase (uses default empty if not provided)

        Returns:
            Training results dict with metrics
        """
        # Create DataFrame
        df = pd.DataFrame(training_data)

        # Ensure labels are in DataFrame
        if "label" not in df.columns and labels:
            df["label"] = labels
        elif "label" not in df.columns:
            raise ValueError(
                "Training data must include 'label' field or provide labels list"
            )

        # Use provided merchant KB, try to load from MongoDB, or use default empty one
        if merchant_kb is not None:
            kb = merchant_kb
        elif self.db is not None:
            try:
                kb = MerchantKnowledgeBase.from_mongodb(
                    self.db, collection_name="merchant_knowledge_base"
                )
            except Exception as e:
                print(f"[WARN] Could not load merchant KB from MongoDB: {e}. Using default.")
                kb = self.default_merchant_kb
        else:
            kb = self.default_merchant_kb

        # Prepare DataFrame for engine
        engine_df = df.copy()
        engine_df["category"] = engine_df["label"]
        if "timestamp" not in engine_df.columns:
            engine_df["timestamp"] = ""
        if "type" not in engine_df.columns:
            engine_df["type"] = engine_df.get("transaction_type", "UNKNOWN")

        # Train model using engine
        self.model = TransactionCategorizationEngine.from_training_data(
            df=engine_df,
            merchant_kb=kb,
            label_col="category",
            description_col="description",
            amount_col="amount",
            timestamp_col="timestamp" if "timestamp" in engine_df.columns else None,
            type_col="type" if "type" in engine_df.columns else None,
            num_epochs=10,
            batch_size=32,
        )

        if save_model:
            user_model_name = f"transaction_categorizer_user_{user_id}"
            
            metadata = {
                "user_id": user_id,
                "training_samples": len(training_data),
                "categories": df["label"].unique().tolist(),
                "training_date": pd.Timestamp.now().isoformat(),
            }
            
            # Save engine directly to GridFS using joblib
            self.model_manager.save_model(
                self.model,
                user_model_name,
                version=version,
                metadata=metadata,
                use_joblib=True,  # Use joblib (handles threading objects better)
            )
            self.model_name = user_model_name

        self.model_loaded = True

        return {
            "status": "success",
            "model_type": "user_specific",
            "user_id": user_id,
            "model_version": version,
            "training_samples": len(training_data),
            "categories": df["label"].unique().tolist(),
        }

