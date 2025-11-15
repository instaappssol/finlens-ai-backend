"""
Transaction Categorization Service

Wraps the transaction categorization model for use in the API.
Handles model loading, prediction, and training.
"""

from typing import Dict, Any, List, Optional
import pandas as pd

from app.services.model_manager import ModelManager

# Import the ML model from app.ml
from app.ml.transaction_categorizer import TransactionCategorizer, PredictionResult


class CategorizationService:
    """Service for transaction categorization using ML models"""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        models_dir: str = "models",
        db=None,
    ):
        """
        Initialize categorization service.

        Args:
            model_manager: Optional ModelManager instance
            models_dir: Directory for model storage (used for local cache)
            db: Optional MongoDB database connection for GridFS storage
        """
        if model_manager is None:
            self.model_manager = ModelManager(
                models_dir=models_dir, db=db, use_gridfs=True
            )
        else:
            self.model_manager = model_manager
        self.model: Optional[TransactionCategorizer] = None
        self.model_name = "transaction_categorizer"
        self.model_loaded = False

    def load_model(self, version: Optional[str] = None) -> bool:
        """
        Load the categorization model from disk.

        Args:
            version: Model version to load (default: latest)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if TransactionCategorizer is None:
                raise ImportError("TransactionCategorizer not available")

            self.model = self.model_manager.load_model(
                self.model_name, version=version, use_joblib=True
            )
            self.model_loaded = True
            return True
        except FileNotFoundError as e:
            print(
                f"Model file not found: {e}. Train a model first using /train-model endpoint."
            )
            self.model_loaded = False
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
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
        transaction.setdefault("transaction_type", "UNKNOWN")
        transaction.setdefault("currency", "INR")
        transaction.setdefault("timestamp", "")

        # Predict
        result: PredictionResult = self.model.predict_one(transaction)

        return {
            "transaction_id": result.transaction_id,
            "category": result.category,
            "confidence_score": result.confidence_score,
            "model_version": result.model_version,
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
    ) -> Dict[str, Any]:
        """
        Train a new categorization model from data.

        Args:
            training_data: List of transaction dicts with 'label' field
            labels: List of category labels (can be extracted from training_data)
            save_model: Whether to save the trained model
            version: Version string for the saved model

        Returns:
            Training results dict with metrics
        """
        if TransactionCategorizer is None:
            raise ImportError("TransactionCategorizer not available")

        # Create DataFrame
        df = pd.DataFrame(training_data)

        # Ensure labels are in DataFrame
        if "label" not in df.columns and labels:
            df["label"] = labels
        elif "label" not in df.columns:
            raise ValueError(
                "Training data must include 'label' field or provide labels list"
            )

        # Initialize and train model
        self.model = TransactionCategorizer()
        self.model.fit_from_dataframe(df)

        # Save model if requested
        if save_model:
            metadata = {
                "training_samples": len(training_data),
                "categories": df["label"].unique().tolist(),
                "training_date": pd.Timestamp.now().isoformat(),
            }
            self.model_manager.save_model(
                self.model,
                self.model_name,
                version=version,
                metadata=metadata,
                use_joblib=True,
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

        return self.model.explain_one(transaction)

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
            if TransactionCategorizer is None:
                raise ImportError("TransactionCategorizer not available")

            self.model = self.model_manager.load_model(
                user_model_name, version=version, use_joblib=True
            )
            self.model_loaded = True
            self.model_name = user_model_name
            return True
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
    ) -> Dict[str, Any]:
        """
        Train a user-specific model.

        Args:
            user_id: User ID for user-specific model
            training_data: List of transaction dicts with 'label' field
            labels: List of category labels
            save_model: Whether to save the trained model
            version: Version string for the saved model

        Returns:
            Training results dict with metrics
        """
        if TransactionCategorizer is None:
            raise ImportError("TransactionCategorizer not available")

        # Create DataFrame
        df = pd.DataFrame(training_data)

        # Ensure labels are in DataFrame
        if "label" not in df.columns and labels:
            df["label"] = labels
        elif "label" not in df.columns:
            raise ValueError(
                "Training data must include 'label' field or provide labels list"
            )

        # Initialize and train model
        self.model = TransactionCategorizer()
        self.model.fit_from_dataframe(df)

        if save_model:
            user_model_name = f"transaction_categorizer_user_{user_id}"
            metadata = {
                "user_id": user_id,
                "training_samples": len(training_data),
                "categories": df["label"].unique().tolist(),
                "training_date": pd.Timestamp.now().isoformat(),
            }
            self.model_manager.save_model(
                self.model,
                user_model_name,
                version=version,
                metadata=metadata,
                use_joblib=True,
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            return self.model_manager.get_model_info(self.model_name)
        except Exception as e:
            return {"error": str(e), "loaded": self.model_loaded}
