"""
Transaction Categorizer ML Model

In-house AI-powered financial transaction categorisation.

Implements:
- Preprocessing & enrichment
- Feature building (text + tabular)
- Baseline model (TF-IDF + LogisticRegression)
- Inference with confidence
- Hook for SHAP/XAI
- Taxonomy alignment

This is the core ML model for transaction categorization.
"""

from __future__ import annotations
import re
import json
import string
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.compose import ColumnTransformer
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIG / TAXONOMY
# -----------------------------------------------------------------------------

DEFAULT_TAXONOMY = {
    "Groceries": ["groceries", "supermarket", "food_store"],
    "Food & Beverage": ["restaurant", "cafe", "coffee"],
    "Fuel": ["fuel", "gas", "petrol", "diesel"],
    "Transfers": ["p2p", "peer-to-peer", "wallet_transfer", "account_transfer"],
    "Shopping": ["ecommerce", "retail", "amazon", "flipkart"],
    "Bills & Utilities": ["electricity", "water", "mobile", "broadband"],
    "Travel": ["flight", "hotel", "taxi", "uber", "ola"],
}

# -----------------------------------------------------------------------------
# 2. PREPROCESSING & ENRICHMENT
# -----------------------------------------------------------------------------


class TransactionPreprocessor:
    COMMON_PREFIXES = [
        r"payment to",
        r"payment frm",
        r"paid to",
        r"upi/",
        r"imps/",
        r"neft/",
        r"pos/",
    ]

    def __init__(self, merchant_db: Optional[Dict[str, str]] = None):
        self.merchant_db = merchant_db or {}

    def clean_text(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower().strip()
        for p in self.COMMON_PREFIXES:
            s = re.sub(p, "", s).strip()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def normalize_merchant(self, description: str) -> str:
        cleaned = self.clean_text(description)
        if cleaned in self.merchant_db:
            return self.merchant_db[cleaned]
        return cleaned

    def enrich(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        desc = row.get("description", "") or row.get("payee_name", "") or ""
        out["clean_description"] = self.clean_text(desc)
        out["canonical_merchant"] = self.normalize_merchant(desc)

        text = out["clean_description"]
        is_p2p = any(k in text for k in ["to", "from"]) and row.get(
            "transaction_type", ""
        ).lower() in (
            "p2p_transfer",
            "p2p",
            "transfer",
        )
        out["is_p2p"] = int(is_p2p)

        ts = row.get("timestamp")
        if ts and isinstance(ts, str) and "t" in ts.lower():
            try:
                timepart = ts.split("T")[1]
                hour = int(timepart.split(":")[0])
            except Exception:
                hour = -1
        else:
            hour = -1
        out["hour"] = hour

        return out


# -----------------------------------------------------------------------------
# 3. FEATURE BUILDER
# -----------------------------------------------------------------------------


class FeatureBuilder:
    def __init__(self):
        self.ct: Optional[ColumnTransformer] = None
        self.text_col = "clean_description"
        self.categorical_cols = ["transaction_type"]
        self.numeric_cols = ["amount", "hour", "is_p2p"]

    def build_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
        num_transformer = StandardScaler()

        self.ct = ColumnTransformer(
            transformers=[
                ("text", text_transformer, self.text_col),
                ("cat", cat_transformer, self.categorical_cols),
                ("num", num_transformer, self.numeric_cols),
            ],
            remainder="drop",
        )
        self.ct.fit(df)
        return self.ct

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.ct is None:
            raise RuntimeError(
                "FeatureBuilder not fitted. Call build_transformer() first."
            )
        return self.ct.transform(df)

    def get_feature_names(self) -> List[str]:
        if self.ct is None:
            return []
        names = []
        for name, trans, cols in self.ct.transformers_:
            if name == "text":
                tfidf = trans
                names.extend([f"text_{f}" for f in tfidf.get_feature_names_out()])
            elif name == "cat":
                oh = trans
                names.extend(oh.get_feature_names_out(cols).tolist())
            elif name == "num":
                names.extend(cols)
        return names


# -----------------------------------------------------------------------------
# 4. MODEL WRAPPER
# -----------------------------------------------------------------------------


@dataclass
class PredictionResult:
    transaction_id: Optional[str]
    category: str
    confidence_score: float
    model_version: str = "v0.1.0"
    explanation_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class TransactionCategorizer:
    def __init__(self, taxonomy: Dict[str, List[str]] = None):
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self.pre = TransactionPreprocessor()
        self.fb = FeatureBuilder()
        self.model = LogisticRegression(max_iter=200, n_jobs=None)
        self.fitted = False

    def fit_from_dataframe(self, df: pd.DataFrame, text_col: str = "description"):
        processed_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            if "description" not in row_dict and text_col in row_dict:
                row_dict["description"] = row_dict[text_col]
            processed = self.pre.enrich(row_dict)
            processed_rows.append(processed)

        proc_df = pd.DataFrame(processed_rows)

        if "transaction_type" not in proc_df.columns:
            proc_df["transaction_type"] = "unknown"

        X = proc_df
        y = df["label"].astype(str)

        self.fb.build_transformer(X)
        X_vec = self.fb.transform(X)

        # Check if we have enough samples per class for stratified split
        # Stratification requires at least 2 samples per class
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_vec, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Not enough samples per class for stratification
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)
        print("Classification report:\n", classification_report(y_te, y_pred))
        print("Macro F1:", f1_score(y_te, y_pred, average="macro"))

        self.fitted = True

    def predict_one(self, txn: Dict[str, Any]) -> PredictionResult:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Train or load weights first.")

        enriched = self.pre.enrich(txn)
        df = pd.DataFrame([enriched])
        if "transaction_type" not in df.columns:
            df["transaction_type"] = "unknown"

        X_vec = self.fb.transform(df)
        proba = self.model.predict_proba(X_vec)[0]
        classes = self.model.classes_
        best_idx = int(np.argmax(proba))
        best_label = classes[best_idx]
        confidence = float(proba[best_idx])

        return PredictionResult(
            transaction_id=txn.get("transaction_id"),
            category=best_label,
            confidence_score=confidence,
        )

    def save(self, path: str):
        payload = {
            "taxonomy": self.taxonomy,
            "pre": self.pre,
            "fb": self.fb,
            "model": self.model,
            "fitted": self.fitted,
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "TransactionCategorizer":
        payload = joblib.load(path)
        obj = TransactionCategorizer(payload["taxonomy"])
        obj.pre = payload["pre"]
        obj.fb = payload["fb"]
        obj.model = payload["model"]
        obj.fitted = payload["fitted"]
        return obj

    def explain_one(self, txn: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Generate dynamic explanation for a transaction prediction.
        
        Args:
            txn: Transaction dictionary
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary with prediction and top contributing features
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Train or load weights first.")
        
        pred = self.predict_one(txn)
        
        try:
            # Get the transformed feature vector
            enriched = self.pre.enrich(txn)
            df = pd.DataFrame([enriched])
            if "transaction_type" not in df.columns:
                df["transaction_type"] = "unknown"
            
            X_vec = self.fb.transform(df)
            
            # Get feature names
            feature_names = self.fb.get_feature_names()
            
            if not feature_names:
                # Fallback if feature names are not available
                return {
                    "transaction_id": pred.transaction_id,
                    "predicted_category": pred.category,
                    "confidence_score": pred.confidence_score,
                    "top_factors": [],
                    "model_version": pred.model_version,
                    "error": "No feature names available",
                }
            
            # Get coefficients for the predicted class
            class_matches = np.where(self.model.classes_ == pred.category)[0]
            if len(class_matches) == 0:
                # Fallback if category not found in classes
                return {
                    "transaction_id": pred.transaction_id,
                    "predicted_category": pred.category,
                    "confidence_score": pred.confidence_score,
                    "top_factors": [],
                    "model_version": pred.model_version,
                    "error": "Category not found in classes",
                }
            
            class_idx = class_matches[0]
            
            # Handle binary classification case where coef_ has shape (1, n_features)
            # For binary: coef_[0] represents class1 vs class0
            # For multiclass: coef_[class_idx] represents the class
            if len(self.model.classes_) == 2 and self.model.coef_.shape[0] == 1:
                # Binary classification: use coef_[0] for class1, negate for class0
                if class_idx == 1:
                    coefficients = self.model.coef_[0]
                else:  # class_idx == 0
                    coefficients = -self.model.coef_[0]
            else:
                # Multiclass or other cases: use coef_[class_idx]
                if class_idx >= self.model.coef_.shape[0]:
                    # Index out of bounds - fallback
                    return {
                        "transaction_id": pred.transaction_id,
                        "predicted_category": pred.category,
                        "confidence_score": pred.confidence_score,
                        "top_factors": [],
                        "model_version": pred.model_version,
                        "error": f"Class index {class_idx} out of bounds for coef_ shape {self.model.coef_.shape}",
                    }
                coefficients = self.model.coef_[class_idx]
            
            # Calculate feature contributions (feature_value * coefficient)
            if hasattr(X_vec, 'toarray'):
                feature_values = X_vec.toarray()[0]
            else:
                feature_values = np.asarray(X_vec[0]).flatten()
            
            # Ensure feature_values and coefficients have matching lengths
            min_len = min(len(feature_values), len(coefficients), len(feature_names))
            if min_len == 0:
                return {
                    "transaction_id": pred.transaction_id,
                    "predicted_category": pred.category,
                    "confidence_score": pred.confidence_score,
                    "top_factors": [],
                    "model_version": pred.model_version,
                    "error": "No feature names available",
                }
            
            # Truncate to minimum length to ensure alignment
            feature_values = feature_values[:min_len]
            coefficients = coefficients[:min_len]
            feature_names = feature_names[:min_len]
            
            contributions = feature_values * coefficients
            
            # Create list of (feature_name, contribution) tuples
            feature_contributions = list(zip(feature_names, contributions))
            
            # Sort by absolute contribution and get top N
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_factors = [
                {"feature": name, "contribution": float(contrib)}
                for name, contrib in feature_contributions[:top_n]
            ]
            
            return {
                "transaction_id": pred.transaction_id,
                "predicted_category": pred.category,
                "confidence_score": pred.confidence_score,
                "top_factors": top_factors,
                "model_version": pred.model_version,
                "error": None,
            }
        except Exception as e:
            # Return a minimal explanation if feature extraction fails
            # This prevents the entire explanation from being null
            return {
                "transaction_id": pred.transaction_id,
                "predicted_category": pred.category,
                "confidence_score": pred.confidence_score,
                "top_factors": [],
                "model_version": pred.model_version,
                "error": f"Could not generate detailed explanation: {str(e)}",
            }
