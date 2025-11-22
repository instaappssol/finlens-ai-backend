"""
End-to-end Transaction Categorization Engine

Key features:
- Uses merchant→category mapping from MongoDB as knowledge base
- Merchant normalization using a semantic vector store (SentenceTransformer)
- Hybrid model: text embeddings + tabular features (PyTorch)
- Train & predict entrypoints
- Model + encoders + knowledge base stored in MongoDB GridFS
- All data input/output uses MongoDB (no CSV files)
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors  # only if you want fallback (optional)
from sklearn.neighbors import NearestNeighbors

from app.core.config import settings


# ============================================================
# 1. Merchant Knowledge Base + Vector Store
# ============================================================

@dataclass
class MerchantRecord:
    name: str
    category: str

class MerchantVectorStore:
    """
    Vector store over merchant names using SentenceTransformer + Qdrant.
    Uses QDRANT_HOST and QDRANT_API_KEY from config.
    """

    def __init__(
        self,
        merchants: List[str],
        categories: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "merchants",
    ):
        assert len(merchants) == len(categories)
        self.merchants = merchants
        self.categories = categories
        self.model_name = model_name
        self.collection_name = collection_name

        # Embedding model
        self.encoder = SentenceTransformer(model_name)

        # Connect to Qdrant instance using config
        self.client = QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY,
        )

        # Create / recreate collection
        dim = self.encoder.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # Build embeddings for canonical merchants (only if merchants exist)
        if len(self.merchants) > 0:
            embeddings = self.encoder.encode(self.merchants).tolist()

            points = [
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={
                        "merchant": self.merchants[i],
                        "category": self.categories[i],
                    },
                )
                for i in range(len(self.merchants))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def __getstate__(self):
        """Custom pickling: exclude non-picklable objects (encoder, client)"""
        state = self.__dict__.copy()
        # Remove non-picklable objects
        state['encoder'] = None
        state['client'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling: recreate encoder and client"""
        self.__dict__.update(state)
        # Recreate encoder
        self.encoder = SentenceTransformer(self.model_name)
        # Recreate Qdrant client
        self.client = QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY,
        )
        # Recreate collection and upsert points
        dim = self.encoder.get_sentence_embedding_dimension()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        if len(self.merchants) > 0:
            embeddings = self.encoder.encode(self.merchants).tolist()
            points = [
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={
                        "merchant": self.merchants[i],
                        "category": self.categories[i],
                    },
                )
                for i in range(len(self.merchants))
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def most_similar(
            self,
            text: str,
            top_k: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (merchant_name, category, similarity_score) sorted best→worst.
        Supports both old .search() and newer .query_points() APIs.
        """
        if not text or not text.strip():
            return []

        # 1) Embed the query
        query_emb = self.encoder.encode([text]).tolist()[0]

        # 2) Try new API first (query_points), fall back to search if not available
        hits = None

        if hasattr(self.client, "query_points"):
            # Newer qdrant-client versions
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_emb,
                limit=top_k,
            )
            hits = res.points
        else:
            # Older qdrant-client versions – use .search()
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_emb,
                limit=top_k,
            )

        results: List[Tuple[str, str, float]] = []
        for p in hits:
            # For query_points, p is usually a PointStruct-like object
            payload = getattr(p, "payload", None) or p.payload
            merch = payload.get("merchant")
            cat = payload.get("category")
            # Qdrant similarity: lower distance → higher score; convert to similarity
            score = getattr(p, "score", None)
            sim = 1.0 - float(score) if score is not None else 0.0
            results.append((merch, cat, sim))

        return results
    '''
    def most_similar(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (merchant_name, category, similarity_score) sorted best→worst.
        """
        if not text or not text.strip():
            return []

        query_emb = self.encoder.encode([text]).tolist()[0]

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            limit=top_k,
        )

        results: List[Tuple[str, str, float]] = []
        for p in hits:
            merch = p.payload.get("merchant")
            cat = p.payload.get("category")
            # Qdrant's score is distance for cosine; convert to similarity
            sim = 1.0 - float(p.score)
            results.append((merch, cat, sim))
        return results
'''
class MerchantKnowledgeBase:
    """
    Loads merchant→category mapping and exposes:
    - exact/substring lookup
    - semantic lookup via MerchantVectorStore
    """

    def __init__(
        self,
        merchants: List[MerchantRecord],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        corrections: Optional[Dict[str, Tuple[str, str]]] = None
    ):
        self.records = merchants
        self.model_name = model_name

        # Dict for exact lookup (lowercased)
        self.exact_map: Dict[str, str] = {
            rec.name.lower(): rec.category for rec in self.records
        }

        # Corrections dictionary (raw_description -> (canonical_merchant, canonical_category))
        self.corrections = corrections or {}

        # List for vector store
        merchant_names = [rec.name for rec in self.records]
        categories = [rec.category for rec in self.records]
        self.vector_store = MerchantVectorStore(
            merchants=merchant_names,
            categories=categories,
            model_name=model_name
        )


    # ---------- Loading / Saving KB ----------

    @staticmethod
    def from_mongodb(db, collection_name: str = "merchant_knowledge_base",
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load merchant knowledge base from MongoDB collection.
        
        Collection structure:
        {
            "merchant": "Amazon",
            "category": "Shopping",
            "merchants": ["Amazon", "Amazon.in"]  # optional list of aliases
        }
        
        Args:
            db: MongoDB database instance
            collection_name: Name of the collection storing merchant mappings
            model_name: Sentence transformer model name
            
        Returns:
            MerchantKnowledgeBase instance
        """
        collection = db[collection_name]
        records: List[MerchantRecord] = []
        
        # Load all merchant records from MongoDB
        for doc in collection.find():
            category = str(doc.get("category", "")).strip()
            merchant = str(doc.get("merchant", "")).strip()
            
            if not merchant or not category:
                continue
            
            # Add main merchant
            records.append(MerchantRecord(name=merchant, category=category))
            
            # Add aliases if provided
            merchants_list = doc.get("merchants", [])
            if isinstance(merchants_list, list):
                for alias in merchants_list:
                    alias_str = str(alias).strip()
                    if alias_str and alias_str.lower() != merchant.lower():
                        records.append(MerchantRecord(name=alias_str, category=category))
        
        # Remove duplicates by merchant name, keeping first category
        seen = {}
        dedup_records = []
        for r in records:
            key = r.name.lower()
            if key not in seen:
                seen[key] = r.category
                dedup_records.append(MerchantRecord(name=r.name, category=r.category))
        
        # Load corrections from MongoDB
        corrections_collection = db["merchant_corrections"]
        corrections = {}
        
        for doc in corrections_collection.find():
            raw_desc = str(doc.get("raw_description", ""))
            canon_merchant = str(doc.get("canonical_merchant", ""))
            canon_category = str(doc.get("canonical_category", ""))
            
            if raw_desc and canon_merchant and canon_category:
                norm = MerchantKnowledgeBase._normalize_for_match(raw_desc)
                corrections[norm] = (canon_merchant, canon_category)
        
        kb = MerchantKnowledgeBase(dedup_records, model_name=model_name, corrections=corrections)
        print(f"[INFO] Loaded {len(dedup_records)} merchants and {len(corrections)} corrections from MongoDB")
        return kb

    @staticmethod
    def from_list(merchant_data: List[Dict[str, Any]],
                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Create MerchantKnowledgeBase from a list of dictionaries.
        
        Args:
            merchant_data: List of dicts with 'merchant' and 'category' keys,
                          optionally 'merchants' list for aliases
            model_name: Sentence transformer model name
            
        Returns:
            MerchantKnowledgeBase instance
        """
        records: List[MerchantRecord] = []
        
        for item in merchant_data:
            category = str(item.get("category", "")).strip()
            merchant = str(item.get("merchant", "")).strip()
            
            if not merchant or not category:
                continue
            
            records.append(MerchantRecord(name=merchant, category=category))
            
            # Add aliases
            merchants_list = item.get("merchants", [])
            if isinstance(merchants_list, list):
                for alias in merchants_list:
                    alias_str = str(alias).strip()
                    if alias_str and alias_str.lower() != merchant.lower():
                        records.append(MerchantRecord(name=alias_str, category=category))
        
        # Remove duplicates
        seen = {}
        dedup_records = []
        for r in records:
            key = r.name.lower()
            if key not in seen:
                seen[key] = r.category
                dedup_records.append(MerchantRecord(name=r.name, category=r.category))
        
        return MerchantKnowledgeBase(dedup_records, model_name=model_name)

    def to_json_dict(self) -> Dict:
        """
        Serialize KB minimally so it can be restored elsewhere.
        We do NOT store embeddings, just names/categories/model_name.
        """
        return {
            "model_name": self.model_name,
            "records": [{"name": r.name, "category": r.category} for r in self.records],
            "corrections": {k: list(v) for k, v in self.corrections.items()},
        }

    @staticmethod
    def from_json_dict(data: Dict):
        records = [MerchantRecord(**r) for r in data["records"]]
        model_name = data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        corrections = data.get("corrections", {})
        # Convert list back to tuple
        corrections_dict = {k: tuple(v) for k, v in corrections.items()}
        return MerchantKnowledgeBase(records, model_name=model_name, corrections=corrections_dict)

    # ---------- Lookup / Normalization ----------

    @staticmethod
    def _basic_clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lookup_exact(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower().strip()
        return self.exact_map.get(t)

    def lookup_substring(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Check if any canonical merchant name appears as substring.
        Useful for cases like "AMZN PAY" containing "Amazon".
        """
        if not text:
            return None
        norm_text = self._normalize_for_match(text)
        for rec in self.records:
            norm_merch = self._normalize_for_match(rec.name)
            if norm_merch and norm_merch in norm_text:
                return rec.name, rec.category
        return None

    def normalize_merchant_with_semantics(
        self,
        raw_description: str,
        similarity_threshold: float = 0.55
    ) -> Tuple[str, Optional[str], float]:
        """
        Main entrypoint for normalization:
        1) exact match on clean merchant
        2) substring presence of canonical merchant name
        3) semantic nearest neighbors from vector store (vector DB)

        Returns:
            canonical_name, category, confidence_score (0~1)
        """
        if not raw_description:
            return "unknown", None, 0.0

        cleaned = self._basic_clean(raw_description)
        norm = self._normalize_for_match(cleaned)

        # -1. Apply manual correction rule before anything else
        if norm in self.corrections:
            merch, cat = self.corrections[norm]
            return merch, cat, 1.0

        # 1. exact match
        cat = self.lookup_exact(cleaned)
        if cat:
            return cleaned, cat, 1.0

        # 2. substring match
        sub = self.lookup_substring(cleaned)
        if sub is not None:
            merch_name, cat = sub
            return merch_name, cat, 0.9

        # 3. semantic similarity via vector DB
        results = self.vector_store.most_similar(cleaned, top_k=3)
        if not results:
            return cleaned, None, 0.0

        best_name, best_cat, best_sim = results[0]
        if best_sim >= similarity_threshold:
            return best_name, best_cat, best_sim
        else:
            return cleaned, None, best_sim


# ============================================================
# 2. Preprocessing & Feature Engineering
# ============================================================

class TransactionPreprocessor:
    """
    Preprocess raw transactions into a normalized structure:
    - cleaned description text
    - normalized merchant (with KB + vector DB)
    - optional base category from merchant KB
    - temporal features from timestamp
    """

    def __init__(self, merchant_kb: MerchantKnowledgeBase):
        self.merchant_kb = merchant_kb

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess_one(self, transaction: Dict) -> Dict:
        """
        Expected transaction dict keys:
        - "description": string describing transaction (merchant + context)
        - "amount": numeric
        - "timestamp": ISO string (optional)
        - "type": optional (e.g., "debit"/"credit")
        """
        processed = dict(transaction)  # shallow copy

        raw_desc = transaction.get("description", "")
        processed["cleaned_description"] = self.clean_text(raw_desc)

        # Merchant normalization using KB + vector DB
        merch_name, merch_cat, merch_conf = self.merchant_kb.normalize_merchant_with_semantics(raw_desc)
        processed["normalized_merchant"] = merch_name
        processed["kb_category"] = merch_cat  # can be None
        processed["merchant_confidence"] = merch_conf

        # Temporal features
        ts = transaction.get("timestamp")
        if ts:
            try:
                dt = pd.to_datetime(ts, utc=True)
                processed["hour"] = int(dt.hour)
                processed["day_of_week"] = int(dt.weekday())
                processed["month"] = int(dt.month)
            except Exception:
                processed["hour"] = 0
                processed["day_of_week"] = 0
                processed["month"] = 0
        else:
            processed["hour"] = 0
            processed["day_of_week"] = 0
            processed["month"] = 0

        # Type fallback
        processed["type"] = transaction.get("type", "UNKNOWN")

        # Amount fallback
        processed["amount"] = float(transaction.get("amount", 0.0))

        return processed

    def preprocess_batch(self, transactions: List[Dict]) -> List[Dict]:
        return [self.preprocess_one(t) for t in transactions]


class FeatureEngineering:
    """
    Turns preprocessed transactions into:
    - text embeddings (cleaned_description)
    - tabular numeric features (amount, time, merchant_conf, kb_category, type)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.text_encoder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.type_encoder = LabelEncoder()
        self.kb_cat_encoder = LabelEncoder()
        self.fitted = False

    def __getstate__(self):
        """Custom pickling: exclude non-picklable text_encoder"""
        state = self.__dict__.copy()
        # Remove non-picklable object
        state['text_encoder'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling: recreate text_encoder"""
        self.__dict__.update(state)
        # Recreate text encoder
        self.text_encoder = SentenceTransformer(self.model_name)

    def fit(self, transactions: List[Dict], labels: List[str]):
        df = pd.DataFrame(transactions)

        # Numeric columns
        amounts = df["amount"].fillna(0).values.reshape(-1, 1)
        amounts_log = np.log1p(amounts)
        self.scaler.fit(amounts_log)

        # Type encoding
        self.type_encoder.fit(df["type"].fillna("UNKNOWN"))

        # KB category encoding (treat None as "NONE")
        kb_cats = df["kb_category"].fillna("NONE")
        self.kb_cat_encoder.fit(kb_cats)

        self.fitted = True
        return self

    def extract_text_features(self, descriptions: List[str]) -> np.ndarray:
        return self.text_encoder.encode(descriptions)

    def extract_tabular_features(self, transactions: List[Dict]) -> np.ndarray:
        df = pd.DataFrame(transactions)

        # Amount
        amounts = df["amount"].fillna(0).values.reshape(-1, 1)
        amounts_log = np.log1p(amounts)
        amounts_scaled = self.scaler.transform(amounts_log)

        # Time features (cyclical encoding for hour & day_of_week)
        hour = df["hour"].fillna(0).astype(int)
        day = df["day_of_week"].fillna(0).astype(int)
        month = df["month"].fillna(1).astype(int)

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        # Encoded type + KB category
        type_enc = self.type_encoder.transform(df["type"].fillna("UNKNOWN"))
        # ---- robust KB category encoding ----
        classes = list(self.kb_cat_encoder.classes_)
        if classes:
            fallback = classes[0]
        else:
            fallback = "NONE"

        kb_cats = df["kb_category"].fillna(fallback).astype(str)
        valid_set = set(classes)
        kb_cats = kb_cats.apply(lambda x: x if x in valid_set else fallback)

        kb_cat_enc = self.kb_cat_encoder.transform(kb_cats)
        # -------------------------------------



        merchant_conf = df["merchant_confidence"].fillna(0.0).values

        # Stack features
        features = np.column_stack([
            amounts_scaled.reshape(-1),
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            month,
            type_enc,
            kb_cat_enc,
            merchant_conf,
        ])

        return features.astype(np.float32)

    def transform(self, transactions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("FeatureEngineering must be fitted before transform.")
        df = pd.DataFrame(transactions)
        text_features = self.extract_text_features(df["cleaned_description"].tolist())
        tabular_features = self.extract_tabular_features(transactions)
        return text_features.astype(np.float32), tabular_features.astype(np.float32)

    def save(self, path: str):
        data = {
            "scaler_mean_": self.scaler.mean_.tolist(),
            "scaler_scale_": self.scaler.scale_.tolist(),
            "type_classes_": self.type_encoder.classes_.tolist(),
            "kb_cat_classes_": self.kb_cat_encoder.classes_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fe = FeatureEngineering(model_name=model_name)
        fe.scaler.mean_ = np.array(data["scaler_mean_"], dtype=np.float64)
        fe.scaler.scale_ = np.array(data["scaler_scale_"], dtype=np.float64)
        fe.scaler.var_ = fe.scaler.scale_ ** 2

        fe.type_encoder.classes_ = np.array(data["type_classes_"])
        fe.kb_cat_encoder.classes_ = np.array(data["kb_cat_classes_"])
        fe.fitted = True
        return fe


# ============================================================
# 3. Hybrid Neural Network
# ============================================================

class HybridTransactionNet(nn.Module):
    """
    Two-pathway network:
    - text_pathway: handles sentence embeddings
    - tabular_pathway: handles numeric + encoded features
    """

    def __init__(self, text_dim: int, tabular_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        # Text pathway
        self.text_pathway = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Tabular pathway
        self.tabular_pathway = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fused_dim = hidden_dim  # (hidden_dim//2 + hidden_dim//2)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text_features: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        t = self.text_pathway(text_features)
        x = self.tabular_pathway(tabular_features)
        fused = torch.cat([t, x], dim=1)
        logits = self.classifier(fused)
        return logits


# ============================================================
# 4. Engine: Training & Prediction
# ============================================================

class TransactionCategorizationEngine:
    def __init__(
        self,
        merchant_kb: MerchantKnowledgeBase,
        feature_engineer: FeatureEngineering,
        model: HybridTransactionNet,
        label_encoder: LabelEncoder,
        device: Optional[str] = None,
    ):
        self.merchant_kb = merchant_kb
        self.preprocessor = TransactionPreprocessor(merchant_kb)
        self.fe = feature_engineer
        self.model = model
        self.label_encoder = label_encoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    # ---------- Training ----------

    @staticmethod
    def from_training_data(
        df: pd.DataFrame,
        merchant_kb: MerchantKnowledgeBase,
        label_col: str = "category",
        description_col: str = "description",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        type_col: Optional[str] = None,
        hidden_dim: int = 256,
        test_size: float = 0.2,
        random_state: int = 42,
        num_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> "TransactionCategorizationEngine":
        """
        df must contain:
          - description_col: merchant + narrative text
          - amount_col
          - label_col: final target category (one of the 8)
          - optional timestamp_col, type_col
        """

        # Build list of raw transactions
        records = []
        for _, row in df.iterrows():
            rec = {
                "description": str(row[description_col]),
                "amount": float(row[amount_col]),
                "timestamp": str(row[timestamp_col]) if timestamp_col in df.columns else None,
                "type": str(row[type_col]) if type_col and type_col in df.columns else "UNKNOWN",
            }
            records.append(rec)

        # Preprocess & labels
        preproc = TransactionPreprocessor(merchant_kb)
        processed = preproc.preprocess_batch(records)
        labels = df[label_col].tolist()

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            processed, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Feature engineer
        fe = FeatureEngineering()
        fe.fit(X_train, labels=y_train)

        X_train_text, X_train_tab = fe.transform(X_train)
        X_val_text, X_val_tab = fe.transform(X_val)

        # Label encoder
        le = LabelEncoder()
        le.fit(labels)

        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)

        text_dim = X_train_text.shape[1]
        tabular_dim = X_train_tab.shape[1]
        num_classes = len(le.classes_)

        model = HybridTransactionNet(
            text_dim=text_dim,
            tabular_dim=tabular_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        def batch_iter(X_t, X_tab, y, bs):
            n = len(y)
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, bs):
                idx = indices[start:start + bs]
                yield (
                    X_t[idx],
                    X_tab[idx],
                    y[idx],
                )

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for Xt_b, Xtab_b, y_b in batch_iter(
                X_train_text, X_train_tab, y_train_enc, batch_size
            ):
                Xt_b = torch.tensor(Xt_b).to(device)
                Xtab_b = torch.tensor(Xtab_b).to(device)
                y_b = torch.tensor(y_b).long().to(device)

                optimizer.zero_grad()
                logits = model(Xt_b, Xtab_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_b)

            avg_loss = total_loss / len(y_train_enc)

            # Simple validation F1
            model.eval()
            with torch.no_grad():
                Xt_v = torch.tensor(X_val_text).to(device)
                Xtab_v = torch.tensor(X_val_tab).to(device)
                yv = torch.tensor(y_val_enc).long().to(device)
                logits_v = model(Xt_v, Xtab_v)
                preds_v = torch.argmax(logits_v, dim=1).cpu().numpy()
                yv_np = yv.cpu().numpy()
                f1 = f1_score(yv_np, preds_v, average="macro")

            print(f"Epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4f}, val_macro_f1={f1:.4f}")

        print("\nValidation classification report:")
        print(classification_report(y_val, le.inverse_transform(preds_v)))

        # Build engine object
        engine = TransactionCategorizationEngine(
            merchant_kb=merchant_kb,
            feature_engineer=fe,
            model=model,
            label_encoder=le,
            device=device,
        )

        return engine

    # ---------- Inference ----------

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        transactions: list of dicts with at least:
          - description
          - amount
          - timestamp (optional)
          - type (optional)
        Returns list of dicts with predicted category and internal details.
        """
        self.model.eval()

        processed = self.preprocessor.preprocess_batch(transactions)
        X_text, X_tab = self.fe.transform(processed)

        Xt = torch.tensor(X_text).to(self.device)
        Xtab = torch.tensor(X_tab).to(self.device)

        with torch.no_grad():
            logits = self.model(Xt, Xtab)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        labels = self.label_encoder.inverse_transform(preds)

        results = []
        for tx, proc, label, prob_vec in zip(transactions, processed, labels, probs):
            kb_cat = proc["kb_category"]
            merch_conf = proc["merchant_confidence"]

            # Base model prediction
            final_label = label
            final_conf = float(np.max(prob_vec))

            # RULE: if merchant KB is confident, trust kb_category over model
            if kb_cat is not None and kb_cat != "NONE" and merch_conf is not None:
                if merch_conf >= 0.8:
                    final_label = kb_cat
                    # you can optionally bump confidence
                    final_conf = max(final_conf, merch_conf)

            result = dict(tx)
            result["normalized_merchant"] = proc["normalized_merchant"]
            result["kb_category"] = kb_cat
            result["merchant_confidence"] = merch_conf
            result["predicted_category"] = final_label
            result["prediction_confidence"] = final_conf

            results.append(result)

        return results

    # ---------- Persistence ----------

    def save_to_dict(self) -> Dict[str, Any]:
        """
        Serialize engine to dictionary for storage in MongoDB/GridFS.
        Returns all components as JSON-serializable dicts.
        """
        # Model state dict (convert to CPU and make serializable)
        model_state = self.model.state_dict()
        model_dict = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                     for k, v in model_state.items()}
        
        # Label encoder
        le_data = {"classes_": self.label_encoder.classes_.tolist()}
        
        # Feature engineering
        fe_data = {
            "scaler_mean_": self.fe.scaler.mean_.tolist(),
            "scaler_scale_": self.fe.scaler.scale_.tolist(),
            "type_classes_": self.fe.type_encoder.classes_.tolist(),
            "kb_cat_classes_": self.fe.kb_cat_encoder.classes_.tolist(),
        }
        
        # Merchant KB
        kb_data = self.merchant_kb.to_json_dict()
        
        return {
            "model": model_dict,
            "label_encoder": le_data,
            "feature_engineering": fe_data,
            "merchant_kb": kb_data,
            "text_dim": 384,  # MiniLM dimension
            "tabular_dim": 9,  # From feature engineering
            "hidden_dim": 256,
            "num_classes": len(self.label_encoder.classes_),
        }

    def save(self, dir_path: str):
        """
        Save engine to directory (for debugging/testing purposes).
        Note: In production, models are saved directly to MongoDB GridFS via pickle.
        """
        os.makedirs(dir_path, exist_ok=True)

        # Model
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model.pt"))

        # Label encoder
        le_data = {"classes_": self.label_encoder.classes_.tolist()}
        with open(os.path.join(dir_path, "label_encoder.json"), "w", encoding="utf-8") as f:
            json.dump(le_data, f, indent=2)

        # Feature engineering
        self.fe.save(os.path.join(dir_path, "feature_engineering.json"))

        # Merchant KB (save as JSON)
        kb_data = self.merchant_kb.to_json_dict()
        with open(os.path.join(dir_path, "merchant_kb.json"), "w", encoding="utf-8") as f:
            json.dump(kb_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(dir_path: str) -> "TransactionCategorizationEngine":
        # KB (load from JSON file)
        with open(os.path.join(dir_path, "merchant_kb.json"), "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        kb = MerchantKnowledgeBase.from_json_dict(kb_data)

        # Feature engineering
        fe = FeatureEngineering.load(os.path.join(dir_path, "feature_engineering.json"))

        # Label encoder
        with open(os.path.join(dir_path, "label_encoder.json"), "r", encoding="utf-8") as f:
            le_data = json.load(f)
        le = LabelEncoder()
        le.classes_ = np.array(le_data["classes_"])

        # Model
        # For simplicity we assume text_dim=384 (MiniLM) and tabular_dim=9 as per engineering above.
        text_dim = 384
        tabular_dim = 9
        hidden_dim = 256
        num_classes = len(le.classes_)

        model = HybridTransactionNet(
            text_dim=text_dim,
            tabular_dim=tabular_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(os.path.join(dir_path, "model.pt"), map_location=device))
        model.to(device)

        return TransactionCategorizationEngine(
            merchant_kb=kb,
            feature_engineer=fe,
            model=model,
            label_encoder=le,
            device=device,
        )


# CLI removed - all operations now go through API endpoints and MongoDB
