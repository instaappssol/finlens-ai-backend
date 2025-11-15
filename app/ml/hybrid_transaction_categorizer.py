"""
Hybrid Transaction Categorizer ML Model

Advanced neural network-based transaction categorization using
Sentence Transformers for text features and traditional ML for tabular features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import json

# For ML components
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TransactionPreprocessor:
    """Handles cleaning and normalization of raw transaction data."""
    
    def __init__(self):
        self.merchant_mappings = {
            'amzn': 'amazon',
            'starbucks': 'starbucks',
            'walmart': 'walmart',
            'shell': 'shell',
            'target': 'target'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize transaction description."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def normalize_merchant(self, description: str) -> str:
        """Extract and normalize merchant name from description."""
        cleaned = self.clean_text(description)
        
        # Check for known merchants
        for key, merchant in self.merchant_mappings.items():
            if key in cleaned:
                return merchant
        
        # Return first word as fallback
        words = cleaned.split()
        return words[0] if words else "unknown"
    
    def preprocess(self, transaction: Dict) -> Dict:
        """Preprocess a single transaction."""
        processed = transaction.copy()
        
        # Clean description
        processed['cleaned_description'] = self.clean_text(
            transaction.get('description', '')
        )
        
        # Normalize merchant
        processed['merchant'] = self.normalize_merchant(
            transaction.get('description', '')
        )
        
        # Extract temporal features
        if 'timestamp' in transaction:
            dt = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))
            processed['hour'] = dt.hour
            processed['day_of_week'] = dt.weekday()
            processed['month'] = dt.month
        
        return processed


class FeatureEngineering:
    """Handles feature extraction and encoding."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.currency_encoder = None
        self.type_encoder = None
        self.fitted = False
    
    def fit(self, transactions: List[Dict], labels: List[str]):
        """Fit encoders and scalers on training data."""
        df = pd.DataFrame(transactions)
        
        # Fit numerical scaler
        numerical_cols = ['amount']
        if numerical_cols[0] in df.columns:
            # Log transform amounts
            amounts = np.log1p(df['amount'].values.reshape(-1, 1))
            self.scaler.fit(amounts)
        
        # Fit categorical encoders
        if 'currency' in df.columns:
            self.currency_encoder = LabelEncoder()
            self.currency_encoder.fit(df['currency'].fillna('USD'))
        
        if 'type' in df.columns:
            self.type_encoder = LabelEncoder()
            self.type_encoder.fit(df['type'].fillna('UNKNOWN'))
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        
        self.fitted = True
        return self
    
    def extract_text_features(self, descriptions: List[str]) -> np.ndarray:
        """Extract text embeddings using sentence transformer."""
        embeddings = self.text_encoder.encode(descriptions)
        return embeddings
    
    def extract_numerical_features(self, transactions: List[Dict]) -> np.ndarray:
        """Extract and scale numerical features."""
        df = pd.DataFrame(transactions)
        
        amounts = df['amount'].values.reshape(-1, 1)
        amounts_log = np.log1p(amounts)
        amounts_scaled = self.scaler.transform(amounts_log)
        
        return amounts_scaled
    
    def extract_temporal_features(self, transactions: List[Dict]) -> np.ndarray:
        """Extract temporal features with cyclical encoding."""
        df = pd.DataFrame(transactions)
        
        features = []
        
        if 'hour' in df.columns:
            # Cyclical encoding for hour
            hour_sin = np.sin(2 * np.pi * df['hour'] / 24)
            hour_cos = np.cos(2 * np.pi * df['hour'] / 24)
            features.extend([hour_sin, hour_cos])
        
        if 'day_of_week' in df.columns:
            # Cyclical encoding for day of week
            dow_sin = np.sin(2 * np.pi * df['day_of_week'] / 7)
            dow_cos = np.cos(2 * np.pi * df['day_of_week'] / 7)
            features.extend([dow_sin, dow_cos])
        
        if 'month' in df.columns:
            # Cyclical encoding for month
            month_sin = np.sin(2 * np.pi * df['month'] / 12)
            month_cos = np.cos(2 * np.pi * df['month'] / 12)
            features.extend([month_sin, month_cos])
        
        return np.column_stack(features) if features else np.zeros((len(df), 6))
    
    def extract_categorical_features(self, transactions: List[Dict]) -> np.ndarray:
        """Extract and encode categorical features."""
        df = pd.DataFrame(transactions)
        
        features = []
        
        if 'currency' in df.columns and self.currency_encoder:
            currency_encoded = self.currency_encoder.transform(
                df['currency'].fillna('USD')
            )
            features.append(currency_encoded)
        
        if 'type' in df.columns and self.type_encoder:
            type_encoded = self.type_encoder.transform(
                df['type'].fillna('UNKNOWN')
            )
            features.append(type_encoded)
        
        return np.column_stack(features) if features else np.zeros((len(df), 2))
    
    def transform(self, transactions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform transactions into text and tabular features."""
        if not self.fitted:
            raise ValueError("FeatureEngineering must be fitted before transform")
        
        # Extract descriptions for text features
        descriptions = [t.get('cleaned_description', '') for t in transactions]
        text_features = self.extract_text_features(descriptions)
        
        # Extract tabular features
        numerical_features = self.extract_numerical_features(transactions)
        temporal_features = self.extract_temporal_features(transactions)
        categorical_features = self.extract_categorical_features(transactions)
        
        # Concatenate all tabular features
        tabular_features = np.column_stack([
            numerical_features,
            temporal_features,
            categorical_features
        ])
        
        return text_features, tabular_features


class HybridTransactionClassifier(nn.Module):
    """Hybrid neural network combining text and tabular pathways."""
    
    def __init__(self, text_dim: int, tabular_dim: int, num_classes: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Text pathway (for transformer embeddings)
        self.text_pathway = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Tabular pathway (for structured features)
        self.tabular_pathway = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion and classification head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, text_features, tabular_features):
        """Forward pass through the hybrid network."""
        # Process text features
        text_out = self.text_pathway(text_features)
        
        # Process tabular features
        tabular_out = self.tabular_pathway(tabular_features)
        
        # Concatenate and fuse
        fused = torch.cat([text_out, tabular_out], dim=1)
        
        # Final classification
        logits = self.fusion(fused)
        
        return logits


class TransactionCategorizationSystem:
    """Complete end-to-end transaction categorization system."""
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.preprocessor = TransactionPreprocessor()
        self.feature_engineer = FeatureEngineering()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, transactions: List[Dict], labels: List[str],
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the categorization model."""
        print(transactions)
        # Preprocess transactions
        processed = [self.preprocessor.preprocess(t) for t in transactions]
        
        # Fit and transform features
        self.feature_engineer.fit(processed, labels)
        text_features, tabular_features = self.feature_engineer.transform(processed)
        
        # Encode labels
        y = self.feature_engineer.label_encoder.transform(labels)
        
        # Split data
        indices = np.arange(len(transactions))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Initialize model
        text_dim = text_features.shape[1]
        tabular_dim = tabular_features.shape[1]
        num_classes = len(self.categories)
        
        self.model = HybridTransactionClassifier(text_dim, tabular_dim, num_classes)
        self.model.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_text = torch.FloatTensor(text_features)
        X_tab = torch.FloatTensor(tabular_features)
        y_tensor = torch.LongTensor(y)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i:i + batch_size]
                
                batch_text = X_text[batch_idx].to(self.device)
                batch_tab = X_tab[batch_idx].to(self.device)
                batch_y = y_tensor[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_text, batch_tab)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 10 == 0:
                val_f1 = self.evaluate(
                    text_features[val_idx],
                    tabular_features[val_idx],
                    y[val_idx]
                )
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_idx):.4f}, Val F1: {val_f1:.4f}")
    
    def evaluate(self, text_features: np.ndarray, tabular_features: np.ndarray,
                 y_true: np.ndarray) -> float:
        """Evaluate model performance using macro F1-score."""
        self.model.eval()
        
        with torch.no_grad():
            X_text = torch.FloatTensor(text_features).to(self.device)
            X_tab = torch.FloatTensor(tabular_features).to(self.device)
            
            outputs = self.model(X_text, X_tab)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        f1 = f1_score(y_true, predictions, average='macro')
        return f1
    
    def predict(self, transaction: Dict) -> Dict:
        """Predict category for a single transaction."""
        self.model.eval()
        
        # Preprocess
        processed = self.preprocessor.preprocess(transaction)
        
        # Extract features
        text_features, tabular_features = self.feature_engineer.transform([processed])
        
        # Predict
        with torch.no_grad():
            X_text = torch.FloatTensor(text_features).to(self.device)
            X_tab = torch.FloatTensor(tabular_features).to(self.device)
            
            outputs = self.model(X_text, X_tab)
            probabilities = torch.softmax(outputs, dim=1)
            
            pred_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_idx].item()
        
        # Decode prediction
        category = self.feature_engineer.label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            'transaction_id': transaction.get('id', 'unknown'),
            'category': category,
            'confidence_score': round(confidence, 4),
            'model_version': 'v1.0.0'
        }


# Example usage
if __name__ == "__main__":
    import random

    # 1. Define possible categories and merchants
    categories = [
        'Groceries', 'Dining', 'Transportation', 'Shopping',
        'Utilities', 'Entertainment', 'Transfers', 'Healthcare'
    ]

    merchants_by_cat = {
        'Groceries': ['Walmart', 'Target', 'Whole Foods', 'Kroger', 'Costco'],
        'Dining': ['Starbucks', 'Dominos', 'Subway', 'McDonalds', 'Pizza Hut'],
        'Transportation': ['Uber', 'Lyft', 'Shell', 'Exxon', 'BP Fuel'],
        'Shopping': ['Amazon', 'eBay', 'Best Buy', 'Macy\'s', 'Adidas'],
        'Utilities': ['Comcast', 'AT&T', 'Verizon', 'PG&E', 'Spectrum'],
        'Entertainment': ['Netflix', 'Spotify', 'YouTube Premium', 'Disney+', 'Hulu'],
        'Transfers': ['Payment to John', 'Bank Transfer', 'Zelle Payment', 'Venmo', 'PayPal'],
        'Healthcare': ['CVS Pharmacy', 'Walgreens', 'Doctor Visit', 'Hospital Bill', 'Dental Care']
    }

    # 2. Generate 100 random transactions
    sample_transactions = []
    sample_labels = []

    for i in range(100):
        category = random.choice(categories)
        merchant = random.choice(merchants_by_cat[category])
        amount = round(random.uniform(5, 500), 2)
        timestamp = f"2025-11-{random.randint(1, 7):02d}T{random.randint(6, 22):02d}:{random.randint(0,59):02d}:00Z"
        txn_type = random.choice(['DEBIT', 'CREDIT', 'P2P_TRANSFER'])

        txn = {
            'id': f'txn_{i+1:03d}',
            'description': merchant,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp,
            'type': txn_type
        }

        sample_transactions.append(txn)
        sample_labels.append(category)

    print("Transaction Categorization System - Demo")
    print("=" * 50)
    print(f"Generated {len(sample_transactions)} synthetic transactions.")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    # 3. Create and train system
    system = TransactionCategorizationSystem(categories)

    print("[INFO] Training hybrid model on synthetic data (100 samples)...")
    system.train(
        transactions=sample_transactions,
        labels=sample_labels,
        epochs=12,           # can increase for better results
        batch_size=8,
        learning_rate=0.001
    )

    # 4. Predict on a new transaction
    new_txn = {
        'id': 'txn_051',
        'description': 'Amazon Mktp purchase',
        'amount': 120.00,
        'currency': 'USD',
        'timestamp': '2025-11-07T21:10:00Z',
        'type': 'DEBIT'
    }

    prediction = system.predict(new_txn)
    print("\nPrediction on new transaction:")
    print(json.dumps(prediction, indent=2))