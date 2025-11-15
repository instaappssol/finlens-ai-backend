# Transaction Categorization Models Explanation

## Overview
Both reference files implement AI-powered transaction categorization systems. They differ in their ML approach:

### `transaction_model.py` - **Baseline Approach (Scikit-learn)**
- **Technology**: Scikit-learn (TF-IDF + Logistic Regression)
- **Complexity**: Simpler, faster training
- **Features**: 
  - Text preprocessing with TF-IDF vectorization
  - Tabular features (amount, hour, transaction type, currency)
  - Logistic Regression classifier
  - Built-in save/load using `joblib`
- **Best for**: Quick prototyping, smaller datasets, production-ready baseline

### `txn_categorization_model.py` - **Advanced Approach (PyTorch + Transformers)**
- **Technology**: PyTorch + Sentence Transformers
- **Complexity**: More sophisticated, requires GPU for best performance
- **Features**:
  - Sentence transformer embeddings for text (all-MiniLM-L6-v2)
  - Hybrid neural network (text + tabular pathways)
  - Cyclical encoding for temporal features
  - Deep learning classifier
- **Best for**: Better accuracy, larger datasets, when you have GPU resources

---

## Key Components Breakdown

### 1. **TransactionPreprocessor**
Both files have this component:
- **Purpose**: Clean and normalize transaction descriptions
- **Functions**:
  - `clean_text()`: Removes prefixes, punctuation, normalizes case
  - `normalize_merchant()`: Maps merchant names to canonical forms
  - `enrich()` / `preprocess()`: Adds derived features (hour, is_p2p, etc.)

### 2. **Feature Engineering**
- **transaction_model.py**: Uses `FeatureBuilder` with ColumnTransformer
  - TF-IDF for text (max 5000 features, 1-2 grams)
  - OneHotEncoder for categoricals
  - StandardScaler for numericals
  
- **txn_categorization_model.py**: Uses `FeatureEngineering` class
  - Sentence transformer for text embeddings (384-dim vectors)
  - Cyclical encoding for time features (sin/cos)
  - Label encoding for categoricals
  - Log transform + scaling for amounts

### 3. **Model Architecture**

**transaction_model.py**:
```python
LogisticRegression(max_iter=200)
# Simple linear classifier on concatenated features
```

**txn_categorization_model.py**:
```python
HybridTransactionClassifier
├── Text Pathway: Linear layers (384 → 256 → 128)
├── Tabular Pathway: Linear layers (tabular_dim → 256 → 128)
└── Fusion: Concatenate → 256 → 128 → num_classes
```

### 4. **Prediction Output**
Both return similar structures:
```python
{
    "transaction_id": "...",
    "category": "Groceries",
    "confidence_score": 0.85,
    "model_version": "v0.1.0"
}
```

---

## Model Persistence Strategy

### Recommended Approach: **joblib** (already in transaction_model.py)
- **Pros**: 
  - Efficient for scikit-learn models
  - Handles numpy arrays well
  - Already implemented in `transaction_model.py`
- **Cons**: 
  - Less flexible for PyTorch models

### Alternative: **pickle**
- **Pros**: 
  - Universal Python serialization
  - Works with any Python object
- **Cons**: 
  - Security concerns (can execute arbitrary code)
  - Larger file sizes
  - Version compatibility issues

### For PyTorch Models: **torch.save()**
- **Pros**: 
  - Native PyTorch serialization
  - Handles GPU/CPU state dicts
- **Cons**: 
  - PyTorch-specific

---

## Integration Plan

1. **Create model storage directory**: `models/` (gitignored)
2. **Create ModelManager service**: Handles save/load operations
3. **Create CategorizationService**: Wraps the model for predictions
4. **Add endpoints**: 
   - `/transactions/categorize` - Predict category for transaction
   - `/transactions/train-model` - Train new model from data
   - `/transactions/model-info` - Get model metadata

---

## File Structure
```
app/
├── services/
│   ├── categorization_service.py  # Main service using the model
│   └── model_manager.py           # Handles save/load
├── models/                         # Storage for .pkl/.joblib files
│   └── .gitkeep
└── core/
    └── config.py                   # Add MODEL_PATH config
```

