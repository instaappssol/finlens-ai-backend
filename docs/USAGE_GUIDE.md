# Transaction Categorization - Usage Guide

## Overview

This system integrates ML-powered transaction categorization into your FastAPI backend. It uses the model from `ref/transaction_model.py` with persistent storage via `joblib`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Your First Model

Send a POST request to `/api/v1/transactions/train-model`:

```json
{
  "transactions": [
    {
      "transaction_id": "txn_1",
      "description": "AMAZON PAYMENTS INDIA",
      "amount": 899.0,
      "transaction_type": "CARD",
      "currency": "INR",
      "timestamp": "2025-11-07T13:00:00Z",
      "label": "Shopping"
    },
    {
      "transaction_id": "txn_2",
      "description": "STARBUCKS COFFEE",
      "amount": 230.0,
      "transaction_type": "CARD",
      "currency": "INR",
      "timestamp": "2025-11-07T09:00:00Z",
      "label": "Food & Beverage"
    }
  ],
  "version": "latest"
}
```

### 3. Categorize a Transaction

Send a POST request to `/api/v1/transactions/categorize`:

```json
{
  "transaction_id": "txn_999",
  "description": "IMPS/Payment to Rakesh",
  "amount": 1200.0,
  "transaction_type": "P2P_TRANSFER",
  "currency": "INR",
  "timestamp": "2025-11-07T10:30:00Z"
}
```

**Response:**
```json
{
  "message": "Category predicted successfully",
  "errors": [],
  "data": {
    "transaction_id": "txn_999",
    "category": "Transfers",
    "confidence_score": 0.85,
    "model_version": "v0.1.0"
  }
}
```

### 4. Batch Categorization

Send a POST request to `/api/v1/transactions/categorize-batch`:

```json
{
  "transactions": [
    {
      "description": "AMAZON PAYMENTS",
      "amount": 500.0
    },
    {
      "description": "STARBUCKS",
      "amount": 200.0
    }
  ]
}
```

### 5. Get Model Information

Send a GET request to `/api/v1/transactions/model-info`

## API Endpoints

### POST `/api/v1/transactions/categorize`
Predict category for a single transaction.

**Request Body:**
- `description` (required): Transaction description
- `amount` (required): Transaction amount
- `transaction_id` (optional): Unique transaction ID
- `transaction_type` (optional): Type of transaction
- `currency` (optional): Currency code
- `timestamp` (optional): Transaction timestamp

### POST `/api/v1/transactions/categorize-batch`
Predict categories for multiple transactions.

**Request Body:**
- `transactions`: Array of transaction objects

### POST `/api/v1/transactions/train-model`
Train a new categorization model.

**Request Body:**
- `transactions`: Array of labeled transactions (must include `label` field)
- `version` (optional): Model version string (default: "latest")

### GET `/api/v1/transactions/model-info`
Get metadata about the current model.

## Model Storage

- **Location**: `models/` directory (configurable via `MODELS_DIR` env var)
- **Format**: `.joblib` files (efficient for scikit-learn models)
- **Metadata**: Stored in `models/model_metadata.json`
- **Versioning**: Supports multiple model versions

## Model Manager Service

The `ModelManager` class handles:
- Saving models with metadata
- Loading models by name and version
- Tracking model versions
- Managing model lifecycle

**Example:**
```python
from app.services.model_manager import ModelManager

manager = ModelManager(models_dir="models")

# Save a model
manager.save_model(
    model=trained_model,
    model_name="transaction_categorizer",
    version="v1.0.0",
    metadata={"accuracy": 0.85, "training_samples": 1000}
)

# Load a model
model = manager.load_model("transaction_categorizer", version="v1.0.0")
```

## Categorization Service

The `CategorizationService` wraps the model for API use:

```python
from app.services.categorization_service import CategorizationService

service = CategorizationService(models_dir="models")

# Load existing model
service.load_model()

# Predict
result = service.predict({
    "description": "AMAZON PAYMENTS",
    "amount": 500.0
})

# Train new model
service.train_model(
    training_data=[...],
    labels=[...],
    save_model=True,
    version="v1.0.0"
)
```

## Supported Categories

Default categories (from `transaction_model.py`):
- Groceries
- Food & Beverage
- Fuel
- Transfers
- Shopping
- Bills & Utilities
- Travel

You can customize categories by training with your own labeled data.

## Troubleshooting

### Model Not Found
If you get "Model not loaded" error:
1. Train a model first using `/train-model` endpoint
2. Ensure `models/` directory exists and is writable
3. Check `model_metadata.json` for available models

### Import Errors
If `TransactionCategorizer` import fails:
1. Ensure `ref/transaction_model.py` exists
2. Check that all dependencies are installed (scikit-learn, pandas, numpy, joblib)
3. Verify Python path includes the `ref/` directory

### Low Confidence Scores
- Train with more diverse data
- Ensure training data includes examples from all categories
- Check that transaction descriptions are similar to training data

## Next Steps

1. **Collect Training Data**: Gather labeled transactions from your database
2. **Train Initial Model**: Use `/train-model` with your data
3. **Evaluate Performance**: Check confidence scores and adjust
4. **Retrain Periodically**: Update model with new data
5. **Monitor**: Track prediction accuracy in production

