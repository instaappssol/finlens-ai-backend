# Quick Start: Train Your First Model

## Problem
You're seeing `"category_source": "failed"` because **no model has been trained yet**. The system needs a trained model to auto-categorize transactions.

## Solution: Train a Model

You have two options:

### Option 1: Train with CSV File (Recommended)

Upload a CSV file with labeled transactions to train an initial model:

```bash
POST /api/v1/transactions/train-model?version=latest
```

**Request:** Multipart form data with CSV file

**CSV Format:**
```csv
description,amount,transaction_type,date,category
AMAZON PAYMENTS INDIA,899.50,DEBIT,2025-11-12,Shopping
STARBUCKS COFFEE,230.00,DEBIT,2025-11-11,Food & Beverage
IMPS/PAYMENT TO RAKESH,1500.00,DEBIT,2025-11-10,Transfers
HPCL FUEL PUMP,1900.00,DEBIT,2025-11-09,Fuel
BIG BAZAAR GROCERIES,1250.75,DEBIT,2025-11-08,Groceries
```

**Required CSV Columns:**
- `description` (required)
- `amount` (required)
- `category` or `label` (required) - either column name works
- `transaction_type` (optional, defaults to "UNKNOWN")
- `date` (optional)
- `currency` (optional, defaults to "INR")

### Option 2: Upload CSV with Categories, Then Retrain

1. **Upload CSV with categories** (include `category` column):
```csv
description,amount,transaction_type,date,category
Sunny Mardal,1500,DEBIT,2025-11-12,Transfers
SRI LEATHER WEAR,1937,DEBIT,2025-11-10,Shopping
AMAZON PAYMENTS,899.50,DEBIT,2025-11-09,Shopping
STARBUCKS COFFEE,230.00,DEBIT,2025-11-08,Food & Beverage
HPCL FUEL PUMP,1900.00,DEBIT,2025-11-07,Fuel
```

2. **Retrain model from database**:
```bash
POST /api/v1/transactions/retrain-model?model_type=global&min_samples=2
```

This will:
- Find all transactions with manual categories (not auto-categorized)
- Train a model from them
- Save the model for future use

## After Training

Once you've trained a model:
- ✅ New CSV uploads will auto-categorize transactions
- ✅ `category_source` will be `"auto_ml"` instead of `"failed"`
- ✅ Transactions will have `category` and `category_confidence` fields

## Check Model Status

```bash
GET /api/v1/transactions/model-info
```

This will show if a model is loaded and ready.

## Minimum Requirements

- **For training**: At least 2-3 transactions per category
- **For good accuracy**: 10+ transactions per category
- **Categories**: At least 2 different categories

## Example: Complete Workflow

1. **Train initial model** (with CSV file):
```bash
curl -X POST "http://localhost:8000/api/v1/transactions/train-model?version=latest" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@training_data.csv"
```

**training_data.csv:**
```csv
description,amount,transaction_type,date,category
Sunny Mardal,1500,DEBIT,2025-11-12,Transfers
SRI LEATHER WEAR,1937,DEBIT,2025-11-10,Shopping
AMAZON PAYMENTS,899.50,DEBIT,2025-11-09,Shopping
STARBUCKS COFFEE,230.00,DEBIT,2025-11-08,Food & Beverage
```

2. **Upload your CSV** (will now auto-categorize):
```bash
curl -X POST "http://localhost:8000/transactions/upload-transactions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_transactions.csv"
```

3. **Check results** - transactions should now have `"category_source": "auto_ml"`

