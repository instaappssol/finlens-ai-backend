# Training Data Generation Scripts

## generate_training_data.py

Generates high-confidence training transactions for the categorization model.

### Features

- ✅ **High Confidence**: Uses clear category keywords (>0.9 confidence expected)
- ✅ **Balanced Data**: Equal samples per category
- ✅ **Realistic**: Realistic amounts, dates, and transaction types
- ✅ **CSV Format**: Ready to upload via `/train-model` endpoint

### Usage

#### Basic Usage (10 samples per category = 70 total)
```bash
python scripts/generate_training_data.py
```

#### Custom Samples Per Category
```bash
python scripts/generate_training_data.py --samples 20
# Generates 20 samples × 7 categories = 140 transactions
```

#### Large Dataset
```bash
python scripts/generate_training_data.py --large 200
# Generates 200 total samples, balanced across categories
```

#### Custom Output File
```bash
python scripts/generate_training_data.py --output my_training_data.csv --samples 15
```

### Generated Categories

The script generates data for all 7 default categories:

1. **Groceries** - Supermarket, grocery stores
2. **Food & Beverage** - Restaurants, cafes
3. **Fuel** - Petrol pumps, gas stations
4. **Transfers** - P2P transfers, IMPS, NEFT
5. **Shopping** - E-commerce, retail
6. **Bills & Utilities** - Electricity, mobile, broadband
7. **Travel** - Uber, Ola, flights, hotels

### CSV Format

Generated CSV includes:
- `description` - Clear merchant/transaction description
- `amount` - Realistic amount for category
- `transaction_type` - CARD, UPI, P2P_TRANSFER, etc.
- `date` - Random date in past 30 days
- `category` - Category label (for training)
- `currency` - INR

### Example Output

```csv
description,amount,transaction_type,date,category,currency
AMAZON PAYMENTS INDIA,1250.50,CARD,2025-11-10,Shopping,INR
STARBUCKS COFFEE,350.00,UPI,2025-11-08,Food & Beverage,INR
HPCL FUEL PUMP,2000.00,CARD,2025-11-12,Fuel,INR
```

### Training the Model

After generating the CSV:

```bash
# Upload to train-model endpoint
curl -X POST "http://localhost:8080/api/v1/transactions/train-model?version=latest" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@training_data.csv"
```

### Tips for High Confidence

1. **Use Clear Keywords**: The script uses category-specific keywords
2. **Balanced Data**: Equal samples per category
3. **Realistic Amounts**: Amounts match category expectations
4. **Proper Transaction Types**: Transaction types match category patterns

### Minimum Requirements

- At least **2 samples per category** (14 total minimum)
- Recommended: **10+ samples per category** (70+ total) for good accuracy

