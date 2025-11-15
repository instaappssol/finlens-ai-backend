# CSV Upload Format Guide

## Required Fields

The CSV file must include at minimum:
- **description** (required): Transaction description/merchant name
- **amount** (required): Transaction amount (numeric)

## Optional Fields

- **transaction_type**: Type of transaction (e.g., `CARD`, `UPI`, `P2P_TRANSFER`, `NEFT`, `IMPS`)
- **currency**: Currency code (e.g., `INR`, `USD`) - defaults to `INR` if not provided
- **date**: Transaction date in format `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`
- **category**: Pre-assigned category (if provided, won't be auto-categorized)

## CSV Format Rules

1. **Header Row**: First row must contain column names
2. **Encoding**: UTF-8 encoding
3. **Delimiter**: Comma (`,`)
4. **Quotes**: Use quotes for fields containing commas or special characters
5. **Empty Fields**: Can be left empty (will use defaults)

## Sample CSV Files

- `sample_transactions.csv` - Transactions without categories (will be auto-categorized)
- `sample_transactions_with_categories.csv` - Transactions with pre-assigned categories

## Field Descriptions

### description
- **Required**: Yes
- **Type**: String
- **Example**: "AMAZON PAYMENTS INDIA", "STARBUCKS COFFEE"
- **Notes**: This is the primary field used for categorization

### amount
- **Required**: Yes
- **Type**: Numeric (float)
- **Example**: `899.50`, `1500.00`
- **Notes**: Must be a positive number

### transaction_type
- **Required**: No
- **Type**: String
- **Common Values**: 
  - `CARD` - Credit/Debit card transactions
  - `UPI` - UPI payments
  - `P2P_TRANSFER` - Peer-to-peer transfers
  - `NEFT` - NEFT transfers
  - `IMPS` - IMPS transfers
- **Default**: `UNKNOWN` if not provided

### currency
- **Required**: No
- **Type**: String (3-letter code)
- **Example**: `INR`, `USD`, `EUR`
- **Default**: `INR` if not provided

### date
- **Required**: No
- **Type**: String (ISO format)
- **Formats**: 
  - `YYYY-MM-DD` (e.g., `2025-01-15`)
  - `YYYY-MM-DDTHH:MM:SS` (e.g., `2025-01-15T14:30:00`)
  - `YYYY-MM-DDTHH:MM:SSZ` (e.g., `2025-01-15T14:30:00Z`)
- **Notes**: Used for temporal feature extraction

### category
- **Required**: No
- **Type**: String
- **Common Categories**:
  - `Groceries`
  - `Food & Beverage`
  - `Fuel`
  - `Transfers`
  - `Shopping`
  - `Bills & Utilities`
  - `Travel`
- **Notes**: 
  - If provided, transaction won't be auto-categorized
  - If empty, ML model will predict the category
  - Used as training data if `category_source` is not "auto_ml"

## Auto-Categorization Behavior

When you upload a CSV:
1. Transactions **without** a category → Auto-categorized by ML model
2. Transactions **with** a category → Category preserved (used for training)
3. Failed categorizations → Saved without category, can be manually labeled later

## Example CSV Content

```csv
description,amount,transaction_type,currency,date,category
AMAZON PAYMENTS INDIA,899.50,CARD,INR,2025-01-15,
STARBUCKS COFFEE,230.00,CARD,INR,2025-01-15,
IMPS/PAYMENT TO RAKESH,1500.00,P2P_TRANSFER,INR,2025-01-14,
HPCL FUEL PUMP,1900.00,CARD,INR,2025-01-14,
```

## Upload Process

1. **Prepare CSV**: Ensure required fields (description, amount) are present
2. **Upload**: POST to `/api/v1/transactions/upload-transactions`
3. **Response**: Includes:
   - Number of transactions inserted
   - Sample of inserted transactions
   - Categorization statistics
   - Training data availability info

## Tips

1. **For Training**: Include `category` field with manual labels to build training data
2. **For Auto-Categorization**: Leave `category` empty to let ML model predict
3. **Mixed Approach**: Some transactions with categories, some without - both work!
4. **Date Format**: Use ISO format for best compatibility
5. **Amount Format**: Use decimal numbers (e.g., `899.50` not `899.5`)

## Common Issues

- **Missing description**: Will fail validation
- **Invalid amount**: Must be numeric and positive
- **Wrong date format**: May not extract temporal features correctly
- **Encoding issues**: Ensure CSV is UTF-8 encoded

