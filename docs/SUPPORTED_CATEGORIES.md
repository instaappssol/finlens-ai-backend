# Supported Transaction Categories

## Default Categories (from transaction_model.py)

The system comes with **7 default categories** defined in the model:

1. **Groceries**
   - Examples: Supermarket, food stores, grocery stores
   - Keywords: groceries, supermarket, food_store

2. **Food & Beverage**
   - Examples: Restaurants, cafes, coffee shops
   - Keywords: restaurant, cafe, coffee

3. **Fuel**
   - Examples: Gas stations, petrol pumps
   - Keywords: fuel, gas, petrol, diesel

4. **Transfers**
   - Examples: P2P transfers, wallet transfers, account transfers
   - Keywords: p2p, peer-to-peer, wallet_transfer, account_transfer

5. **Shopping**
   - Examples: E-commerce, retail stores, online shopping
   - Keywords: ecommerce, retail, amazon, flipkart

6. **Bills & Utilities**
   - Examples: Electricity, water, mobile, broadband bills
   - Keywords: electricity, water, mobile, broadband

7. **Travel**
   - Examples: Flights, hotels, taxis, ride-sharing
   - Keywords: flight, hotel, taxi, uber, ola

## Important Note: Categories are Flexible

⚠️ **The model can learn ANY categories you train it with!**

The default taxonomy is just a starting point. When you train the model with your own labeled data, it will learn to predict whatever categories you provide.

### Example: Custom Categories

If you train with these categories:
- "Personal"
- "Business"
- "Investment"
- "Charity"

The model will learn to predict these instead of (or in addition to) the default ones.

## How Categories Work

1. **Default Model**: If you train without specifying categories, it uses the default taxonomy
2. **Custom Training**: When you provide labeled data with categories, the model learns those specific categories
3. **Dynamic Learning**: Each time you retrain, the model adapts to the categories in your training data

## Current Implementation

The system uses `ref/transaction_model.py` which has this DEFAULT_TAXONOMY:

```python
DEFAULT_TAXONOMY = {
    "Groceries": ["groceries", "supermarket", "food_store"],
    "Food & Beverage": ["restaurant", "cafe", "coffee"],
    "Fuel": ["fuel", "gas", "petrol", "diesel"],
    "Transfers": ["p2p", "peer-to-peer", "wallet_transfer", "account_transfer"],
    "Shopping": ["ecommerce", "retail", "amazon", "flipkart"],
    "Bills & Utilities": ["electricity", "water", "mobile", "broadband"],
    "Travel": ["flight", "hotel", "taxi", "uber", "ola"],
}
```

## Recommendations

1. **Start with Defaults**: Use the 7 default categories for initial training
2. **Customize as Needed**: Add or modify categories based on your specific use case
3. **Consistent Labels**: Use the exact same category names in all training data
4. **Minimum Samples**: Have at least 2-3 examples per category for training

## Example Usage

When training, you can use any of these categories (or create your own):

```json
{
  "transactions": [
    {"description": "AMAZON", "amount": 500, "label": "Shopping"},
    {"description": "STARBUCKS", "amount": 200, "label": "Food & Beverage"},
    {"description": "HPCL FUEL", "amount": 1000, "label": "Fuel"},
    {"description": "PAYMENT TO JOHN", "amount": 500, "label": "Transfers"}
  ]
}
```

The model will learn to predict these exact category names.

