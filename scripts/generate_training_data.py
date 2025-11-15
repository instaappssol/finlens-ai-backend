"""
Generate High-Confidence Training Data for Transaction Categorization

This script generates training transactions with clear category indicators
to ensure the model learns with high confidence (>0.9).
"""

import csv
import random
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path


# Category templates with high-confidence keywords
CATEGORY_TEMPLATES = {
    "Groceries": [
        "BIG BAZAAR SUPERMARKET",
        "RELIANCE FRESH STORE",
        "DMART GROCERIES",
        "MORE SUPERMARKET",
        "SPENCERS FOOD STORE",
        "FOOD BAZAAR",
        "GROCERY STORE",
        "SUPERMARKET PURCHASE",
    ],
    "Food & Beverage": [
        "STARBUCKS COFFEE",
        "DOMINOS PIZZA",
        "MCDONALDS RESTAURANT",
        "KFC RESTAURANT",
        "CAFE COFFEE DAY",
        "BARISTA COFFEE",
        "PIZZA HUT",
        "RESTAURANT BILL",
        "CAFE PAYMENT",
    ],
    "Fuel": [
        "HPCL FUEL PUMP",
        "INDIAN OIL PETROL",
        "BPCL DIESEL",
        "SHELL FUEL STATION",
        "RELIANCE PETROL PUMP",
        "ESSAR FUEL",
        "GAS STATION",
        "PETROL PUMP",
    ],
    "Transfers": [
        "UPI/PAYMENT TO RAJESH",
        "IMPS/PAYMENT TO PRIYA",
        "NEFT/TRANSFER TO AMIT",
        "PAYMENT TO FRIEND",
        "WALLET TRANSFER",
        "ACCOUNT TRANSFER",
        "P2P TRANSFER",
        "PEER TO PEER PAYMENT",
    ],
    "Shopping": [
        "AMAZON PAYMENTS INDIA",
        "FLIPKART ONLINE",
        "MYNTRA SHOPPING",
        "NYKA FASHION",
        "AJIO RETAIL",
        "SNAPDEAL ECOMMERCE",
        "ONLINE SHOPPING",
        "E-COMMERCE PURCHASE",
    ],
    "Bills & Utilities": [
        "BSES ELECTRICITY BILL",
        "TATA POWER ELECTRICITY",
        "AIRTEL MOBILE BILL",
        "JIO BROADBAND",
        "VODAFONE MOBILE",
        "WATER BILL PAYMENT",
        "ELECTRICITY BILL",
        "MOBILE RECHARGE",
    ],
    "Travel": [
        "UBER RIDE",
        "OLA CAB",
        "MAKEMYTRIP FLIGHT",
        "GOIBIBO HOTEL",
        "IRCTC TRAIN",
        "TAXI BOOKING",
        "HOTEL BOOKING",
        "FLIGHT TICKET",
    ],
}

# Transaction types by category
TRANSACTION_TYPES = {
    "Groceries": ["CARD", "UPI"],
    "Food & Beverage": ["CARD", "UPI"],
    "Fuel": ["CARD", "UPI"],
    "Transfers": ["P2P_TRANSFER", "IMPS", "NEFT", "UPI"],
    "Shopping": ["CARD", "UPI"],
    "Bills & Utilities": ["UPI", "NEFT", "CARD"],
    "Travel": ["CARD", "UPI"],
}

# Amount ranges by category (in INR)
AMOUNT_RANGES = {
    "Groceries": (200, 5000),
    "Food & Beverage": (100, 2000),
    "Fuel": (500, 5000),
    "Transfers": (100, 10000),
    "Shopping": (300, 20000),
    "Bills & Utilities": (100, 5000),
    "Travel": (500, 50000),
}


def generate_transaction(category: str, date: datetime) -> Dict[str, str]:
    """Generate a single transaction with high-confidence category indicators"""
    description = random.choice(CATEGORY_TEMPLATES[category])
    transaction_type = random.choice(TRANSACTION_TYPES[category])
    min_amount, max_amount = AMOUNT_RANGES[category]
    amount = round(random.uniform(min_amount, max_amount), 2)
    
    return {
        "description": description,
        "amount": str(amount),
        "transaction_type": transaction_type,
        "date": date.strftime("%Y-%m-%d"),
        "category": category,
        "currency": "INR",
    }


def generate_training_dataset(
    samples_per_category: int = 10,
    output_file: str = "training_data.csv",
    start_date: datetime = None,
) -> str:
    """
    Generate a CSV file with training transactions.
    
    Args:
        samples_per_category: Number of samples to generate per category
        output_file: Output CSV filename
        start_date: Start date for transactions (defaults to 30 days ago)
    
    Returns:
        Path to generated CSV file
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    
    transactions = []
    categories = list(CATEGORY_TEMPLATES.keys())
    
    # Generate transactions for each category
    for category in categories:
        for i in range(samples_per_category):
            # Distribute dates over the past 30 days
            days_ago = random.randint(0, 30)
            date = start_date + timedelta(days=days_ago)
            transaction = generate_transaction(category, date)
            transactions.append(transaction)
    
    # Shuffle transactions for better training
    random.shuffle(transactions)
    
    # Write to CSV
    output_path = Path(output_file)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["description", "amount", "transaction_type", "date", "category", "currency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
    
    print(f"✅ Generated {len(transactions)} training transactions")
    print(f"   Categories: {len(categories)}")
    print(f"   Samples per category: {samples_per_category}")
    print(f"   Output file: {output_path.absolute()}")
    
    return str(output_path.absolute())


def generate_large_dataset(
    total_samples: int = 200,
    output_file: str = "training_data_large.csv",
) -> str:
    """
    Generate a larger dataset with balanced distribution across categories.
    
    Args:
        total_samples: Total number of samples to generate
        output_file: Output CSV filename
    
    Returns:
        Path to generated CSV file
    """
    categories = list(CATEGORY_TEMPLATES.keys())
    samples_per_category = total_samples // len(categories)
    remainder = total_samples % len(categories)
    
    start_date = datetime.now() - timedelta(days=30)
    transactions = []
    
    # Generate balanced samples
    for idx, category in enumerate(categories):
        # Add extra samples to first few categories if there's a remainder
        count = samples_per_category + (1 if idx < remainder else 0)
        
        for i in range(count):
            days_ago = random.randint(0, 30)
            date = start_date + timedelta(days=days_ago)
            transaction = generate_transaction(category, date)
            transactions.append(transaction)
    
    # Shuffle
    random.shuffle(transactions)
    
    # Write to CSV
    output_path = Path(output_file)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["description", "amount", "transaction_type", "date", "category", "currency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
    
    print(f"✅ Generated {len(transactions)} training transactions")
    print(f"   Categories: {len(categories)}")
    print(f"   Output file: {output_path.absolute()}")
    
    return str(output_path.absolute())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for transaction categorization")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per category (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.csv",
        help="Output CSV filename (default: training_data.csv)",
    )
    parser.add_argument(
        "--large",
        type=int,
        metavar="TOTAL",
        help="Generate large dataset with total number of samples",
    )
    
    args = parser.parse_args()
    
    if args.large:
        generate_large_dataset(args.large, args.output)
    else:
        generate_training_dataset(args.samples, args.output)

