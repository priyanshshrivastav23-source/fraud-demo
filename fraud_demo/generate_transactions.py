import random
import pandas as pd
from datetime import datetime, timedelta

# Number of transactions to simulate
num_transactions = 1000

# Example users and merchants
users = ["U001", "U002", "U003", "U004", "U005"]
merchants = ["Amazon", "Flipkart", "Myntra", "Paytm", "Swiggy", "Zomato", "Uber", "Ola"]

# Function to create random transactions
def generate_transactions(n):
    data = []
    start_time = datetime.now() - timedelta(days=30)  # simulate last 30 days

    for i in range(n):
        transaction_id = f"T{i+1:04d}"
        user_id = random.choice(users)
        amount = round(random.uniform(100, 50000), 2)  # amount between 100 and 50,000
        timestamp = start_time + timedelta(minutes=random.randint(1, 43200))  # random time in 30 days
        merchant = random.choice(merchants)
        location = random.choice(["Mumbai", "Delhi", "Bangalore", "Indore", "New York", "London"])
        txn_type = random.choice(["Online", "In-Store", "ATM", "UPI"])

        # Simulate suspicious behavior (label=1) if unusual
        if amount > 40000 or location in ["New York", "London"]:
            fraud = 1
        else:
            fraud = 0

        data.append([transaction_id, user_id, amount, timestamp, merchant, location, txn_type, fraud])

    return pd.DataFrame(data, columns=["TransactionID", "UserID", "Amount", "Timestamp", 
                                       "Merchant", "Location", "Type", "FraudLabel"])

# Generate transactions
df = generate_transactions(num_transactions)

# Save to CSV
df.to_csv("transactions.csv", index=False)
print("✅ Transaction data generated and saved to transactions.csv")
print(df.head(10))  # show first 10 rows
