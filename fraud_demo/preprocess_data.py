# preprocess_data.py
import pandas as pd

# 1) Load original transactions
df = pd.read_csv("transactions.csv", parse_dates=["Timestamp"])

# 2) Derive time features (keep raw strings for categories)
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek  # 0=Mon ... 6=Sun

# 3) Keep a clean, raw dataset (NO manual encoding or scaling)
cols = ["UserID", "Amount", "Merchant", "Location", "Type", "Hour", "DayOfWeek", "FraudLabel"]
df_out = df[cols].copy()

print("🔹 Preview of raw dataset (no scaling/encoding):")
print(df_out.head(5))
print("\n🔹 Shapes => X:", df_out.shape, "  y:", df_out["FraudLabel"].shape)

# 4) Save for training
df_out.to_csv("dataset_raw.csv", index=False)
print("\n✅ Saved raw dataset to dataset_raw.csv")
