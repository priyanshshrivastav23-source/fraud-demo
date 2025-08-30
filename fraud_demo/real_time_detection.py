# real_time_detection.py
import random, time, joblib, json
import pandas as pd

# -----------------------------
# Load trained model & metadata
# -----------------------------
pipeline = joblib.load("fraud_pipeline.pkl")
with open("model_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

THRESHOLD = meta["threshold"]
FEATURE_ORDER = meta["feature_order"]

print(f"✅ Loaded model (threshold={THRESHOLD})")

# -----------------------------
# Simulated transaction generator
# -----------------------------
users = [f"U{str(i).zfill(3)}" for i in range(1, 51)]
merchants = ["Amazon","Flipkart","Myntra","Uber","Dominos",
             "Paytm","PhonePe","BigBazaar"]
locations = ["Delhi","Mumbai","Indore","Kolkata","Chennai","Hyderabad","New York"]
types = ["Online","In-Store","ATM","UPI","Card-Present"]

def generate_transaction():
    return {
        "UserID": random.choice(users),
        "Amount": round(random.uniform(100, 10000), 2),
        "Merchant": random.choice(merchants),
        "Location": random.choice(locations),
        "Type": random.choice(types),
        "Hour": random.randint(0, 23),
        "DayOfWeek": random.randint(0, 6)
    }

# -----------------------------
# Helper: explain why suspicious
# -----------------------------
def explain_reason(tx):
    reasons = []
    if tx["Location"] == "New York":
        reasons.append("Foreign location")
    if tx["Hour"] < 6 or tx["Hour"] > 22:
        reasons.append("Odd hour")
    if tx["Amount"] > 8000:
        reasons.append("Unusually high amount")
    if tx["Type"] == "ATM" and tx["Amount"] > 5000:
        reasons.append("Large ATM withdrawal")
    return reasons if reasons else ["No obvious anomaly"]

# -----------------------------
# Real-time stream simulation
# -----------------------------
for i in range(1, 11):  # generate 10 transactions
    tx = generate_transaction()

    # Keep features in correct order
    df = pd.DataFrame([tx])[FEATURE_ORDER]

    prob = pipeline.predict_proba(df)[0][1]  # fraud probability
    label = 1 if prob >= THRESHOLD else 0

    print(f"\nTransaction {i}: {tx}")
    print(f"Score: {prob:.3f}")

    if label == 1:
        print(f"🚨 ALERT: Suspicious (threshold={THRESHOLD}). Reasons: {', '.join(explain_reason(tx))}")
    else:
        print("✅ Normal transaction")

    time.sleep(1)  # simulate real-time delay
