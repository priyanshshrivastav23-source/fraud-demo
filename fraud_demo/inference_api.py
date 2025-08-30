from fastapi import FastAPI
import joblib
import json
import pandas as pd
import os

app = FastAPI()

# -------------------------
# 1. Load model & metadata
# -------------------------
model_path = "models/fraud_model.pkl"
meta_path = "models/model_meta.json"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model file not found! Run train_model.py first.")

if not os.path.exists(meta_path):
    raise FileNotFoundError("❌ Metadata file not found! Run train_model.py first.")

# Load model
model = joblib.load(model_path)

# Load metadata correctly
with open(meta_path, "r") as f:
    meta = json.load(f)

@app.get("/")
def home():
    return {"message": "🚀 Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return {
        "prediction": int(prediction),
        "fraud_probability": float(proba)
    }
