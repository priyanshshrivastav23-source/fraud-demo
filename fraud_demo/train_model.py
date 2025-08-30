# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# 1. Generate dummy dataset
data = pd.DataFrame({
    "UserID": np.arange(100),
    "Amount": np.random.uniform(10, 5000, 100),
    "Location": np.random.choice(["Delhi", "Mumbai", "Bangalore"], 100),
    "Device": np.random.choice(["Mobile", "Desktop"], 100),
    "Merchant": np.random.choice(["Amazon", "Flipkart", "Paytm"], 100),
    "Label": np.random.choice([0, 1], 100, p=[0.8, 0.2])  # 0 = Safe, 1 = Fraud
})

# 2. Features and target
X = data.drop(columns=["Label"])
y = data["Label"]

# 3. Preprocessing
numeric_features = ["Amount"]
numeric_transformer = StandardScaler()

categorical_features = ["Location", "Device", "Merchant"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 4. Pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train
clf.fit(X_train, y_train)

# 7. Save model properly with pickle
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved as fraud_model.pkl")
