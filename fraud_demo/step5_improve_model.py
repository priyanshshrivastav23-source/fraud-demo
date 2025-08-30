# step5_improve_model.py
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score, accuracy_score
)
import joblib

# -----------------------------
# 1) Load dataset
# -----------------------------
if os.path.exists("dataset_raw.csv"):
    df = pd.read_csv("dataset_raw.csv")
    print("✅ Loaded dataset_raw.csv")
else:
    raise FileNotFoundError("dataset_raw.csv not found! Run preprocess_data.py first.")

# unify label name: FraudLabel -> is_fraud
if "is_fraud" not in df.columns and "FraudLabel" in df.columns:
    df["is_fraud"] = df["FraudLabel"]

# -----------------------------
# 2) Features & Target
# -----------------------------
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Split: train / val / test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)
print(f"Split sizes => train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# -----------------------------
# 3) Preprocessor + Pipeline
# -----------------------------
numeric_features = ["Amount", "Hour", "DayOfWeek"]
categorical_features = ["UserID", "Merchant", "Location", "Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

rf = RandomForestClassifier(random_state=42, class_weight="balanced")

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", rf),
])

# -----------------------------
# 4) Hyperparameter tuning
# -----------------------------
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 3]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=3,
    scoring="average_precision",
    n_jobs=1,   # Windows safe
    verbose=1
)
grid.fit(X_train, y_train)

best_pipe = grid.best_estimator_
print("\n🔹 Best params:", grid.best_params_)
print("🔹 Best CV avg precision (PR-AUC):", round(grid.best_score_, 4))

# -----------------------------
# 5) Pick threshold (maximize F1)
# -----------------------------
val_proba = best_pipe.predict_proba(X_val)[:, 1]
best_f1, best_th = -1.0, 0.5
for t in np.linspace(0.05, 0.95, 19):
    preds = (val_proba >= t).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_th = f1, float(t)

print(f"\n🔹 Chosen threshold (max F1 on val): {best_th:.3f} (F1={best_f1:.3f})")
print("🔹 Validation PR-AUC:", round(average_precision_score(y_val, val_proba), 4))
print("🔹 Validation ROC-AUC:", round(roc_auc_score(y_val, val_proba), 4))

# -----------------------------
# 6) Final test evaluation
# -----------------------------
test_proba = best_pipe.predict_proba(X_test)[:, 1]
test_preds = (test_proba >= best_th).astype(int)

print("\n🔹 Test Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))
print("\n🔹 Test Classification Report:")
print(classification_report(y_test, test_preds, zero_division=0))
print("🔹 Test Accuracy:", round(accuracy_score(y_test, test_preds), 4))
print("🔹 Test PR-AUC:", round(average_precision_score(y_test, test_proba), 4))
print("🔹 Test ROC-AUC:", round(roc_auc_score(y_test, test_proba), 4))

# -----------------------------
# 7) Save pipeline + metadata
# -----------------------------
joblib.dump(best_pipe, "fraud_pipeline.pkl")
meta = {
    "threshold": best_th,
    "feature_order": ["UserID","Amount","Hour","DayOfWeek","Location","Merchant","Type"]
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Saved tuned pipeline to fraud_pipeline.pkl")
print("✅ Saved model metadata to model_meta.json")
