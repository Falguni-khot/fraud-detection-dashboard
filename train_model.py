"""
Financial Fraud Detection - Model Training Script
===================================================
Loads the transaction dataset, cleans it, engineers features,
trains a Logistic Regression baseline and a Random Forest model,
evaluates both properly (not with plain accuracy), and saves the
best model + preprocessing objects for the Streamlit dashboard to use.

Run with:  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

DATA_PATH = "data/financial_fraud_detection_dataset.csv"
MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL_COLS = ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]
NUMERIC_COLS = [
    "Transaction_Amount",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
]
BINARY_MAP_COL = "Suspicious_Keyword"
ID_COLS = ["Transaction_ID", "Customer_ID", "Transaction_Date"]
TARGET_COL = "Fraudulent"


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    df[BINARY_MAP_COL] = df[BINARY_MAP_COL].map({"Yes": 1, "No": 0})
    df = df.dropna()
    return df


def build_features(df):
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df_encoded


def main():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    print(f"  {len(df)} rows after cleaning")
    print(f"  Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")

    df_encoded = build_features(df)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted_models = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)

        report = classification_report(y_test, preds, output_dict=True)
        roc_auc = roc_auc_score(y_test, proba)

        print(f"--- {name} ---")
        print(classification_report(y_test, preds))
        print(f"ROC-AUC: {roc_auc:.4f}\n")

        results[name] = {
            "precision_fraud": report["1"]["precision"],
            "recall_fraud": report["1"]["recall"],
            "f1_fraud": report["1"]["f1-score"],
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted_models[name] = model

    best_name = max(
        results, key=lambda n: (results[n]["recall_fraud"], results[n]["roc_auc"])
    )
    best_model = fitted_models[best_name]
    print(f"Best model selected: {best_name}")

    proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba_best)
    fpr, tpr, _ = roc_curve(y_test, proba_best)

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
    feature_importance = sorted(
        zip(feature_columns, importances.tolist()), key=lambda x: -x[1]
    )

    joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")

    metadata = {
        "best_model": best_name,
        "results": results,
        "feature_importance": feature_importance,
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "fraud_rate": float(df[TARGET_COL].mean()),
        "n_rows": int(len(df)),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_values": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_COLS},
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    df.to_csv(f"{MODEL_DIR}/cleaned_data.csv", index=False)

    print(f"\nSaved model, scaler, and metadata to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()