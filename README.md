# Financial Fraud Detection Model + Dashboard

An end-to-end machine learning project that detects fraudulent financial
transactions and presents the results through an interactive Streamlit
dashboard.

🔗 **Live app:** https://fraud-detection-dashboard-bfqhhfpz3xuw2qj9qdrwta.streamlit.app/

## Overview

Fraud detection is an imbalanced classification problem — in this dataset,
only 9.64% of transactions are fraudulent. This project handles that
imbalance properly, evaluates the model on metrics that actually matter for
fraud detection (not just accuracy), and presents the results through a
dashboard usable by non-technical stakeholders.

## Dataset

`data/financial_fraud_detection_dataset.csv` — 5,000 transactions, 9.64%
fraud rate. Features include merchant category, payment method, device type,
location, international transaction flag, customer transaction history,
account age, and a suspicious-keyword indicator.

## Approach

- **Preprocessing:** dropped identifier columns to avoid data leakage,
  one-hot encoded categorical features, scaled numeric features using
  `StandardScaler` fit only on the training set
- **Handling imbalance:** used `class_weight="balanced"` in both models
  rather than SMOTE or undersampling, to avoid synthesizing or discarding data
- **Models trained:** Logistic Regression and Random Forest
- **Model selection:** chose the best model by **recall on the fraud class**,
  tie-broken by ROC-AUC — catching fraud matters more than raw accuracy in
  this context
- **Evaluation:** precision, recall, F1, ROC-AUC, and confusion matrix,
  all reported for the fraud class specifically

## Dashboard

Built with Streamlit, the dashboard includes:

- **Overview** — transaction volume, fraud rate, category-level fraud trends,
  and a Business Impact section translating model performance into estimated
  value saved, fraud missed, and analyst investigation cost
- **Explore Data** — fraud rate broken down by any categorical dimension
- **Model Performance** — confusion matrix, precision-recall curve, ROC
  curve, and feature importance
- **Live Prediction** — input transaction details and receive a real-time
  fraud probability from the trained model

## How to run locally
1. Install dependencies:
                        pip install -r requirements.txt
2. Train the model (only needed once, or after changing the data/pipeline):
                        python train_model.py
3. Launch the dashboard:
                        streamlit run app.py


The dashboard opens at `http://localhost:8501`.

## Tech stack

Python · pandas · scikit-learn · Streamlit · matplotlib

## Key results

- Best model: Logistic Regression
- Recall (fraud class): 64%
- ROC-AUC: 0.785

## Future improvements

- Add SHAP values for per-prediction explainability
- Add an adjustable decision threshold to visualize the precision-recall
  tradeoff
- Validate the pipeline on a larger, more imbalanced dataset to test
  scalability
- Explore gradient boosting models (XGBoost, LightGBM)
