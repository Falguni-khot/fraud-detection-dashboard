"""
Financial Fraud Detection Dashboard
====================================
Run with: streamlit run app.py

Requires Model/ folder produced by train_model.py (run that first).
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Sidebar background - dark navy blue */
    section[data-testid="stSidebar"] {
        background-color: #0a1f44;
    }

    /* Sidebar title ("Fraud Detection" headline) */
    section[data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
    }

    /* All radio option text - make white and bigger */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* Ensure the currently selected option's text is fully white and bold */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* Border around KPI metric cards - fixed height so all cards match */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #0a1f44;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Sidebar collapse/expand button - white circle so the icon is visible either way */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="collapsedControl"] button {
        background-color: #ffffff !important;
        border-radius: 50% !important;
    }
    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg,
    [data-testid="collapsedControl"] svg {
        color: #0a1f44 !important;
        fill: #0a1f44 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Model/model.pkl")
    scaler = joblib.load("Model/scaler.pkl")
    feature_columns = joblib.load("Model/feature_columns.pkl")
    with open("Model/metadata.json") as f:
        metadata = json.load(f)
    df = pd.read_csv("Model/cleaned_data.csv")
    return model, scaler, feature_columns, metadata, df


model, scaler, feature_columns, metadata, df = load_artifacts()

NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
CATEGORICAL_VALUES = metadata["categorical_values"]

# ---------- Sidebar navigation ----------
st.sidebar.title("Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore Data", "Model Performance", "Live Prediction"],
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Financial Fraud Detection Dashboard")
    st.caption("Overview of transaction data and fraud rates")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{metadata['n_rows']:,}")
    col2.metric("Fraud Rate", f"{metadata['fraud_rate']*100:.2f}%")
    col3.metric("Fraudulent Transactions", f"{int(metadata['fraud_rate']*metadata['n_rows']):,}")

    st.markdown("---")

    st.subheader("Fraud Rate by Merchant Category")
    fraud_by_cat = df.groupby("Merchant_Category")["Fraudulent"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fraud_by_cat.plot(kind="bar", ax=ax, color="#c0392b")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("")
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    st.subheader("Transaction Amount: Fraud vs. Legitimate")
    fig2, ax2 = plt.subplots(figsize=(6, 3.3))
    df[df["Fraudulent"] == 0]["Transaction_Amount"].plot(
        kind="hist", bins=40, alpha=0.6, label="Legitimate", ax=ax2, color="#2980b9"
    )
    df[df["Fraudulent"] == 1]["Transaction_Amount"].plot(
        kind="hist", bins=40, alpha=0.6, label="Fraud", ax=ax2, color="#c0392b"
    )
    ax2.set_xlabel("Transaction Amount")
    ax2.legend()
    st.pyplot(fig2, use_container_width=False)

    # ---- THE BUSINESS IMPACT BLOCK  ----
    st.markdown("---")

    st.subheader("Business Impact")

    avg_fraud_amount = df[df["Fraudulent"] == 1]["Transaction_Amount"].mean()
    investigation_cost_per_case = 15

    fraud_count = int(metadata["fraud_rate"] * metadata["n_rows"])
    results = metadata["results"][metadata["best_model"]]
    recall = results["recall_fraud"]
    precision = results["precision_fraud"]

    fraud_caught = int(fraud_count * recall)
    fraud_missed = fraud_count - fraud_caught
    total_flagged = int(fraud_caught / precision) if precision > 0 else 0
    false_alarms = total_flagged - fraud_caught

    money_saved = fraud_caught * avg_fraud_amount
    money_lost_to_missed_fraud = fraud_missed * avg_fraud_amount
    investigation_cost = total_flagged * investigation_cost_per_case

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Caught", f"{fraud_caught} / {fraud_count}", f"{recall*100:.0f}% recall")
    c2.metric("Estimated Value Saved", f"${money_saved:,.0f}")
    c3.metric("False Alarms to Investigate", f"{false_alarms}", f"~${investigation_cost:,.0f} cost")

    st.caption(
        f"Assumes an average fraud transaction of \\${avg_fraud_amount:.0f} and "
        f"\\${investigation_cost_per_case} analyst cost per flagged case reviewed. "
        f"At the current threshold, the model misses {fraud_missed} fraud cases "
        f"(~\\${money_lost_to_missed_fraud:,.0f} in unstopped fraud) in exchange for "
        f"{false_alarms} false alarms analysts must review."
    )

# ============================================================
# PAGE 2: EXPLORE DATA
# ============================================================
elif page == "Explore Data":
    st.title("Explore the Data")

    dimension = st.selectbox(
        "Break down fraud rate by:",
        CATEGORICAL_COLS + ["Suspicious_Keyword", "Is_International"],
    )

    fraud_by_dim = df.groupby(dimension)["Fraudulent"].agg(["mean", "count"])
    fraud_by_dim.columns = ["Fraud Rate", "Transaction Count"]
    fraud_by_dim = fraud_by_dim.sort_values("Fraud Rate", ascending=False)

    st.dataframe(
        fraud_by_dim.style.format({"Fraud Rate": "{:.2%}"}),
        width='stretch',
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    fraud_by_dim["Fraud Rate"].plot(kind="bar", ax=ax, color="#8e44ad")
    ax.set_ylabel("Fraud rate")
    st.pyplot(fig, use_container_width=False)

    st.subheader("Raw data sample")
    st.dataframe(df.sample(20, random_state=1), width='stretch')

# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.caption(f"Best model: **{metadata['best_model'].replace('_', ' ').title()}**")

    results = metadata["results"][metadata["best_model"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision (Fraud)", f"{results['precision_fraud']:.2f}")
    col2.metric("Recall (Fraud)", f"{results['recall_fraud']:.2f}")
    col3.metric("F1 (Fraud)", f"{results['f1_fraud']:.2f}")
    col4.metric("ROC-AUC", f"{results['roc_auc']:.2f}")

    st.info(
        "Note: precision/recall/F1 are shown for the **fraud class only** — "
        "these matter far more than overall accuracy on an imbalanced dataset like this one."
    )

    st.markdown("---")

    st.subheader("Confusion Matrix")
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Legit", "Predicted: Fraud"])
    ax.set_yticklabels(["Actual: Legit", "Actual: Fraud"])
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Precision-Recall Curve")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.plot(metadata["pr_curve"]["recall"], metadata["pr_curve"]["precision"])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        st.pyplot(fig2, use_container_width=False)

    with col_b:
        st.subheader("ROC Curve")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.plot(metadata["roc_curve"]["fpr"], metadata["roc_curve"]["tpr"])
        ax3.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        st.pyplot(fig3, use_container_width=False)

    st.markdown("---")

    st.subheader("Top Features Driving Predictions")
    top_features = metadata["feature_importance"][:10]
    feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#27ae60")
    st.pyplot(fig4, use_container_width=False)

# ============================================================
# PAGE 4: LIVE PREDICTION
# ============================================================
elif page == "Live Prediction":
    st.title("Live Fraud Prediction")
    st.caption("Enter transaction details to get a fraud probability from the trained model")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            avg_spend = st.number_input("Customer's Average Spend", min_value=0.0, value=80.0)
            account_age = st.number_input("Account Age (days)", min_value=0, value=500)
            prev_transactions = st.number_input("Previous Transactions", min_value=0, value=20)
            is_international = st.selectbox("International Transaction?", ["No", "Yes"])

        with c2:
            merchant_category = st.selectbox("Merchant Category", CATEGORICAL_VALUES["Merchant_Category"])
            payment_method = st.selectbox("Payment Method", CATEGORICAL_VALUES["Payment_Method"])
            device_type = st.selectbox("Device Type", CATEGORICAL_VALUES["Device_Type"])
            location = st.selectbox("Location", CATEGORICAL_VALUES["Location"])
            suspicious_keyword = st.selectbox("Suspicious Keyword Detected?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = {
            "Transaction_Amount": amount,
            "Is_International": 1 if is_international == "Yes" else 0,
            "Previous_Transactions": prev_transactions,
            "Average_Spend": avg_spend,
            "Account_Age_Days": account_age,
            "Suspicious_Keyword": 1 if suspicious_keyword == "Yes" else 0,
            "Merchant_Category": merchant_category,
            "Payment_Method": payment_method,
            "Device_Type": device_type,
            "Location": location,
        }
        input_df = pd.DataFrame([raw_input])
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS)

        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        input_encoded[NUMERIC_COLS] = scaler.transform(input_encoded[NUMERIC_COLS])

        proba = model.predict_proba(input_encoded)[0, 1]
        prediction = "FRAUD" if proba >= 0.5 else "LEGITIMATE"

        st.markdown("---")
        if prediction == "FRAUD":
            st.error(f"Prediction: **{prediction}**  —  Fraud probability: {proba:.1%}")
        else:
            st.success(f"Prediction: **{prediction}**  —  Fraud probability: {proba:.1%}")

        st.progress(min(int(proba * 100), 100))