import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("model/fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("💳 Fraud Detection System (Prototype)")
st.write("Upload a transaction file or enter transaction details to check fraud risk.")

# ---------- Helper: Scaling ----------
scaler = StandardScaler()

def preprocess_data(df):
    # Handle scaling for Time & Amount if present
    if "Amount" in df.columns and "Time" in df.columns:
        df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df

# ---------- Option 1: Upload CSV ----------
st.header("📂 Upload Transaction CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop Class column if present
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    # Preprocess
    data = preprocess_data(data)

    # Predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    # Add results
    results = data.copy()
    results["Fraud Prediction"] = predictions
    results["Fraud Probability"] = probabilities

    st.subheader("🔍 Results (first 20 shown)")
    st.dataframe(results.head(20))

    fraud_count = sum(predictions)
    st.success(f"⚠️ Fraudulent Transactions Detected: {fraud_count}")

# ---------- Option 2: Manual Input ----------
st.header("✍️ Manual Transaction Check")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.number_input("Transaction Time (seconds since start)", min_value=0.0, value=1000.0)

if st.button("Check Fraud Risk"):
    # Create sample with dummy V1...V28 features
    sample = pd.DataFrame([[time, amount] + [0]*28],
                          columns=["Time", "Amount"] + [f"V{i}" for i in range(1,29)])
    sample = preprocess_data(sample)

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0, 1]

    st.write(f"**Prediction:** {'🚨 Fraud' if pred==1 else '✅ Normal'}")
    st.write(f"**Fraud Probability:** {prob:.2f}")
