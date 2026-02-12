import streamlit as st
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("🩺 Breast Cancer Classification App")

# ---------------- LOAD METADATA ----------------
with open("model/metrics.json") as f:
    METRICS = json.load(f)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# Load preprocessing objects
imputer = joblib.load("model/imputer.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------- DATASET UPLOAD ----------------
st.header("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "diagnosis" not in df.columns:
        st.error("CSV must contain a 'diagnosis' column")
        st.stop()

    # Clean dataset
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df.dropna(axis=1, how="all", inplace=True)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"M": 1, "B": 0})

    # Preprocess
    X = imputer.transform(X)
    X = scaler.transform(X)

    # Model selection
    st.header("🤖 Select Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    model = joblib.load(MODEL_FILES[model_name])

    # Prediction
    y_pred = model.predict(X)

    # Metrics Display
    st.header("📊 Evaluation Metrics")

    m = METRICS[model_name.lower().replace(" ", "_")]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{m['accuracy']:.3f}")
    col2.metric("Precision", f"{m['precision']:.3f}")
    col3.metric("Recall", f"{m['recall']:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{m['f1_score']:.3f}")
    col5.metric("AUC", f"{m['auc']:.3f}")
    col6.metric("MCC", f"{m['mcc']:.3f}")

    # Confusion matrix
    st.header("📉 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.header("📄 Classification Report")
    report = classification_report(y, y_pred, target_names=["Benign", "Malignant"])
    st.text(report)
