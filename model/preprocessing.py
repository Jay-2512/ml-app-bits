import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Dataset Name: Breast Cancer Wisconsin (Diagnostic) Data Set
# URL         : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
DATA_PATH = "datasets/data.csv" 

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Drop non-feature columns
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Drop completely empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Convert the target labels
    # M -> Malignant : 1
    # B -> Benign    : 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Imputation
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save preprocessing objects
    joblib.dump(imputer, "model/imputer.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    return X_train, X_test, y_train, y_test
