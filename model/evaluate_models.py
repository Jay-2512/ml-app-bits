import json
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model_names = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost"
]

metrics = {}

for name in model_names:
    model = joblib.load(f"model/{name}.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics stored successfully in model/metrics.json")
