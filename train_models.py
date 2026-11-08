"""Train the five assignment classifiers and save models plus a held-out test CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

ROOT = Path(__file__).parent
data = pd.read_csv(ROOT / "breast_cancer_data.csv")
X, y = data.drop(columns=["target"]), data["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(ROOT / "test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_prob)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    filename = {"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib", "K-Nearest Neighbors": "k_nearest_neighbors.joblib", "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib", "Support Vector Machine": "support_vector_machine.joblib"}[name]
    joblib.dump(model, ROOT / "model" / filename)
with open(ROOT / "model" / "metrics.json", "w") as file:
    json.dump(results, file, indent=2)
print(json.dumps(results, indent=2))
