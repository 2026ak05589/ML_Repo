import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification – Model Evaluation Lab")
st.write(
    "Compare five pre-trained classification models using an uploaded labelled test CSV. "
    "The application does not train or retrain models on uploaded data."
)
st.info(
    "Dataset target encoding: 0 = malignant, 1 = benign. "
    "Upload a CSV containing all 30 feature columns and a target column."
)

uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
selected_model = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        required = FEATURES + ["target"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif data.empty:
            st.error("The uploaded CSV contains no rows.")
        elif not set(pd.to_numeric(data["target"], errors="coerce").dropna().unique()).issubset({0, 1}):
            st.error("Target values must be encoded as 0 (malignant) or 1 (benign).")
        else:
            X_test = data[FEATURES].apply(pd.to_numeric, errors="coerce")
            y_test = pd.to_numeric(data["target"], errors="coerce")
            if X_test.isna().any().any() or y_test.isna().any():
                st.error("Features and target must contain numeric, non-missing values.")
            else:
                model_path = MODEL_DIR / MODEL_FILES[selected_model]
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, y_score) if y_test.nunique() == 2 else float("nan")
                metric_values = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC": auc,
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                    "MCC": matthews_corrcoef(y_test, y_pred),
                }
                st.subheader(f"Evaluation results – {selected_model}")
                cols = st.columns(6)
                for col, (label, value) in zip(cols, metric_values.items()):
                    col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

                left, right = st.columns(2)
                with left:
                    st.markdown("#### Confusion matrix")
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    st.dataframe(
                        pd.DataFrame(cm, index=["Actual malignant (0)", "Actual benign (1)"],
                                     columns=["Predicted malignant (0)", "Predicted benign (1)"]),
                        use_container_width=True
                    )
                with right:
                    st.markdown("#### Classification report")
                    st.code(classification_report(
                        y_test, y_pred, labels=[0, 1],
                        target_names=["malignant", "benign"], zero_division=0
                    ))
                st.caption(f"Rows evaluated: {len(data)} | No model training was performed in this app.")
    except Exception as exc:
        st.error(f"Could not process the uploaded file: {exc}")
else:
    st.warning("Upload the included test_data.csv file to view the evaluation results.")
    st.write("The included CSV has the 30 required input features plus the target column.")
