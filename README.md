# Machine Learning Assignment 2 – Breast Cancer Classification

## a. Problem statement

The objective is to implement and compare five supervised classification algorithms on the same labelled dataset. The models predict whether a breast tumour record is **malignant (0)** or **benign (1)**. Performance is evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). A Streamlit application allows an evaluator to upload labelled test data, select a saved model and inspect its metrics and confusion matrix. The app loads saved models and does not retrain them on uploaded test data.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset.
- **Original public source:** UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Records:** 569.
- **Input features:** 30 numeric features derived from digitized images of fine needle aspirates of breast masses.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Task:** Binary classification.
- **Files:** `breast_cancer_data.csv` contains all 569 records; `test_data.csv` contains the stratified 20% held-out test partition.
- **Split:** 80% training / 20% testing, stratified, `random_state=42`.

The dataset is available through scikit-learn's bundled copy of the UCI Wisconsin Diagnostic dataset. The original UCI page is cited above.

## c. GitHub Repository Link

**GitHub Repository:** [PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]

Repository files include `app.py`, `train_models.py`, `requirements.txt`, this `README.md`, `breast_cancer_data.csv`, `test_data.csv`, and five saved model files under `model/`.

## d. Models used

All five classifiers were trained on the same training partition and evaluated on the same held-out test partition. StandardScaler is included in the Logistic Regression, KNN and SVM pipelines. Decision Tree and Gaussian Naive Bayes are used without scaling. AUC is calculated from predicted probability for the positive class (benign = 1).

### Model comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy and 0.9954 AUC. Standardization supports this distance/optimization-based model. It made only two errors on the held-out test set. |
| Decision Tree | Achieved 91.23% accuracy and 0.9157 AUC. It is interpretable, but its performance was lower than the other four models on this split, which can be consistent with overfitting or sensitivity to the training sample. |
| K-Nearest Neighbors | Achieved 95.61% accuracy and 0.9788 AUC. Feature standardization is important for distance calculations. It performed strongly but made six test errors. |
| Gaussian Naive Bayes | Achieved 93.86% accuracy and 0.9878 AUC. It produced a strong AUC, although its classification accuracy was below Logistic Regression and SVM on this split. |
| Support Vector Machine | Achieved 98.25% accuracy and 0.9950 AUC. The RBF-kernel SVM performed very strongly after feature standardization, matching Logistic Regression on accuracy, precision, recall, F1 and MCC. |
| **Overall winner** | **Logistic Regression** is the selected winner on this split: it achieved the highest AUC (0.9954) and tied with SVM for the best accuracy (0.9825), precision, recall, F1 and MCC. SVM is a very close second. |

### Experimental notes

- Test set size: 114 records.
- Training set size: 455 records.
- Positive class for Precision, Recall and F1: benign (target = 1).
- Confusion matrix order: rows = actual class [malignant, benign], columns = predicted class [malignant, benign].
- Results depend on the selected split and random seed; rerun `python train_models.py` to reproduce the experiment in the supplied environment.

## Streamlit deployment

**Live Streamlit App:** [PASTE YOUR DEPLOYED STREAMLIT APP URL HERE]

1. Push the complete project folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select this repository, branch `main`, and file `app.py`.
4. Deploy. Ensure the repository includes the `model/` folder and all required files.
5. Open the deployed app, upload `test_data.csv`, and select each model to inspect its results.

## Application features

- CSV uploader for labelled test data.
- Model selection dropdown for all five saved classifiers.
- Accuracy, AUC, Precision, Recall, F1 and MCC metrics.
- Confusion matrix and classification report.
- Input validation for missing columns, non-numeric values and invalid target labels.
- No retraining on uploaded test data.

## BITS Virtual Lab execution evidence

**Insert the required screenshot here.** Capture the assignment execution inside BITS Virtual Lab and include the screenshot in the final submission PDF. This evidence must be captured by the student; it cannot be generated by this project package.

## Reproduction

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
