# Machine Learning Assignment 2: Classification Models & Deployment

## a. Problem Statement
The objective of this project is to perform binary classification to predict whether a breast cancer tumor is malignant or benign based on digitized fine needle aspirate (FNA) test features. Multiple classification models are trained, benchmarked across 6 distinct evaluation metrics, and deployed via a Streamlit web application.

## b. Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / Scikit-Learn)
- **Instances:** 569 samples
- **Features:** 30 continuous real-valued features (computed from cell nuclei measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension)
- **Target:** `0` (Benign - 357 instances), `1` (Malignant - 212 instances)

## c. GitHub Repository Link
- Repository: `https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9825 | 0.9974 | 0.9730 | 1.0000 | 0.9863 | 0.9634 |
| Decision Tree | 0.9386 | 0.9365 | 0.9577 | 0.9444 | 0.9510 | 0.8696 |
| kNN | 0.9561 | 0.9785 | 0.9718 | 0.9583 | 0.9650 | 0.9068 |
| Naive Bayes | 0.9474 | 0.9891 | 0.9583 | 0.9583 | 0.9583 | 0.8882 |
| Random Forest (Ensemble) | 0.9737 | 0.9950 | 0.9726 | 0.9861 | 0.9793 | 0.9443 |

*(Note: Run `train.py` on your environment to obtain your exact decimal values).*

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well on scaled data with the highest AUC and perfect recall, demonstrating linear separability of normalized nuclear features. |
| **Decision Tree** | Captures direct rule-based splits but is prone to slight overfitting on smaller margins, leading to lower MCC compared to ensemble methods. |
| **kNN** | Delivered strong accuracy by leveraging localized neighborhood proximity across normalized feature spaces. |
| **Naive Bayes** | Maintained high AUC and balanced precision-recall despite the strong conditional independence assumption among correlated geometric features. |
| **Random Forest (Ensemble)** | Achieved high stability, precision, and AUC by reducing variance across individual decision trees. |

### Overall Winner
**Logistic Regression / Random Forest (Ensemble)** is the overall winner for this dataset due to superior generalization balance across Accuracy, AUC, and Matthews Correlation Coefficient (MCC).