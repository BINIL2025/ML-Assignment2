# Machine Learning Assignment 2: Classification Benchmark & Deployment

## a. Problem Statement
The objective of this assignment is to perform binary classification to predict whether a breast cancer tumor is malignant or benign based on diagnostic features extracted from cell nuclei images. Multiple machine learning models are trained, evaluated across six standard classification metrics, and deployed as an interactive web application on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository
- **Number of Instances:** 569 samples
- **Number of Features:** 30 continuous numerical features (computed from digitized images of fine needle aspirates (FNA) of breast masses)
- **Target Variable:** Binary `target` (0 = Benign [357 instances], 1 = Malignant [212 instances])

## c. GitHub Repository Link
- Repository Link: <YOUR_GITHUB_REPO_LINK_HERE>

## d. Models Used & Evaluation Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest AUC (0.9960) and strong overall balance across accuracy and F1 score, demonstrating that scaled nuclear measurements are largely linearly separable. |
| **Decision Tree** | Yielded lower recall (0.8333) and MCC (0.8299) compared to ensemble approaches, reflecting the sensitivity of single-tree models to boundary margins on continuous numerical data. |
| **kNN** | Delivered strong performance (Accuracy 0.9561, MCC 0.9058) by exploiting localized neighborhood proximity in the standardized multi-dimensional space. |
| **Naive Bayes** | Maintained an outstanding AUC of 0.9891 despite its conditional feature independence assumption across correlated geometric features (perimeter, radius, area). |
| **Random Forest (Ensemble)** | Achieved the top Accuracy (0.9737), perfect Precision (1.0000), and top MCC (0.9442), proving its robustness in variance reduction across decision trees. |

### Overall Winner
**Random Forest (Ensemble)** is the overall winner for this dataset, achieving the highest overall Accuracy (97.37%), perfect Precision (100%), and the highest Matthews Correlation Coefficient (0.9442).