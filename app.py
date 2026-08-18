import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Classification Model Benchmark", layout="wide")

st.title("Breast Cancer Classification Benchmark")
st.write("Upload the test CSV data, choose a classification model, and inspect the evaluation metrics and confusion matrix.")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (test_data.csv)", type=["csv"])

model_options = {
    "Logistic Regression": ("model/logistic_regression.pkl", True),
    "Decision Tree": ("model/decision_tree.pkl", False),
    "kNN": ("model/knn.pkl", True),
    "Naive Bayes": ("model/naive_bayes.pkl", True),
    "Random Forest (Ensemble)": ("model/random_forest_ensemble.pkl", False)
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

@st.cache_resource
def load_scaler_and_model(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Dataset Preview")
    st.dataframe(test_df.head(5))

    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']

        model_path, requires_scaling = model_options[selected_model_name]
        model, scaler = load_scaler_and_model(model_path)

        # Scale features if required
        X_eval = scaler.transform(X_test) if requires_scaling else X_test

        # Predictions
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.markdown("---")
        st.subheader(f"Performance Metrics: {selected_model_name}")

        # Metrics display in cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("ROC AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        col_cm, col_report = st.columns([1, 1])

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=['Benign (0)', 'Malignant (1)'],
                        yticklabels=['Benign (0)', 'Malignant (1)'])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col_report:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4))

else:
    st.info("Please upload `test_data.csv` from the sidebar to evaluate the models.")