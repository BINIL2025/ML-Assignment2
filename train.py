import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load the raw UCI wdbc.data file (No headers)
# We define the 30 feature names manually based on wdbc.names documentation
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# The first two columns in wdbc.data are 'id' and 'diagnosis'
column_names = ["id", "diagnosis"] + feature_names

# Read the file
df = pd.read_csv("wdbc.data", header=None, names=column_names)

# Drop the 'id' column as it has no predictive power
df = df.drop(columns=["id"])

# Convert target column 'diagnosis' into a binary 'target' column (Malignant 'M' -> 1, Benign 'B' -> 0)
df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(columns=['diagnosis'])

X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the required test_data.csv for your Streamlit UI file upload
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Generated and saved test_data.csv successfully!")

# 3. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 4. Train & Save Models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    "kNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), True),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=100, random_state=42), False)
}

results = {}

for name, (model, requires_scaling) in models.items():
    X_tr = X_train_scaled if requires_scaling else X_train
    X_te = X_test_scaled if requires_scaling else X_test
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    results[name] = metrics
    filename = f"model/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(model, filename)

# Save summary
with open("model/metrics_summary.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n--- Evaluation Summary Table ---")
print(pd.DataFrame(results).T)