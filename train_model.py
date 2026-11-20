import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

DATA_PATH = os.path.join("dataset", "customer_churn.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# TotalCharges is stored as text in the original Telco dataset.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Churn is the target: Yes = 1, No = 0.
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn", "customerID"])

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced")
}

results = []
fitted_pipelines = {}

for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })
    fitted_pipelines[name] = pipeline

results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
print("\nInitial model comparison:")
print(results_df.to_string(index=False))

# Tune both algorithms using recall as the optimization metric.
# Recall is useful in churn projects because it measures how many actual
# churners the model successfully identifies.
param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1.0, 5.0],
        "classifier__solver": ["liblinear"]
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_split": [2, 10, 20],
        "classifier__min_samples_leaf": [1, 5, 10]
    }
}

tuned_results = []
best_model_name = None
best_pipeline = None
best_recall = -1

for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring="recall",
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    candidate = search.best_estimator_
    pred = candidate.predict(X_test)
    prob = candidate.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, pred, zero_division=0)

    tuned_results.append({
        "Model": name,
        "Best CV Recall": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": precision_score(y_test, pred, zero_division=0),
        "Test Recall": recall,
        "Test F1-score": f1_score(y_test, pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, prob),
        "Best Parameters": search.best_params_
    })

    if recall > best_recall:
        best_recall = recall
        best_model_name = name
        best_pipeline = candidate

tuned_df = pd.DataFrame(tuned_results)
print("\nTuned model comparison:")
print(tuned_df.drop(columns=["Best Parameters"]).to_string(index=False))
print("\nBest parameters:")
for row in tuned_results:
    print(row["Model"], ":", row["Best Parameters"])

# Save the complete pipeline. It includes imputing, scaling, encoding and model.
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "churn_pipeline.pkl"))
with open(os.path.join(MODEL_DIR, "model_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"Selected model: {best_model_name}\n")
    f.write(f"Test recall: {best_recall:.4f}\n")

print(f"\nSaved models/churn_pipeline.pkl (selected model: {best_model_name})")

final_predictions = best_pipeline.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, final_predictions, target_names=["No Churn", "Churn"]))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, final_predictions))
