#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------
# Load data
# ---------------------------
DATA_PATH = Path("data/german_credit_cleaned.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"].map({"bad": 0, "good": 1})

# ---------------------------
# Preprocessing pipeline
# ---------------------------
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

# ---------------------------
# Model pipeline
# ---------------------------
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

pipe = Pipeline([("preprocess", preprocessor), ("model", xgb)])

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# Baseline CV
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"Baseline CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ---------------------------
# Hyperparameter tuning
# ---------------------------
param_dist = {
    "model__n_estimators": randint(300, 900),
    "model__max_depth": randint(3, 8),
    "model__learning_rate": uniform(0.02, 0.18),
    "model__min_child_weight": randint(1, 6),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__gamma": uniform(0.0, 2.0),
}

tuned = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

tuned.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = tuned.predict(X_test)
y_proba = tuned.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# Save final pipeline
# ---------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(tuned.best_estimator_, "models/xgb_pipeline.joblib")
print("✅ Pipeline model saved successfully at models/xgb_pipeline.joblib")
