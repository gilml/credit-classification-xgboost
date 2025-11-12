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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# ---------------------------
# Data Load
# ---------------------------
DATA_PATH = Path("../data/german_credit_cleaned.csv")
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Feature Engineering
# ---------------------------
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "target"]

# Ordinal encodings
checking_map = {"no_cheking_acc": 0, "below_0": 1, "below_200": 2, "above:200": 3}
saving_map = {
    "unknown_no_saving_acc": 0,
    "below_100": 1,
    "below_500": 2,
    "below_1000": 3,
    "above_1000": 4,
}
employment_map = {
    "unemployed": 0,
    "below_1y": 1,
    "below_4y": 2,
    "below_7y": 3,
    "above_7y": 4,
}

df["checking_acc_status"] = df["checking_acc_status"].map(checking_map)
df["saving_acc_bonds"] = df["saving_acc_bonds"].map(saving_map)
df["present_employment_since"] = df["present_employment_since"].map(employment_map)

# Split personal_stat_gender
df[["gender", "personal_status"]] = df["personal_stat_gender"].str.split(
    ":", expand=True
)
df.drop(columns=["personal_stat_gender"], inplace=True)
cat_cols.remove("personal_stat_gender")
cat_cols.extend(["gender", "personal_status"])

# One-hot encode
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_encoded["target"] = df["target"].map({"bad": 0, "good": 1})

# Convert ordinal numeric features to int
ordinal_cols = [
    "installment_rate",
    "present_residence_since",
    "num_curr_loans",
    "num_people_provide_maint",
]
for c in ordinal_cols:
    if c in df_encoded.columns:
        df_encoded[c] = df_encoded[c].astype(int)

# ---------------------------
# Train-Test Split
# ---------------------------
X = df_encoded.drop(columns=["target"])
y = df_encoded["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Baseline Model + CV
# ---------------------------
xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"Baseline CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ---------------------------
# Hyperparameter Tuning
# ---------------------------
param_dist = {
    "n_estimators": randint(300, 900),
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.02, 0.18),
    "min_child_weight": randint(1, 6),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0.0, 2.0),
}

tuned = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=42, n_jobs=-1, eval_metric="logloss", tree_method="hist"
    ),
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
tuned.fit(X_train, y_train)
best_params = tuned.best_params_
print("Best ROC-AUC:", tuned.best_score_)
print("Best Params:", best_params)

# ---------------------------
# Final Model Training
# ---------------------------
xgb_best = XGBClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",
)
xgb_best.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred_best = xgb_best.predict(X_test)
y_proba_best = xgb_best.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba_best):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# ---------------------------
# Export Model
# ---------------------------
os.makedirs("../models", exist_ok=True)
joblib.dump(xgb_best, "../models/xgb_best_model.joblib")

# Verify reload
loaded = joblib.load("../models/xgb_best_model.joblib")
assert (
    loaded.predict(X_test) == y_pred_best
).all(), "Reloaded model predictions differ!"
print("✅ Model saved and verified successfully.")
