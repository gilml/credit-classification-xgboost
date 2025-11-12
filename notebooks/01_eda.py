#!/usr/bin/env python
# coding: utf-8

# ## Data Load and Setup

# In[1]:


import pandas as pd
from pathlib import Path

DATA_PATH = Path("../data/german_credit_cleaned.csv")
df = pd.read_csv(DATA_PATH)
df.shape


# ## Exploratory Data Analysis (EDA)

# In[2]:


# Check the balance between "good" and "bad" credit
df["target"].value_counts(normalize=True)


# In[3]:


df.describe().T


# In[4]:


# Check for missing values across all columns
df.isnull().sum().sort_values(ascending=False)


# In[5]:


# Identify numeric and categorical features
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

len(num_cols), len(cat_cols), num_cols[:5], cat_cols[:5]


# In[6]:


# Count unique values in each categorical column
df[cat_cols].nunique().sort_values()


# In[7]:


## Feature Engineering & Data Preparation


# In[8]:


# Separate numeric categorical vs continuous numeric features
u = df.nunique()
numeric_categorical = [c for c in num_cols if u[c] <= 10]
numeric_continuous = [c for c in num_cols if u[c] > 10]

numeric_categorical, numeric_continuous


# In[9]:


# Inspect values of a numeric categorical feature
df["installment_rate"].unique()


# In[10]:


# Convert numeric categorical features to category dtype (for clarity)
for c in numeric_categorical:
    df[c] = df[c].astype("category")

df[numeric_categorical].dtypes


# In[11]:


cat_cols


# In[12]:


# Remove target from categorical features and check unique category counts
cat_cols = [c for c in cat_cols if c != "target"]

df[cat_cols].nunique().sort_values()


# In[13]:


# Inspect all categorical feature values to verify categories and spot formatting issues
for c in cat_cols:
    print(c, "→", df[c].unique())


# In[14]:


# Encode 'checking_acc_status' as ordered categories (higher = better financial standing)
checking_map = {"no_cheking_acc": 0, "below_0": 1, "below_200": 2, "above:200": 3}

df["checking_acc_status"] = df["checking_acc_status"].map(checking_map)


# In[15]:


df["saving_acc_bonds"].unique()


# In[16]:


# Encode 'saving_acc_bonds' (higher = more savings)
saving_map = {
    "unknown_no_saving_acc": 0,
    "below_100": 1,
    "below_500": 2,
    "below_1000": 3,
    "above_1000": 4,
}
df["saving_acc_bonds"] = df["saving_acc_bonds"].map(saving_map)
df["saving_acc_bonds"].unique()


# In[17]:


df["present_employment_since"].unique()


# In[18]:


# Encode 'present_employment_since' (higher = longer employment stability)
employment_map = {
    "unemployed": 0,
    "below_1y": 1,
    "below_4y": 2,
    "below_7y": 3,
    "above_7y": 4,
}
df["present_employment_since"] = df["present_employment_since"].map(employment_map)
df["present_employment_since"].unique()


# In[19]:


df["personal_stat_gender"].unique()


# In[20]:


# Split 'personal_stat_gender' into 'gender' and 'personal_status'
df[["gender", "personal_status"]] = df["personal_stat_gender"].str.split(
    ":", expand=True
)

df[["gender", "personal_status"]].head()


# In[21]:


# Split 'personal_stat_gender' into two separate categorical features
df.drop(columns=["personal_stat_gender"], inplace=True)

df[["gender", "personal_status"]].nunique()


# In[22]:


# Update categorical feature list after splitting
cat_cols.remove("personal_stat_gender")
cat_cols.extend(["gender", "personal_status"])


# In[23]:


# One-hot encode remaining categorical features
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

df_encoded.shape


# In[24]:


# Make label numeric and verify all features are numeric
df_encoded["target"] = df["target"].map({"bad": 0, "good": 1})

df_encoded.select_dtypes(exclude=["number"]).columns.tolist()


# In[25]:


# Convert ordinal categorical columns to integer for XGBoost compatibility
ordinal_cols = [
    "installment_rate",
    "present_residence_since",
    "num_curr_loans",
    "num_people_provide_maint",
]
for c in ordinal_cols:
    if c in df_encoded.columns:
        df_encoded[c] = df_encoded[c].astype(int)

df_encoded.dtypes.value_counts()


# ## Modeling

# In[26]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Separate features and target
X = df_encoded.drop(columns=["target"])
y = df_encoded["target"]

# Stratified train-test split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize baseline XGBoost model
xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",  # binary cross-entropy; standard internal objective for classification
)


# In[27]:


# Train the model
xgb_clf.fit(X_train, y_train)


# ## Model Evaluation

# In[28]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test set
y_pred = xgb_clf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[29]:


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# Plot ROC curve
RocCurveDisplay.from_estimator(xgb_clf, X_test, y_test)
plt.title("ROC Curve - XGBoost Baseline")
plt.show()


# ## Cross-Validation Baseline

# In[30]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)


# In[31]:


print(f"Mean ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")


# ## Hyperparameter Tuning

# In[32]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Reusing CV
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

tuned.best_score_, tuned.best_params_


# In[33]:


# Hyperparameters of the best model

best_params = tuned.best_params_
best_params


# In[34]:


# Train final model with best hyperparameters
xgb_best = XGBClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist"
)

xgb_best.fit(X_train, y_train)


# ## Evaluate Tuned Model

# In[35]:


from sklearn.metrics import roc_auc_score

# Evaluate tuned model on test set
y_pred_best  = xgb_best.predict(X_test)
y_proba_best = xgb_best.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba_best):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))


# ## Feature Importance

# In[36]:


from xgboost import plot_importance

plot_importance(xgb_best, importance_type="gain", max_num_features=15)
plt.title("Top 15 Features by Importance (Gain)")
plt.show()


# ## Decision Threshold Tuning (focus: class 0)

# In[37]:


import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# y_proba_best = P(class=1 = "good"). We'll predict "good" if proba >= t, else "bad".
ths = np.linspace(0.20, 0.90, 29)  # scan thresholds
rows = []
for t in ths:
    y_pred_t = (y_proba_best >= t).astype(int)  # 1=good, 0=bad
    # per-class metrics, order is [class 0, class 1]
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_t, average=None, labels=[0,1])
    rows.append((t, prec[0], rec[0], f1[0]))

# Strategy A: maximize F1 for class 0 (balanced precision/recall)
best_f1 = max(rows, key=lambda x: x[3])

# Strategy B: require precision>=0.60 for class 0, then maximize recall
candidates = [r for r in rows if r[1] >= 0.60]
best_recall_with_prec = max(candidates, key=lambda x: x[2]) if candidates else None

print("Best F1 for class 0  ->  threshold=%.3f  precision=%.3f  recall=%.3f  f1=%.3f"
      % best_f1)
if best_recall_with_prec:
    t, p, r, f = best_recall_with_prec
    print("Max recall with precision>=0.60  ->  threshold=%.3f  precision=%.3f  recall=%.3f  f1=%.3f"
          % (t, p, r, f))
else:
    print("No threshold meets precision>=0.60 for class 0.")


# In[38]:


th_df = pd.DataFrame(rows, columns=["threshold", "prec_class0", "rec_class0", "f1_class0"])

plt.plot(th_df["threshold"], th_df["f1_class0"], marker="o")
plt.title("F1-score for Class 0 vs Decision Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("F1-score (Class 0)")
plt.grid(True)
plt.show()


# ## Export Tuned Model

# 

# In[39]:


import os
import joblib

# ensure models folder exists
os.makedirs("../models", exist_ok=True)

# save model and reload
joblib.dump(xgb_best, "../models/xgb_best_model.joblib")
loaded = joblib.load("../models/xgb_best_model.joblib")
print(type(loaded))


# In[40]:


# verify loaded model produces identical predictions
y_pred_loaded = loaded.predict(X_test)
assert (y_pred_loaded == y_pred_best).all(), "Loaded model predictions differ!"
print("✅ Loaded model verified - identical predictions.")


# ## Final Model Artifact
# 
# Exported and verified model stored under `models/xgb_best_model.joblib`.

# 

# In[1]:


from joblib import load
model = load("../models/xgb_best_model.joblib")
print("✅ Model ready for downstream use (inference / deployment).")

