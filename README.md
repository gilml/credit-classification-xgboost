# Credit Classification with XGBoost

This project builds and evaluates a credit risk classification model using **XGBoost** and **Scikit-Learn**.  
The goal is to predict whether a loan applicant is likely to be a *good* or *bad* credit risk, based on structured features from the classic German Credit dataset.

---

## 📋 Project Overview

**Pipeline summary:**
1. **EDA:** dataset inspection, class balance, and feature typing.  
2. **Feature Engineering:** numeric / categorical separation, encoding, and data preparation.  
3. **Modeling:** baseline XGBoost training with log-loss objective.  
4. **Evaluation:** accuracy, ROC-AUC, confusion matrix, and class-wise metrics.  
5. **Hyperparameter Tuning:** randomized search with stratified 5-fold CV.  
6. **Threshold Optimization:** F1/precision-recall trade-off for class 0 (bad credit).  
7. **Model Export:** final tuned model serialized with `joblib`.

---

## 📁 Repository Structure

```
credit-classification-xgboost/
├── data/                     # Input data (not versioned)
│   └── german_credit_cleaned.csv
├── notebooks/
│   └── 01_eda.ipynb          # Main Jupyter workflow
├── models/
│   └── xgb_best_model.joblib # Trained XGBoost model
├── src/
│   └── credit_risk/          # Future Python package (helpers, pipelines)
├── scripts/                  # Automation or CLI utilities
├── reports/                  # Figures, plots, analysis outputs
│   └── figures/
├── docs/                     # Optional documentation
├── requirements.txt          # Environment dependencies
└── README.md
```

---

## ⚙️ Environment

Python 3.13 (virtual environment)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Run the main notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Outputs:**
- Baseline + tuned XGBoost metrics  
- ROC curve and threshold analysis  
- Serialized model in `models/xgb_best_model.joblib`

---

## 📝 Notes

- `StratifiedKFold` ensures balanced folds for the binary target.  
- Randomized search (40 iterations) achieved ROC-AUC ≈ 0.79.  
- Decision threshold tuning improved interpretability for loan approval policy.  
- The notebook is modular — each markdown section corresponds to a distinct training phase.

---
