<div align="center">

# Fraud Detection under Concept Drift using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-006600)
![SHAP](https://img.shields.io/badge/SHAP-0.44-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

**A production-grade machine learning pipeline for detecting fraudulent transactions in highly imbalanced, temporally evolving data.**

</div>

---

## Overview

Financial fraud detection is a high-stakes classification problem where traditional accuracy metrics fail due to extreme class imbalance. This project builds a robust, end-to-end ML pipeline that addresses three critical real-world challenges simultaneously: severe class imbalance (~1.5% fraud rate), concept drift over time, and the need for transparent, explainable predictions.

The pipeline achieves a **~12x improvement** in Precision-Recall AUC over the random baseline, moving from ~0.015 to ~0.19, while providing actionable business insights through cost-sensitive threshold optimization and SHAP-based model explainability.

---

## Problem Statement

In real-world payment systems, fraudulent transactions represent a tiny fraction of total volume. A naive model predicting "not fraud" for every transaction achieves 98.5% accuracy but catches zero fraud. This makes accuracy a misleading metric.

**Core Challenges:**
- Extreme class imbalance (legitimate transactions outnumber fraud ~66:1)
- Fraud patterns evolve over time (concept drift), degrading static models
- Missed fraud carries significantly higher cost than false alarms
- Regulatory and business requirements demand model transparency

**Objective:** Build a fraud classifier evaluated on **Precision-Recall AUC (PR-AUC)** that maximizes fraud recall while keeping false positives operationally manageable, and that remains robust as transaction patterns shift over time.

---

## Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | Synthetic financial transaction dataset (mobile money) |
| **Total Transactions** | ~6.3 million |
| **Fraud Rate** | ~1.5% |
| **Features** | Transaction type, amount, origin/destination balances, timestamps |
| **Target Variable** | `isFraud` (binary) |

> **Why PR-AUC?** In datasets where the positive class is rare (~1.5%), ROC-AUC can be misleadingly optimistic because the large number of true negatives inflates the curve. PR-AUC focuses exclusively on precision and recall for the minority class, providing a far more honest measure of model utility.

---

## Approach

1. **Exploratory Data Analysis** -- Uncovered fraud concentration in specific transaction types, temporal windows, and amount ranges
2. **Feature Engineering** -- Constructed temporal (hour, day), interaction (balance deltas), and behavioral features to capture fraud-indicative signals
3. **Temporal Train-Validation-Test Split** -- Strict time-based partitioning (60/20/20) to prevent data leakage and simulate real deployment conditions
4. **Multi-Model Benchmarking** -- Evaluated Logistic Regression, Decision Tree, Random Forest, XGBoost, and SGD classifiers
5. **Hyperparameter Tuning** -- GridSearchCV with temporal cross-validation on XGBoost to maximize PR-AUC without overfitting
6. **Threshold Optimization** -- Swept classification thresholds to find the operating point that minimizes total business cost
7. **Explainability** -- SHAP analysis to identify the features driving fraud predictions

---

## Model Pipeline

```
Raw Data
  |
  v
Feature Engineering (temporal, interaction, behavioral)
  |
  v
Temporal Split (train / validation / test)
  |
  v
Baseline Models (Logistic Regression, Decision Tree)
  |
  v
Advanced Models (Random Forest, XGBoost, SGD)
  |
  v
Hyperparameter Tuning (GridSearchCV on XGBoost)
  |
  v
Threshold Optimization (cost-sensitive)
  |
  v
Evaluation (PR-AUC, F1, Confusion Matrix)
  |
  v
Explainability (SHAP) + Drift Monitoring
```

---

## Results

### Class Distribution

![Fraud Distribution](outputs/plots/fraud_distribution.png)

The dataset exhibits a ~66:1 imbalance ratio, confirming that accuracy is inadequate as an evaluation metric.

### Model Performance Comparison

![Model Comparison](outputs/plots/model_comparison.png)

| Model | PR-AUC | Notes |
|-------|--------|-------|
| Random Baseline | ~0.015 | Equivalent to fraud rate |
| Logistic Regression | Low | Underfits non-linear patterns |
| Decision Tree | Moderate | Prone to overfitting |
| Random Forest | Strong | Robust ensemble performance |
| **XGBoost (Tuned)** | **~0.19** | **Best overall -- 12x improvement** |

### Precision-Recall Curves

![PR Curves](outputs/plots/pr_curves.png)

The Precision-Recall curves demonstrate XGBoost's clear dominance across all recall levels, maintaining higher precision at equivalent recall thresholds compared to all other models.

---

## Threshold Optimization

The default 0.5 classification threshold is suboptimal for imbalanced fraud detection. By sweeping thresholds from 0.05 to 0.50 and evaluating the F1-score at each point, the pipeline identifies an optimal threshold that:

- **Increases recall** substantially (catching more fraud)
- **Maintains acceptable precision** (keeping false positives manageable)
- **Minimizes total business cost** when weighted by the asymmetric cost of missed fraud vs. false alarms

---

## Business Impact

Fraud detection is not just a classification problem -- it is a cost optimization problem.

- **Cost of a missed fraud (False Negative):** Direct financial loss + customer trust erosion + regulatory penalties
- **Cost of a false alarm (False Positive):** Investigation overhead + customer friction

By explicitly modeling these asymmetric costs and tuning the decision threshold accordingly, this pipeline shifts from a purely statistical model to a **financially optimized decision system**.

---

## Concept Drift

Static models degrade as fraud tactics evolve. This project includes a concept drift analysis that tracks PR-AUC and recall across weekly time cohorts to quantify performance degradation.

**Key Finding:** Model performance fluctuates across temporal windows, confirming the presence of concept drift and underscoring the need for periodic retraining in production deployments.

---

## Explainability

Model transparency is non-negotiable in financial applications. SHAP (SHapley Additive exPlanations) was used to decompose predictions into per-feature contributions.

**Key SHAP Insights:**
- Transaction amount and balance delta features are the strongest fraud indicators
- Temporal features (hour of day) contribute meaningfully, reflecting fraud concentration during off-peak hours
- Transaction type acts as a categorical gate, with certain types exhibiting disproportionate fraud rates

These explanations enable compliance teams to understand and audit model decisions at the individual transaction level.

---

## Key Insights

- PR-AUC is the correct metric for highly imbalanced fraud detection; accuracy and ROC-AUC are misleading
- Temporal validation is essential to prevent optimistic performance estimates caused by data leakage
- Threshold tuning alone can dramatically shift the precision-recall trade-off without retraining
- Concept drift is real and measurable -- models must be monitored and retrained periodically
- SHAP provides the transparency required for regulatory compliance and business trust

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML Frameworks | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/mahirdesai2004/fraud-detection-concept-drift-ml.git
cd fraud-detection-concept-drift-ml

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline notebook
jupyter notebook notebooks/FINAL_DEMO.ipynb
```

Alternatively, execute the modular scripts in `src/` for preprocessing, training, and evaluation independently.

---

## Future Work

- Deploy as a real-time inference API with FastAPI and containerized via Docker
- Implement automated drift detection triggers for model retraining (Evidently AI)
- Explore graph-based features to capture ring-fraud and collusion patterns
- Scale to distributed training on cloud platforms (AWS SageMaker, GCP Vertex AI)
- Add A/B testing framework for safe model rollout in production

---

## Resume Bullet

> Built a production-grade Fraud Detection pipeline on a highly imbalanced dataset (1.5% fraud rate), achieving a 12x improvement in PR-AUC using XGBoost with GridSearchCV, cost-sensitive threshold tuning, SHAP explainability, and temporal drift monitoring.
