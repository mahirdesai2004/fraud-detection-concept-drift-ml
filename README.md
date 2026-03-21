# Robust Fraud Detection under Concept Drift

## 1. Problem Statement
Financial fraud is a continuously evolving challenge. Fraudsters constantly change their strategies to avoid detection, a phenomenon known as **Concept Drift**. This project builds a robust fraud detection system that remains effective even when transaction patterns change over time, specifically focusing on mobile money transactions.

## 2. Why this project is important
In the financial and fintech industries, static machine learning models degrade over time as real-world data distributions change. An industry-level model must detect concept drift and adapt to new fraud patterns. Demonstrating the ability to handle temporal data, class imbalance, and concept drift showcases applied, production-ready ML engineering skills.

## 3. Dataset Description
- **Source**: Hugging Face (`electricsheepafrica/nigerian-banking-mobile-money`)
- **Content**: Timestamped mobile money transactions from Nigeria.
- **Target**: `fraud_flag` (Binary classification: 1 for Fraud, 0 for Legitimate)
- **Significance**: The timestamped nature of this dataset is crucial for simulating real-world chronological transaction flows and evaluating model resilience over time.

## 4. Machine Learning Models Used
The project compares multiple algorithms to evaluate their robustness against concept drift:
1. **Logistic Regression** (Baseline)
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**
5. **Support Vector Machine (SVM)**
6. **SGDClassifier** (for online/incremental learning)

## 5. Key Features
- **Concept Drift Handling**: Monitoring performance across time and adapting to data changes.
- **Temporal Validation**: Using time-based train-test splits rather than random splits to prevent data leakage from the future.
- **Comparative Analysis**: Evaluating multiple models cleanly and clearly.
- **Explainability**: Simple and understandable model explanations (e.g., feature importance, SHAP).

## 6. Project Structure
```text
├── data/               # Raw and processed datasets
├── models/             # Saved model artifacts
├── notebooks/          # Step-by-step Jupyter notebooks for EDA and experimentation
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   ├── 05_online_learning.ipynb
│   └── 06_comparative_analysis.ipynb
├── outputs/            # Figures, charts, and evaluation metrics
├── src/                # Reusable, modular Python scripts
│   ├── preprocess.py   # Data cleaning and feature engineering functions
│   ├── train.py        # Model training routines
│   └── evaluate.py     # Evaluation metrics and plotting functions
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 7. How to Run
1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run notebooks**: Execute the Jupyter notebooks in `notebooks/` step-by-step. Reusable logic is imported from `src/`.

## 8. Results
*(Placeholder: Evaluation metrics, performance against drift, and final model selection will be documented here after execution.)*

## 9. Future Work
- Deploy the optimal model as a simple web service.
- Produce automated drift monitoring dashboards for production inference.
