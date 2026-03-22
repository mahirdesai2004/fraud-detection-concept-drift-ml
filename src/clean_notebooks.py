import os
import json

notebooks_config = {
    "01_data_loading.ipynb": {
        "title": "# 1. Data Loading and Exploration",
        "desc": "In this notebook, we load the Nigerian Mobile Money transaction dataset and explore its raw properties.\n\nThe dataset is loaded from `data/raw_data.csv` which was downloaded using the HuggingFace API.",
        "code": "import pandas as pd\n\ndf = pd.read_csv('../data/raw_data.csv')\ndisplay(df.head())\nprint(df.info())"
    },
    "02_preprocessing.ipynb": {
        "title": "# 2. Data Preprocessing",
        "desc": "Here we apply our preprocessing pipeline carefully designed to handle Concept Drift.\n- Extract time features (`hour`, `day_of_week`)\n- Add `is_night` feature since fraud often happens at night\n- Log-transform financial amounts\n- Handle missing values\n- Apply Temporal Train/Validation/Test Split to prevent data leakage.",
        "code": "import sys\nsys.path.append('../src')\nfrom preprocess import load_and_preprocess_data, temporal_split_and_scale\n\ndf = load_and_preprocess_data('../data/raw_data.csv')\nX_train, X_val, X_test, y_train, y_val, y_test = temporal_split_and_scale(df)\nprint(f'Train size: {len(X_train)}, Test size: {len(X_test)}')"
    },
    "03_baseline_models.ipynb": {
        "title": "# 3. Baseline Models",
        "desc": "We establish our baselines using simple interpretable models: Logistic Regression and Decision Trees.\nClass imbalance is handled using `class_weight='balanced'`.",
        "code": "from train import train_and_predict\nfrom evaluate import evaluate_model\n\n# Logistic Regression\nmodel_lr, y_pred_lr, y_prob_lr = train_and_predict('logistic', X_train, y_train, X_val, y_val, X_test)\nmetrics_lr = evaluate_model(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression')\nprint(metrics_lr)"
    },
    "04_advanced_models.ipynb": {
        "title": "# 4. Advanced Models",
        "desc": "To better capture complex non-linear patterns, we train advanced ensemble models (Random Forest, XGBoost) and Support Vector Machines (SVM). XGBoost employs early stopping to prevent overfitting.",
        "code": "from train import train_and_predict\n\n# XGBoost\nmodel_xgb, y_pred_xgb, y_prob_xgb = train_and_predict('xgboost', X_train, y_train, X_val, y_val, X_test)\n# Random Forest\nmodel_rf, y_pred_rf, y_prob_rf = train_and_predict('forest', X_train, y_train, X_val, y_val, X_test)"
    },
    "05_online_learning.ipynb": {
        "title": "# 5. Online Learning to handle Concept Drift",
        "desc": "Concept Drift happens when underlying data distributions evolve. A static model degrades over time. Here we use an `SGDClassifier` which is updated incrementally with new batches of data representing the flow of time.",
        "code": "from train import train_and_predict\n\n# Online SGD\nmodel_sgd, y_pred_sgd, y_prob_sgd = train_and_predict('sgd', X_train, y_train, X_val, y_val, X_test)"
    },
    "06_comparative_analysis.ipynb": {
        "title": "# 6. Comparative Analysis and Visualization",
        "desc": "Finally, we evaluate all models comprehensively. Since the dataset is highly imbalanced, we focus on Precision-Recall AUC (PR-AUC) and F1-Score rather than pure Accuracy.",
        "code": "import pandas as pd\nimport IPython.display as display\nfrom IPython.display import Image\n\n# Display calculated results\nresults = pd.read_csv('../outputs/results.csv')\ndisplay.display(results)\n\n# Show visual comparisons\ndisplay.display(Image(filename='../outputs/model_comparison.png'))\ndisplay.display(Image(filename='../outputs/pr_curves.png'))"
    }
}

template = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

for nb_name, content in notebooks_config.items():
    nb = template.copy()
    nb["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [content["title"] + "\n\n" + content["desc"]]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [content["code"].replace('\n', '\n')]}
    ]
    path = os.path.join("notebooks", nb_name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Cleaned {path}")
