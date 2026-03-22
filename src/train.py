import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_and_predict(model_type, X_train, y_train, X_val, y_val, X_test):
    """
    Train model, handle imbalance, and return predictions + probabilities.
    """

    model = None

    # --- MODEL SELECTION ---
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'tree':
        model = DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=5,
            random_state=42
        )

    elif model_type == 'forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'xgboost':
        # Proper imbalance handling
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

        model = XGBClassifier(
            objective='binary:logistic',   
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )

    elif model_type == 'sgd':
        model = SGDClassifier(
            loss='log_loss',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --- TRAINING ---
    print(f"Training {model_type}...")
    start_time = time.time()

    if model_type == 'xgboost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    print(f"{model_type} trained in {time.time() - start_time:.2f}s")

    # --- PREDICTIONS ---
    y_pred = model.predict(X_test)

    # --- PROBABILITIES (IMPORTANT FOR PR-AUC) ---
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback (rare)
        y_prob = None

    return model, y_pred, y_prob