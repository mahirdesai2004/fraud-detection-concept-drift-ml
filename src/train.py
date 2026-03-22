import time
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pandas as pd

def train_and_predict(model_type, X_train, y_train, X_val, y_val, X_test):
    """
    Train a specified model, evaluate on both train and validation sets, 
    and return test predictions/probabilities. 
    
    Args:
        model_type: 'logistic', 'tree', 'forest', 'xgboost', 'svm', 'sgd'
    """
    model = None
    
    # Keep class_weight='balanced' where available to handle fraud class imbalance
    # Use random_state=42 for reproducibility
    # Limit depth/estimators to avoid overfitting
    
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
        
    elif model_type == 'tree':
        # max_depth=5 to avoid overfitting the tree heavily
        model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
        
    elif model_type == 'forest':
        # Moderate n_estimators and depth to keep it simple and stable
        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        
    elif model_type == 'xgboost':
        # scale_pos_weight is the XGBoost equivalent of class_weight='balanced'
        # Usually positive class weight = count(negative) / count(positive)
        # Assuming typical imbalance, we let it be 1 for now but set simple stable parameters
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
        model = XGBClassifier(
            n_estimators=200, 
            max_depth=5, 
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=10,
            random_state=42,
            n_jobs=-1
        )
        
    elif model_type == 'svm':
        # SVM is computationally expensive on large datasets.
        # We limit the training data size to avoid infinite training time.
        sample_size = min(len(X_train), 10000) 
        X_train_sub = X_train.iloc[:sample_size]
        y_train_sub = y_train.iloc[:sample_size]
        print(f"  Note: SVM using subset of {sample_size} samples for training to avoid underfitting/hanging.")
        
        # Using probability=True to allow PR-AUC calculation later
        model = SVC(class_weight='balanced', probability=True, random_state=42)
        
        print(f"Training {model_type}...")
        start_time = time.time()
        model.fit(X_train_sub, y_train_sub)
        print(f"{model_type} trained in {time.time() - start_time:.2f}s")
        
    elif model_type == 'sgd':
        # SGDClassifier for incremental/online learning
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight
        classes_arr = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes_arr, y=y_train)
        cw_dict = {classes_arr[i]: cw[i] for i in range(len(classes_arr))}
        
        model = SGDClassifier(loss='log_loss', class_weight=cw_dict, random_state=42, n_jobs=-1)
        
        # Using partial fit simulating online learning across batches
        print(f"Training {model_type} using partial_fit...")
        start_time = time.time()
        classes = sorted(y_train.unique())
        
        # Simulate feeding in mini-batches
        batch_size = 1000
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train.iloc[i:i+batch_size]
            y_batch = y_train.iloc[i:i+batch_size]
            model.partial_fit(X_batch, y_batch, classes=classes)
        print(f"{model_type} trained in {time.time() - start_time:.2f}s")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # For models that we can fit using normal .fit (everything except early stopping logic for xgb or SVM subsetting above)
    if model_type not in ['svm', 'sgd']:
        print(f"Training {model_type}...")
        start_time = time.time()
        
        if model_type == 'xgboost':
            # Early stopping using validation set to prevent overfitting
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
            
        print(f"{model_type} trained in {time.time() - start_time:.2f}s")

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Get probabilities for positive class (useful for PR curve and AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # For some models like basic SVM (if prob=False but we set it True), we'd use decision function, 
        # but since we ensured probability=True or log_loss for SGD, we should have predict_proba.
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None
        
    return model, y_pred, y_prob
