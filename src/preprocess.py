import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath, time_col='timestamp', target_col='fraud_flag'):
    """
    Load data from CSV, handle missing values, generate time features, 
    encode categoricals, and return a clean DataFrame sorted by time.
    """
    # 1. Load dataset from CSV
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 2. Handle missing values
    # Simple approach: forward fill then backward fill for time series nature, 
    # or just fill numeric with median and categorical with mode.
    # For simplicity and robustness, we drop rows where target is missing,
    # and fill others with median/mode.
    # Handle target definition
    if target_col == 'status' and 'status' in df.columns:
        # Create a binary target where 'false_positive' is 0, others (resolved, confirmed, etc.) are 1
        df['fraud_flag'] = (df['status'] != 'false_positive').astype(int)
        df = df.drop(columns=['status'])
        target_col = 'fraud_flag'
        
    df = df.dropna(subset=[target_col])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Convert timestamp to datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 4. Sort data by time
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # 5. Create time-based features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        
        # We can drop the original time column now since we extracted features
        df = df.drop(columns=[time_col])
    else:
        print(f"Warning: Time column '{time_col}' not found. Skipping time features.")
        
    # 6. Encode categorical variables using LabelEncoder
    cat_cols_to_encode = df.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    for col in cat_cols_to_encode:
        le = LabelEncoder()
        # Convert to string to avoid mixed type issues
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df

def temporal_split_and_scale(df, target_col='fraud_flag'):
    """
    Perform a robust 60-20-20 temporal split and apply feature scaling.
    This guarantees no data leakage from the future.
    """
    # Calculate split indices
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    # Split the data sequentially
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Separate features (X) and target (y)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # 7. Feature scaling (StandardScaler)
    # Fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print(f"Split sizes -> Train: {len(X_train_scaled)}, Validation: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
