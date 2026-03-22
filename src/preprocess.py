import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """
    Load data from CSV to allow Exploratory Data Analysis (EDA) before preprocessing.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_features(df, time_col='timestamp', target_col='fraud_flag'):
    """
    Handle missing values, generate time features, 
    encode categoricals, and return a clean DataFrame sorted by time.
    """
    df = df.copy()
    
    # Handle target definition
    if target_col == 'status' and 'status' in df.columns:
        # Create a binary target where 'false_positive' is 0, others (resolved, confirmed, etc.) are 1
        df['fraud_flag'] = np.where(df['status'] != 'false_positive', 1, 0)
        df = df.drop(columns=['status'])
        target_col = 'fraud_flag'
        
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number', 'datetime']).columns
    
    # Simple median/mode imputation
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Convert timestamp to datetime, sort, extract features
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort data by time to guarantee no data leakage during temporal split
        df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # Create time-based features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 18)).astype(int)
        
        df = df.drop(columns=[time_col])
    else:
        print(f"Warning: Time column '{time_col}' not found. Skipping time features.")
        
    # Log transform of amount to handle skewness
    if 'amount' in df.columns:
        df['log_amount'] = np.log1p(df['amount'])
        df = df.drop(columns=['amount'])
        
    # Encode categorical variables
    cat_cols_to_encode = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

def temporal_split_and_scale(df, target_col='fraud_flag'):
    """
    Perform a robust 60-20-20 temporal split and apply feature scaling.
    This guarantees no data leakage from the future since the data is sorted by time.
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
    
    # Feature scaling (StandardScaler)
    # Fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print(f"Split sizes -> Train: {len(X_train_scaled)}, Validation: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
