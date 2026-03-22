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
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # --- Handle target ---
    if target_col == 'status' and 'status' in df.columns:
        df['fraud_flag'] = np.where(df['status'] != 'false_positive', 1, 0)
        df = df.drop(columns=['status'])
        target_col = 'fraud_flag'

    df = df.dropna(subset=[target_col])

    # --- Convert timestamp ---
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col).reset_index(drop=True)

        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 18)).astype(int)

        # ⭐ ADD (important)
        df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

        #df = df.drop(columns=[time_col])

    # --- Drop useless columns ---
    df.drop(columns=['transaction_id', 'wallet_id', 'agent_id'], inplace=True, errors='ignore')

    # --- Handle missing values ---
    #df.fillna(0, inplace=True)
    df = df.fillna(0)

    # --- Convert boolean → int ---
    df[target_col] = df[target_col].astype(int)
    if 'churn_30d' in df.columns:
        df['churn_30d'] = df['churn_30d'].astype(int)

    # --- One-hot encoding (FIXED) ---
    df = pd.get_dummies(df, columns=[
        'transaction_type', 'channel', 'device_os', 'kyc_tier'
    ], drop_first=True)

    # --- Interaction Features (NEW) ---

    # Avoid division by zero
    df['amount_per_balance'] = df['amount_ngn'] / (df['balance_after_ngn'] + 1)

    # Cashout indicator (only if exists after encoding)
    cashout_cols = [col for col in df.columns if 'transaction_type_cashout' in col]
    if len(cashout_cols) > 0:
        df['is_cashout'] = df[cashout_cols[0]]
    else:
        df['is_cashout'] = 0

    # Amount * night interaction
    df['night_amount'] = df['amount_ngn'] * df['is_night']

    # --- Scaling ---
    if 'amount_ngn' in df.columns:
        scaler = StandardScaler()
        df[['amount_ngn']] = scaler.fit_transform(df[['amount_ngn']])

    return df

def temporal_split_and_scale(df, target_col='fraud_flag'):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # --- Split features and target ---
    X = df.drop(columns=[target_col, 'timestamp'], errors='ignore')
    y = df[target_col]

    # --- Temporal split (60-20-20) ---
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # --- Scaling (ONLY here → no leakage) ---
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test