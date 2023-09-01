import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_path, feature_cols):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    sorted_df = df.sort_values('Date', ignore_index=True)
    
    for col in feature_cols:
        if sorted_df[col].dtype == 'object':
            sorted_df[col] = sorted_df[col].str.replace('$', '').astype(float)
                    
    return sorted_df

def normalize_data(df, feature_cols, train_size_ratio):
    selected_data = df[feature_cols].values.astype(float)
    
    train_size = int(train_size_ratio * len(df))
    train_data = selected_data[:train_size]
    test_data = selected_data[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    normalized_train_data = scaler.transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    
    normalized_data = np.concatenate((normalized_train_data, normalized_test_data), axis=0)

    return normalized_data, scaler, train_size

def prepare_training_data(normalized_data, sequence_length, train_size):
    X, y = [], []
    
    for i in range(len(normalized_data) - sequence_length - 1):
        X.append(normalized_data[i:i + sequence_length])
        y.append(normalized_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test