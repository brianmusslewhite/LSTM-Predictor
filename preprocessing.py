import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def load_and_preprocess_data(file_path, feature_cols):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    sorted_df = df.sort_values('Date', ignore_index=True)

    for col in feature_cols:
        if sorted_df[col].dtype == 'object':
            sorted_df[col] = sorted_df[col].str.replace('$', '').astype(float)

    return sorted_df


def normalize_data(df, feature_cols, train_size_ratio, scaling_method='minmax'):
    selected_data = df[feature_cols].values.astype(float)

    train_size = int(train_size_ratio * len(df))
    train_data = selected_data[:train_size]
    test_data = selected_data[train_size:]

    if scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)
    elif scaling_method == 'log':
        log_scaler = FunctionTransformer(np.log1p, np.expm1, validate=True)
        log_scaler.fit(train_data)
        normalized_train_data = log_scaler.transform(train_data)
        normalized_test_data = log_scaler.transform(test_data)
        scaler = log_scaler
    else:
        raise ValueError("Invalid scaling method. Choose either 'minmax' or 'log'.")

    normalized_data = np.concatenate((normalized_train_data, normalized_test_data), axis=0)

    return normalized_data, scaler, train_size


def prepare_training_data(normalized_data, sequence_length, prediction_length, train_size):
    x, y = [], []

    for i in range(len(normalized_data) - sequence_length - prediction_length):
        x.append(normalized_data[i:i + sequence_length])
        y.append(normalized_data[i + sequence_length:i + sequence_length + prediction_length])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, y_train, x_test, y_test
