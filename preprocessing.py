import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


class Model_Data:
    def __init__(self, x_train, y_train, x_test, y_test, test_start_date, train_start_date, scaler, y_prediction_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.test_start_date = test_start_date
        self.train_start_date = train_start_date
        self.scaler = scaler
        self.y_prediction_test = y_prediction_test


# Loads data from a file, converts date to datetime, sorts
# removes '$' signs, and only returns the date + feature columns
def load_and_clean_data(file_path, feature_cols):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    sorted_df = df.sort_values('Date', ignore_index=True)

    for col in feature_cols:
        if sorted_df[col].dtype == 'object':
            sorted_df[col] = sorted_df[col].str.replace('$', '').astype(float)

    sorted_df = sorted_df[['Date'] + feature_cols]

    return sorted_df


# Split, normalize, and make multi-step data that is ready for the model
def process_for_model(df, user_options, best_params=None):
    # Convert to float. Potentially move this to load and clean
    selected_data = df[user_options.feature_cols].values.astype(float)

    if best_params is None:
        train_size_ratio = user_options.d_train_size_ratio
        scaling_method = user_options.d_scaling_method
        sequence_length = user_options.d_sequence_length
    else:
        train_size_ratio = best_params.train_size
        scaling_method = best_params.scaling_method
        sequence_length = best_params.sequence_length

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
    elif scaling_method is None:
        normalized_train_data = train_data
        normalized_test_data = test_data
        scaler = None
    else:
        raise ValueError("Invalid scaling method. Choose either 'minmax', 'log', or None.")

    normalized_data = np.concatenate((normalized_train_data, normalized_test_data), axis=0)

    x_train, y_train, x_test, y_test = make_multi_step(normalized_data, sequence_length, user_options.days_to_predict, train_size)

    train_start_date = df['Date'].iloc[0]
    test_start_date = df['Date'].iloc[train_size]

    model_data = Model_Data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, test_start_date=test_start_date, train_start_date=train_start_date, scaler=scaler, y_prediction_test=None)

    return model_data


def make_multi_step(normalized_data, sequence_length, days_to_predict, train_size):
    x, y = [], []

    for i in range(len(normalized_data) - sequence_length - days_to_predict):
        x.append(normalized_data[i:i + sequence_length])
        y.append(normalized_data[i + sequence_length:i + sequence_length + days_to_predict])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return x_train, y_train, x_test, y_test
