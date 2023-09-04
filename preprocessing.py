import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer


class Model_Data:
    def __init__(self, x_train, y_train, x_test, y_test, train_size, train_dates, test_dates, scaler, y_predicted_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_size = train_size
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.scaler = scaler
        self.y_prediction_test = y_predicted_test


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
def process_for_model(df, user_options=None):
    if user_options is None:
        train_size_ratio = 0.8
        scaling_method = 'minmax'
        sequence_length = 90
        days_to_predict = 5
        feature_cols = ['Close/Last', 'Volume']
    else:
        train_size_ratio = user_options.train_size
        scaling_method = user_options.scaling_method
        sequence_length = user_options.sequence_length
        days_to_predict = user_options.days_to_predict
        feature_cols = user_options.feature_cols

    # Convert to float. Potentially move this to load and clean
    selected_data = df[feature_cols].values.astype(float)

    train_size = int(train_size_ratio * len(df))
    train_data = selected_data[:train_size]
    test_data = selected_data[train_size:]

    original_dates = df['Date'].values

    if scaling_method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)
    elif scaling_method == 'standard':
        scaler = StandardScaler()
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)
    elif scaling_method == 'robust':
        scaler = RobustScaler()
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)
    elif scaling_method == 'power':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)
    elif scaling_method == 'quantile':
        scaler = QuantileTransformer(output_distribution='uniform')
        scaler.fit(train_data)
        normalized_train_data = scaler.transform(train_data)
        normalized_test_data = scaler.transform(test_data)    
    elif scaling_method is None:
        normalized_train_data = train_data
        normalized_test_data = test_data
        scaler = None
    else:
        raise ValueError("Invalid scaling method. Choose either 'minmax', 'log', or None.")

    normalized_data = np.concatenate((normalized_train_data, normalized_test_data), axis=0)

    # Generate x and y using the entire normalized dataset
    x, y = make_multi_step(normalized_data, sequence_length, days_to_predict)

    # The adjusted_train_size should be calculated as the original train_size,
    # minus the offsets for the sequence and prediction lengths
    adjusted_train_size = train_size - (sequence_length + days_to_predict)
    adjusted_test_size = len(test_data) - (sequence_length + days_to_predict)

    # Split the sequence data into training and test portions
    x_train, x_test = x[:adjusted_train_size], x[adjusted_train_size:]
    y_train, y_test = y[:adjusted_train_size], y[adjusted_train_size:]

    train_dates = original_dates[sequence_length: sequence_length + adjusted_train_size]
    test_dates = original_dates[train_size + sequence_length + days_to_predict: train_size + sequence_length + days_to_predict + adjusted_test_size]

    model_data = Model_Data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
                            train_size=adjusted_train_size, train_dates=train_dates, 
                            test_dates=test_dates, scaler=scaler, y_predicted_test=None)

    return model_data


def make_multi_step(normalized_data, sequence_length, days_to_predict):
    x, y = [], []

    for i in range(len(normalized_data) - sequence_length - days_to_predict):
        x.append(normalized_data[i:i + sequence_length])
        y.append(normalized_data[i + sequence_length:i + sequence_length + days_to_predict])

    x = np.array(x)
    y = np.array(y)

    return x, y
