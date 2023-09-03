from datetime import datetime
import copy
import itertools
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from lstm_model import train_model


def optimize_parameters(df, user_options, optimization_options):
    parameter_combinations = list(itertools.product(optimization_options.scaling_method_options, optimization_options.sequence_length_options, optimization_options.epochs_options,
                                                    optimization_options.train_size_ratio_options, optimization_options.lstm_units_options, optimization_options.dropout_rate_options, 
                                                    optimization_options.batch_size_options, optimization_options.optimizer_options))
    optimization_results = []

    if not os.path.exists('optimization_results'):
        os.makedirs('optimization_results')

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'optimization_results/optimization_results_{now}.txt'

    def train_and_evaluate(potential_params):
        scaling_method, sequence_length, epochs, train_size_ratio, lstm_units, dropout_rate, batch_size, optimizer = potential_params

        temp_user_options = copy.deepcopy(user_options)
        temp_user_options.d_scaling_method = scaling_method
        temp_user_options.d_sequence_length = sequence_length
        temp_user_options.d_epochs = epochs
        temp_user_options.d_train_size_ratio = train_size_ratio
        temp_user_options.d_lstm_units = lstm_units
        temp_user_options.d_dropout_rate = dropout_rate
        temp_user_options.d_batch_size = batch_size
        temp_user_options.d_optimizer = optimizer

        model, model_data = train_model(df, temp_user_options)

        y_pred = model.predict(model_data.x_test)

        timesteps = model_data.y_train.shape[1]
        features = model_data.y_train.shape[2]

        mse_per_timestep = []

        for t in range(timesteps):
            for f in range(features):
                mse = mean_squared_error(model_data.y_test[:, t, f], y_pred[:, t, f])
                mse_per_timestep.append(mse)

        mse = np.mean(mse_per_timestep)

        optimization_results.append({'mse': mse, 'params': potential_params})

        with open(file_name, 'a') as f:
            f.write(f"MSE: {mse}, Parameters: Scaling Method - {scaling_method} Sequence Length - {sequence_length} Epochs - {epochs} Train Size Ratio - {train_size_ratio} LSTM Units - {lstm_units}, Dropout Rate - {dropout_rate}, Batch Size - {batch_size}, Optimizer - {optimizer}\n")

    for potential_params in tqdm(parameter_combinations, total=len(parameter_combinations)):
        train_and_evaluate(potential_params)

    best_result = min(optimization_results, key=lambda x: x['mse'])
    return best_result['mse'], best_result['params']


def time_series_cross_val(x, y, params, lstm_units, dropout_rate, batch_size, epochs, use_early_stopping, optimizer_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_idx, test_idx in tscv.split(x):
        x_train_cv, y_train_cv = x[train_idx], y[train_idx]
        x_test_cv, y_test_cv = x[test_idx], y[test_idx]
        model = train_model(x_train_cv, y_train_cv, x_test_cv, y_test_cv, params, lstm_units, dropout_rate, batch_size, epochs, use_early_stopping, optimizer_name)
        y_pred_cv = model.predict(x_test_cv)

        timesteps = y_test_cv.shape[1]
        features = y_test_cv.shape[2]

        mse_per_timestep = []

        for t in range(timesteps):
            for f in range(features):
                mse = mean_squared_error(y_test_cv[:, t, f], y_pred_cv[:, t, f])
                mse_per_timestep.append(mse)

        mse_avg = np.mean(mse_per_timestep)
        mse_scores.append(mse_avg)

    return np.mean(mse_scores)
