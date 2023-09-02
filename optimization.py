from datetime import datetime
import itertools
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from lstm_model import train_model


def optimize_parameters(x_train, y_train, params):
    parameter_combinations = list(itertools.product(params.lstm_units_options, params.dropout_rate_options, params.batch_size_options, params.optimizer_options))
    optimization_results = []

    if not os.path.exists('optimization_results'):
        os.makedirs('optimization_results')

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'optimization_results/optimization_results_{now}.txt'

    def train_and_evaluate(potential_params):
        units, dropout, batch, optimizer_name = potential_params
        mse_cv = time_series_cross_val(x_train, y_train, params, units, dropout, batch, params.epochs, params.use_early_stopping, optimizer_name)
        optimization_results.append({'mse': mse_cv, 'params': potential_params})

        with open(file_name, 'a') as f:
            f.write(f"MSE: {mse_cv}, Parameters: LSTM Units - {units}, Dropout Rate - {dropout}, Batch Size - {batch}, Optimizer - {optimizer_name}\n")

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
