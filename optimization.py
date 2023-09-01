from datetime import datetime
import itertools
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from lstm_model import train_model


def optimize_parameters(X_train, y_train, params):
    parameter_combinations = list(itertools.product(params.lstm_units_options, params.dropout_rate_options, params.batch_size_options, params.optimizer_options))

    optimization_results = []

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'optimization_results/optimization_results_{now}.txt'

    def train_and_evaluate(potential_params):
        units, dropout, batch, optimizer_name = potential_params

        mse_cv = time_series_cross_val(X_train, y_train, units, dropout, batch, params.epochs, params.use_early_stopping, optimizer_name)
        optimization_results.append({'mse': mse_cv, 'params': potential_params})

        with open(file_name, 'a') as f:
            f.write(f"MSE: {mse_cv}, Parameters: LSTM Units - {units}, Dropout Rate - {dropout}, Batch Size - {batch}, Optimizer - {optimizer_name}\n")

    for potential_params in tqdm(parameter_combinations, total=len(parameter_combinations)):
        train_and_evaluate(potential_params)

    best_result = min(optimization_results, key=lambda x: x['mse'])

    return best_result['mse'], best_result['params']


def time_series_cross_val(X, y, lstm_units, dropout_rate, batch_size, epochs, use_early_stopping, optimizer_name, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_test_cv, y_test_cv = X[test_idx], y[test_idx]
        model = train_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv, lstm_units, dropout_rate, batch_size, epochs, use_early_stopping, optimizer_name)
        y_pred_cv = model.predict(X_test_cv)
        mse = mean_squared_error(y_test_cv, y_pred_cv)
        mse_scores.append(mse)

    return np.mean(mse_scores)
