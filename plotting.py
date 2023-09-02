import numpy as np
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(train_dates, y_train, test_dates, y_test, y_pred, future_data, scaler, params, best_params=None):
    n_features = len(params.feature_cols)

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 6 * n_features))

    if n_features == 1:
        axes = [axes]

    for feature_idx, feature_name in enumerate(params.feature_cols):
        ax = axes[feature_idx]

        y_train_actual = np.concatenate([scaler.inverse_transform(y_train[i])[-1, :].reshape(1, -1) for i in range(y_train.shape[0])], axis=0)[:, feature_idx]
        y_test_actual = np.concatenate([scaler.inverse_transform(y_test[i])[-1, :].reshape(1, -1) for i in range(y_test.shape[0])], axis=0)[:, feature_idx]
        y_pred_actual = np.concatenate([scaler.inverse_transform(y_pred[i])[-1, :].reshape(1, -1) for i in range(y_pred.shape[0])], axis=0)[:, feature_idx]

        ax.plot(pd.to_datetime(train_dates), y_train_actual, label=f'Actual Train {feature_name}', color='purple')
        ax.plot(pd.to_datetime(test_dates), y_test_actual, label=f'Actual Test {feature_name}', color='blue')
        ax.plot(pd.to_datetime(test_dates), y_pred_actual, label=f'Predicted Test {feature_name}', color='red')

        # Plotting future data (if available)
        if future_data.get(feature_name) is not None:
            future_indices = pd.date_range(pd.to_datetime(test_dates).iloc[-1], periods=len(future_data[feature_name]) + 1)[1:]
            ax.plot(future_indices, future_data[feature_name], label=f'Predicted Future {feature_name}', color='green', linestyle='--')

        ax.set_title(f'{feature_name} Prediction (Multi-Step if applicable)')
        ax.set_xlabel('Date')
        ax.set_ylabel(feature_name)
        ax.legend()

    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    file_name = params.file_path.split('/')[-1].replace('.csv', '')

    if params.perform_optimization:
        filename = f"pictures/{file_name}_SL-{params.sequence_length}_E-{params.epochs}_FD-{params.days_to_predict}_OPT-{params.perform_optimization}_ES-{params.use_early_stopping}_TSR-{params.train_size_ratio}_BestParams-LSTM-{best_params[0]}_Dropout-{best_params[1]}_Batch-{best_params[2]}_Optimizer-{best_params[3]}.png"
    else:
        filename = f"pictures/{file_name}_SL-{params.sequence_length}_E-{params.epochs}_FD-{params.days_to_predict}_OPT-{params.perform_optimization}_ES-{params.use_early_stopping}_TSR-{params.train_size_ratio}_Optimizer-adam.png"

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()
