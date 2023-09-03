import numpy as np
import os

import matplotlib.pyplot as plt


def plot_results(model_data, user_options, future_predictions=None, best_params=None):
    n_features = len(user_options.feature_cols)

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 6 * n_features))

    if n_features == 1:
        axes = [axes]

    for feature_idx, feature_name in enumerate(user_options.feature_cols):
        ax = axes[feature_idx]

        if model_data.scaler:
            y_train_actual = np.concatenate([model_data.scaler.inverse_transform(model_data.y_train[i])[-1, :].reshape(1, -1) for i in range(model_data.y_train.shape[0])], axis=0)[:model_data.train_size, feature_idx]
            y_test_actual = np.concatenate([model_data.scaler.inverse_transform(model_data.y_test[i])[-1, :].reshape(1, -1) for i in range(model_data.y_test.shape[0])], axis=0)[:len(model_data.test_dates), feature_idx]
            y_pred_test_actual = np.concatenate([model_data.scaler.inverse_transform(model_data.y_predicted_test[i])[-1, :].reshape(1, -1) for i in range(model_data.y_predicted_test.shape[0])], axis=0)[:len(model_data.test_dates), feature_idx]

        else:
            y_train_actual = model_data.y_train[:, -1, feature_idx]
            y_test_actual = model_data.y_test[:, -1, feature_idx]
            y_pred_test_actual = model_data.y_pred[:, -1, feature_idx]

        ax.plot(model_data.train_dates, y_train_actual, label=f'Train Data {feature_name}', color='purple')
        ax.plot(model_data.test_dates, y_test_actual, label=f'Actual Data {feature_name}', color='blue')
        ax.plot(model_data.test_dates, y_pred_test_actual, label=f'Predicted Data {feature_name}', color='red')

        # Future prediction plots, if available
        if future_predictions is not None:
            last_date = np.datetime64(model_data.test_dates[-1])  # last date in your test_dates
            future_dates = np.array([last_date + np.timedelta64(x, 'D') for x in range(1, future_predictions.shape[0] + 1)])

            ax.plot(future_dates, future_predictions[:, feature_idx], label=f'Predicted Future Data {feature_name}', linestyle='--', color='green')

        ax.set_title(f'{feature_name} Multi-Step Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel(feature_name)
        ax.legend()

    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    file_name = user_options.file_path.split('/')[-1].replace('.csv', '')

    if user_options.perform_optimization:
        filename = f"pictures/{file_name}_SL-{params.sequence_length}_E-{params.epochs}_FD-{params.days_to_predict}_OPT-{params.perform_optimization}_ES-{params.use_early_stopping}_TSR-{params.train_size_ratio}_BestParams-LSTM-{best_params[0]}_Dropout-{best_params[1]}_Batch-{best_params[2]}_Optimizer-{best_params[3]}.png"
    else:
        filename = f"pictures/{file_name}_SL-{user_options.d_sequence_length}_E-{user_options.d_epochs}_FD-{user_options.days_to_predict}_OPT-{user_options.perform_optimization}_ES-{user_options.use_early_stopping}_TSR-{user_options.d_train_size_ratio}_Optimizer-{user_options.d_train_size_ratio}.png"

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()
