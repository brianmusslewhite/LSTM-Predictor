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

    filename = (f"pictures/{file_name}_"
                f"Scale-{user_options.d_scaling_method}"
                f"SL-{user_options.d_sequence_length}_"
                f"E-{user_options.d_epochs}_"
                f"FD-{user_options.days_to_predict}_"
                f"OPT-{user_options.perform_optimization}_"
                f"ES-{user_options.use_early_stopping}_"
                f"TSR-{user_options.d_train_size_ratio}_"
                f"LSTM-{user_options.d_lstm_units}_"
                f"Dropout-{user_options.d_dropout_rate}_"
                f"Batch-{user_options.d_batch_size}_"
                f"Opt-{user_options.d_optimizer}_"
                f"LR-{user_options.d_learning_rate}_"
                f"B1-{user_options.d_beta_1}_"
                f"B2-{user_options.d_beta_2}_"
                f"MT-{user_options.d_model_type}_"
                f"ML-{user_options.d_model_layers}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()
