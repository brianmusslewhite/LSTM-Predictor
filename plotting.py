import matplotlib.pyplot as plt
import pandas as pd


def plot_results(train_dates, y_train, test_dates, y_test, y_pred, future_dates, y_pred_future, scaler, params):
    n_features = len(params.feature_cols)

    fig, axes = plt.subplots(n_features, 1, figsize=(15, 6 * n_features))

    if n_features == 1:
        axes = [axes]

    for feature_idx, feature_name in enumerate(params.feature_cols):
        ax = axes[feature_idx]

        y_train_actual = scaler.inverse_transform(y_train)[:, feature_idx]
        y_test_actual = scaler.inverse_transform(y_test)[:, feature_idx]
        y_pred_actual = scaler.inverse_transform(y_pred)[:, feature_idx]
        y_pred_future_actual = scaler.inverse_transform(y_pred_future)[:, feature_idx]

        ax.plot(pd.to_datetime(train_dates), y_train_actual, label=f'Actual Train {feature_name}', color='purple')
        ax.plot(pd.to_datetime(test_dates), y_test_actual, label=f'Actual Test {feature_name}', color='blue')
        ax.plot(pd.to_datetime(test_dates), y_pred_actual, label=f'Predicted Test {feature_name}', color='red')
        ax.plot(pd.to_datetime(future_dates), y_pred_future_actual, label=f'Future {feature_name}', color='orange')

        ax.set_title(f'{feature_name} Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel(feature_name)
        ax.legend()

    plt.tight_layout()
    plt.show()
