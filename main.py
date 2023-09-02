from preprocessing import load_and_preprocess_data, normalize_data, prepare_training_data
from optimization import optimize_parameters
from lstm_model import train_model, predict_future_prices
from plotting import plot_results


class Parameters:
    def __init__(self, file_path, feature_cols, scaling_method, sequence_length, epochs, days_to_predict, perform_optimization, use_early_stopping, train_size_ratio, sequence_length_options, lstm_units_options, dropout_rate_options, batch_size_options, optimizer_options):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.scaling_method = scaling_method
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.days_to_predict = days_to_predict
        self.perform_optimization = perform_optimization
        self.use_early_stopping = use_early_stopping
        self.train_size_ratio = train_size_ratio
        self.sequence_length_options = sequence_length_options
        self.lstm_units_options = lstm_units_options
        self.dropout_rate_options = dropout_rate_options
        self.batch_size_options = batch_size_options
        self.optimizer_options = optimizer_options


if __name__ == '__main__':
    params = Parameters(
        file_path='HistoricalData_1692981828643_GME_NASDAQ_short.csv',
        feature_cols=['Close/Last', 'Volume'],
        scaling_method='minmax',
        sequence_length=90,
        epochs=100,
        days_to_predict=30,
        perform_optimization=False,
        use_early_stopping=True,
        train_size_ratio=0.8,
        sequence_length_options=[60, 90],
        lstm_units_options=[30, 50, 70, 100, 150],  # [30, 50, 70, 100, 150, 200],
        dropout_rate_options=[0.1, 0.2, 0.3],  # [0.1, 0.2, 0.3, 0.4, 0.5]
        batch_size_options=[8, 16, 32, 64],  # [8, 16, 32, 64, 128, 256]
        optimizer_options=['adam']  # ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl']
    )

    # Load, preprocess, and prep data
    raw_df = load_and_preprocess_data(params.file_path, params.feature_cols)
    normalized_data, scaler, train_size = normalize_data(raw_df, params.feature_cols, params.train_size_ratio, params.scaling_method)
    x_train, y_train, x_test, y_test = prepare_training_data(normalized_data, params.sequence_length, params.days_to_predict, train_size)

    # Extract corresponding dates for training and test sets for plotting
    train_dates = raw_df['Date'][params.sequence_length + 1:params.sequence_length + 1 + len(y_train)]
    test_dates = raw_df['Date'][train_size + params.sequence_length + 1:train_size + params.sequence_length + 1 + len(y_test)]

    # Perform hyperparameter optimization if specified
    if params.perform_optimization:
        best_mse, best_params = optimize_parameters(x_train, y_train, x_test, y_test, params, False)
        print(f"Best MSE: {best_mse}, Best Parameters: {best_params}")
        model = train_model(x_train, y_train, x_test, y_test, params, lstm_units=best_params[0], dropout_rate=best_params[1], batch_size=best_params[2], epochs=params.epochs, use_early_stopping=params.use_early_stopping, optimizer_name=best_params[3])
    else:
        model = train_model(x_train, y_train, x_test, y_test, params)

    # Predict stock prices of the test data
    y_pred_test = model.predict(x_test)

    # Predict future prices
    last_known_sequence = x_test[-1]
    future_prices = predict_future_prices(model, last_known_sequence, scaler, params)

    # Plot the results
    if params.perform_optimization:
        plot_results(train_dates, y_train, test_dates, y_test, y_pred_test, future_prices, scaler, params, best_params)
    else:
        plot_results(train_dates, y_train, test_dates, y_test, y_pred_test, future_prices, scaler, params)
