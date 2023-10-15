from preprocessing import load_and_preprocess_data, normalize_data, prepare_training_data
from lstm_model import train_model, predict_next_n_days
from plotting import plot_results
from keras_tuner.tuners import BayesianOptimization
import pandas as pd


class Parameters:
    def __init__(self, file_path, feature_cols, sequence_length, epochs, perform_optimization, use_early_stopping, train_size_ratio):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.perform_optimization = perform_optimization
        self.use_early_stopping = use_early_stopping
        self.train_size_ratio = train_size_ratio


def build_model(hp):
    lstm_units1 = hp.Int('lstm_units1', 50, 1000, step=50)
    lstm_units2 = hp.Int('lstm_units2', 50, 1000, step=50)
    dropout_rate1 = hp.Float('dropout_rate1', 0.01, 0.5, step=0.01)
    dropout_rate2 = hp.Float('dropout_rate2', 0.01, 0.5, step=0.01)
    batch_size = hp.Int('batch_size', 8, 64, step=8)
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'nadam'])

    return train_model(x_train, y_train, x_test, y_test, lstm_units1=lstm_units1, lstm_units2=lstm_units2, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, batch_size=batch_size, optimizer_name=optimizer_choice)


if __name__ == '__main__':
    params = Parameters(
        file_path='/home/p1g3/Documents/LSTM Predictor/HistoricalData_1692981828643_GME_NASDAQ.csv',
        feature_cols=['Close/Last', 'Volume'],
        sequence_length=90,
        epochs=100,
        perform_optimization=True,
        use_early_stopping=True,
        train_size_ratio=0.8,
    )

    # Load, preprocess, and prep data
    raw_df = load_and_preprocess_data(params.file_path, params.feature_cols)
    normalized_data, scaler, train_size = normalize_data(raw_df, params.feature_cols, params.train_size_ratio)
    x_train, y_train, x_test, y_test = prepare_training_data(normalized_data, params.sequence_length, train_size)

    # Extract corresponding dates for training and test sets for plotting
    train_dates = raw_df['Date'][params.sequence_length: params.sequence_length + len(y_train)]
    test_dates = raw_df['Date'][train_size + params.sequence_length: train_size + params.sequence_length + len(y_test)]

    # If true, run optimization
    if params.perform_optimization:
        tuner = BayesianOptimization(
            build_model,
            objective='val_loss',
            max_trials=2,
            executions_per_trial=1,
            directory='tuner_results',
            project_name='lstm_tuning_oct15_5'
        )

        tuner.search(x_train, y_train, epochs=params.epochs, validation_data=(x_test, y_test))

        best_model = tuner.get_best_models(num_models=1)[0]
        trained_model = best_model
    else:
        # Train model with defaults
        trained_model = train_model(x_train, y_train, x_test, y_test)

    # Predict stock prices of the test data
    y_pred_test = trained_model.predict(x_test)

    # Predict price for next sequence length and create corresponding dates
    last_sequence = x_test[-1:]  # Take the last sequence from x_test
    n_days = params.sequence_length
    y_pred_future = predict_next_n_days(trained_model, last_sequence, n_days)

    last_known_date = raw_df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=n_days, freq='D')

    plot_results(train_dates, y_train, test_dates, y_test, y_pred_test, future_dates, y_pred_future, scaler, params)
