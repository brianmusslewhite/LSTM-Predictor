from preprocessing import load_and_clean_data
from lstm_model import train_model
from plotting import plot_results


class User_Options:
    def __init__(self, file_path, feature_cols, days_to_predict, perform_optimization, use_early_stopping,
                 d_scaling_method, d_sequence_length, d_epochs, d_train_size_ratio, d_lstm_units,
                 d_dropout_rate, d_batch_size, d_optimizer):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.days_to_predict = days_to_predict
        self.perform_optimization = perform_optimization
        self.use_early_stopping = use_early_stopping
        self.d_train_size_ratio = d_train_size_ratio
        self.d_scaling_method = d_scaling_method
        self.d_sequence_length = d_sequence_length
        self.d_epochs = d_epochs
        self.d_lstm_units = d_lstm_units
        self.d_dropout_rate = d_dropout_rate
        self.d_batch_size = d_batch_size
        self.d_optimizer = d_optimizer


class Optimization_Options:
    def __init__(self, scaling_method_options, sequence_length_options, epochs_options, train_size_ratio_options,
                 lstm_units_options, dropout_rate_options, batch_size_options, optimizer_options):
        self.scaling_method_options = scaling_method_options
        self.sequence_length_options = sequence_length_options
        self.epochs_options = epochs_options
        self.train_size_ratio_options = train_size_ratio_options
        self.lstm_units_options = lstm_units_options
        self.dropout_rate_options = dropout_rate_options
        self.batch_size_options = batch_size_options
        self.optimizer_options = optimizer_options


if __name__ == '__main__':
    user_options = User_Options(
        file_path='HistoricalData_1692981828643_GME_NASDAQ.csv',
        feature_cols=['Close/Last', 'Volume'],
        days_to_predict=30,
        perform_optimization=False,
        use_early_stopping=True,
        d_train_size_ratio=0.8,
        d_scaling_method='minmax',
        d_sequence_length=90,
        d_epochs=100,
        d_lstm_units=150,
        d_dropout_rate=0.1,
        d_batch_size=32,
        d_optimizer='adam'
    )
    optimization_options = Optimization_Options(
        scaling_method_options=['minmax', 'log', None],
        sequence_length_options=[60, 90],
        epochs_options=[20, 30, 50, 100],
        train_size_ratio_options=[0.6, 0.7, 0.8, 0.9],
        lstm_units_options=[30, 50, 70, 100, 150, 200],
        dropout_rate_options=[0.1, 0.2, 0.3, 0.4, 0.5],
        batch_size_options=[8, 16, 32, 64, 128, 256],
        optimizer_options=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl']
    )

    # Load and clean data
    df = load_and_clean_data(user_options.file_path, user_options.feature_cols)

    # Perform hyperparameter optimization if specified and train the model
    if user_options.perform_optimization:
        #lowest_error, best_params = optimize_parameters(df, user_options, optimization_options)
        #print(f"Best MSE: {lowest_error}, Best Parameters: {best_params}")
        model, x_train, y_train, x_test, y_test = train_model()
    else:
        model, model_data = train_model(df, user_options)
        plot_results(model_data, user_options)
# TO-DO:
# Fix Dates in Plot
# Fix Optimization
#
#
# Check if data is non-stationary and fix if so
# Cross-Validation folding