from preprocessing import load_and_clean_data
from lstm_model import train_model, predict_future_prices
from plotting import plot_results
from optimization import optimize_parameters
from classes import User_Options, Optimization_Options


if __name__ == '__main__':
    user_options = User_Options(
        file_path='HistoricalData_1692981828643_GME_NASDAQ.csv',
        feature_cols=['Close/Last', 'Volume'],
        days_to_predict=15,
        perform_optimization=True,
        use_early_stopping=True,
        d_train_size_ratio=0.8,
        d_scaling_method='minmax',
        d_sequence_length=90,
        d_epochs=100,
        d_lstm_units=150,
        d_dropout_rate=0.1,
        d_batch_size=16,
        d_optimizer='adam'
    )
    optimization_options = Optimization_Options(
        scaling_method_options=['minmax', 'log'],
        sequence_length_options=[60, 90, 120],
        epochs_options=[50],
        train_size_ratio_options=[0.75, 0.8, 0.85],
        lstm_units_options=[30, 50, 70, 100, 150, 200],
        dropout_rate_options=[0.1, 0.2, 0.3, 0.4, 0.5],
        batch_size_options=[8, 16, 32, 64, 128, 256],
        optimizer_options=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl']
    )

    # Load and clean data
    df = load_and_clean_data(user_options.file_path, user_options.feature_cols)

    # Perform hyperparameter optimization if specified and train the model
    if user_options.perform_optimization:
        lowest_error, best_params = optimize_parameters(df, user_options, optimization_options)
        print(f"Best MSE: {lowest_error}, Best Parameters: {best_params}")
        #model, model_data = train_model()
    else:
        model, model_data = train_model(df, user_options)
        future_predictions = predict_future_prices(model, model_data.x_test[-1], user_options.days_to_predict, model_data.scaler)
        plot_results(model_data, user_options, future_predictions)

    

# TO-DO:
# Fix Optimization
#
# Future predictions fit to weekends
# Check if data is non-stationary and fix if so
# Fix input data to have 5 data points per week
# Cross-Validation folding
