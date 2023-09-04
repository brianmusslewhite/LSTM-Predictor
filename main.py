from preprocessing import load_and_clean_data
from lstm_model import train_model, predict_future
from plotting import plot_results
from optimization import optimize_parameters
from classes import User_Options, Optimization_Options


if __name__ == '__main__':
    user_options = User_Options(
        file_path='HistoricalData_1692981828643_GME_NASDAQ.csv',
        feature_cols=['Close/Last', 'Volume'],
        days_to_predict=3,
        days_to_forecast=20,
        perform_optimization=True,
        use_early_stopping=True,
        d_train_size_ratio=0.8,
        d_scaling_method='robust',
        d_sequence_length=90,
        d_epochs=100,
        d_lstm_units=64,
        d_dropout_rate=0.3,
        d_batch_size=8,
        d_optimizer='Adamax',
        d_learning_rate=1e-2,
        d_beta_1=0.9,
        d_beta_2=0.995,
        d_model_type='lstmbidirectional',
        d_num_layers=2
    )
    optimization_options = Optimization_Options(
        scaling_method_options=['standard', 'robust', 'power', 'quantile', 'minmax'],
        sequence_length_options=[60],  # [30, 60, 90, 120],
        epochs_options=[100],
        train_size_ratio_options=[0.8],
        lstm_units_options=[128],  # [32, 64, 128, 256],
        dropout_rate_options=[0.3],
        batch_size_options=[8],
        optimizer_options=['Adamax', 'Nadam'],
        learning_rate_options=[1e-2, 1e-3],
        beta_1_options=[0.9],
        beta_2_options=[0.995],
        model_type_options=['lstmbidirectional'],
        model_layer_options=[2]
    )

    # Load and clean data
    df = load_and_clean_data(user_options.file_path, user_options.feature_cols)

    # Perform hyperparameter optimization if specified and train the model
    if user_options.perform_optimization:
        lowest_error, best_params = optimize_parameters(df, user_options, optimization_options)
        print(f"Best MSE: {lowest_error}, Best Parameters: {best_params}")
    else:
        model, model_data = train_model(df, user_options)
        future_predictions = predict_future(model, model_data.x_test[-1], user_options.days_to_forecast, model_data.scaler, len(user_options.feature_cols))
        plot_results(model_data, user_options, future_predictions)

# TO-DO:
# Plot optimal model after optimization
# Update plot print file name
# Future predictions fit to weekends
# Check if data is non-stationary and fix if so
# Fix input data to have 5 data points per week (weekends)
# Cross-Validation folding
