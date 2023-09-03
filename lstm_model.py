import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Lambda
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import process_for_model


def train_model(df, user_options, best_params=None):
    # Split, normalize, and prepare data for a multi-step model
    model_data = process_for_model(df, user_options, best_params)

    print(f"Shape of model_data.x_train after process_for_model: {model_data.x_train.shape}")
    print(f"Shape of model_data.y_train after process_for_model: {model_data.y_train.shape}")
    print(f"Shape of model_data.x_test after process_for_model: {model_data.x_test.shape}")
    print(f"Shape of model_data.y_test after process_for_model: {model_data.y_test.shape}")

    if best_params is None:
        lstm_units = user_options.d_lstm_units
        dropout_rate = user_options.d_dropout_rate
        optimizer_name = user_options.d_optimizer
        epochs = user_options.d_epochs
        batch_size = user_options.d_batch_size
    else:
        lstm_units = best_params.lstm_units
        dropout_rate = best_params.dropout_rate
        optimizer_name = best_params.d_optimizer
        epochs = best_params.d_epochs
        batch_size = best_params.d_batch_size

    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        TimeDistributed(Dense(units=2)),
        Lambda(lambda x: x[:, -user_options.days_to_predict:, :])
    ])

    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    if user_options.use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks_list = [early_stopping]
    else:
        callbacks_list = []

    model.fit(model_data.x_train, model_data.y_train, epochs=epochs, batch_size=batch_size, validation_data=(model_data.x_test, model_data.y_test), callbacks=callbacks_list, verbose=0)

    # Create prediction for test data
    model_data.y_predicted_test = model.predict(model_data.x_test)

    print(f"Shape of model_data.y_predicted_test after fit: {model_data.x_train.shape}")

    return model, model_data


def predict_future_prices(model, last_sequence, n_future_predictions, scaler):
    future_predictions = []
    input_data = last_sequence.copy()  # Copy the last sequence from test data

    for _ in range(n_future_predictions):
        # Reshape and expand dims to fit the input shape of the model: (batch_size, sequence_length, num_features)
        model_input = np.expand_dims(input_data, axis=0)

        # Make a prediction
        prediction = model.predict(model_input)

        # Take the last time step from the predicted sequence and append it to future_predictions
        last_timestep_prediction = prediction[0, -1, :]
        future_predictions.append(last_timestep_prediction)

        # Remove the first time step from the input sequence
        input_data = input_data[1:, :]

        # Append the last_timestep_prediction to the input sequence
        input_data = np.vstack([input_data, last_timestep_prediction])

    # If your data was scaled, apply inverse transform to the future predictions
    if scaler is not None:
        future_predictions = scaler.inverse_transform(future_predictions)

    return np.array(future_predictions)