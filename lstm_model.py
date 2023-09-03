import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Lambda, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import process_for_model


def train_model(df, user_options):
    model_data = process_for_model(df, user_options)
    model = build_model(model_data, user_options)

    if user_options.use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks_list = [early_stopping]
    else:
        callbacks_list = []

    model.fit(model_data.x_train, model_data.y_train, epochs=user_options.d_epochs, batch_size=user_options.d_batch_size, validation_data=(model_data.x_test, model_data.y_test), callbacks=callbacks_list, verbose=0)
    model_data.y_predicted_test = model.predict(model_data.x_test)

    return model, model_data


def build_model(model_data, user_options):
    model = Sequential()

    if user_options.d_model_type == 'default':
        model.add(LSTM(units=user_options.d_lstm_units, return_sequences=True, input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])))
        model.add(Dropout(user_options.d_dropout_rate))

        for _ in range(1, user_options.d_num_layers):
            model.add(LSTM(units=user_options.d_lstm_units, return_sequences=True))
            model.add(Dropout(user_options.d_dropout_rate))

    elif user_options.d_model_type == 'bidirectional':
        model.add(Bidirectional(LSTM(units=user_options.d_lstm_units, return_sequences=True), input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])))
        model.add(Dropout(user_options.d_dropout_rate))

        for _ in range(1, user_options.d_num_layers):
            model.add(Bidirectional(LSTM(units=user_options.d_lstm_units, return_sequences=True)))
            model.add(Dropout(user_options.d_dropout_rate))

    model.add(TimeDistributed(Dense(units=2)))
    model.add(Lambda(lambda x: x[:, -user_options.days_to_predict:, :]))

    model.compile(optimizer=user_options.d_optimizer, loss='mean_squared_error')

    return model


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
