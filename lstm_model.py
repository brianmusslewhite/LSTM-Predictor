import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Lambda
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import process_for_model


def train_model(df, user_options, best_params=None):
    # Split, normalize, and prepare data for a multi-step model
    model_data = process_for_model(df, user_options, best_params)

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

    return model, model_data


def predict_future_prices(model, last_known_sequence, scaler, params):
    future_data = {}
    current_sequence = last_known_sequence.copy()

    for feature_name in params.feature_cols:
        future_data[feature_name] = []

    steps_to_predict = last_known_sequence.shape[0]

    for i in range(0, params.days_to_predict, steps_to_predict):
        predicted = model.predict(current_sequence[np.newaxis, :, :])[0]
        predicted = predicted.reshape(-1, len(params.feature_cols))

        if scaler:
            predicted_transformed = scaler.inverse_transform(predicted)
        else:
            predicted_transformed = predicted

        for j, feature_name in enumerate(params.feature_cols):
            future_data[feature_name].extend(predicted_transformed[:, j])

        current_sequence = np.vstack((current_sequence[predicted.shape[0]:, :], predicted))

    return future_data
