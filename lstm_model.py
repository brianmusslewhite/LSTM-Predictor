import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


def train_model(x_train, y_train, x_test=None, y_test=None, lstm_units=150, dropout_rate=0.1, batch_size=16, epochs=50, use_early_stopping=True, optimizer_name='nadam'):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=y_train.shape[1], activation='relu')
    ])

    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    if use_early_stopping and x_test is not None and y_test is not None:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks_list = [early_stopping]
    else:
        callbacks_list = []

    validation_data = (x_test, y_test) if x_test is not None and y_test is not None else None

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks_list, verbose=0)

    return model


def predict_future_prices(model, last_known_sequence, scaler, params):
    future_data = {}
    current_sequence = last_known_sequence.copy()

    for i in range(params.days_to_predict):
        # Predict the next value based on the current sequence
        predicted = model.predict(current_sequence[np.newaxis, :, :])[0]

        # Save the predictions
        for feature_idx, feature_name in enumerate(params.feature_cols):
            if feature_name not in future_data:
                future_data[feature_name] = []
            future_data[feature_name].append(predicted[feature_idx])

        # Prepare the input for the next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = predicted

    # Inverse transform the scaled predictions
    for feature_name in params.feature_cols:
        feature_data = np.array(future_data[feature_name]).reshape(-1, 1)
        dummy_array = np.zeros((len(feature_data), scaler.scale_.shape[0]))
        feature_index = params.feature_cols.index(feature_name)
        dummy_array[:, feature_index] = feature_data[:, 0]
        future_data[feature_name] = scaler.inverse_transform(dummy_array)[:, feature_index]

    return future_data
