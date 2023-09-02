import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed, Lambda
from tensorflow.keras.callbacks import EarlyStopping


def train_model(x_train, y_train, x_test, y_test, params, lstm_units=150, dropout_rate=0.1, batch_size=16, epochs=100, use_early_stopping=True, optimizer_name='adam'):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        TimeDistributed(Dense(units=2, activation='relu')),
        Lambda(lambda x: x[:, -params.days_to_predict:, :])
    ])

    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks_list = [early_stopping]
    else:
        callbacks_list = []

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=0)

    return model


def predict_future_prices(model, last_known_sequence, scaler, params):
    future_data = {}
    current_sequence = last_known_sequence.copy()

    for feature_name in params.feature_cols:
        future_data[feature_name] = []

    steps_to_predict = last_known_sequence.shape[0]

    for i in range(0, params.days_to_predict, steps_to_predict):
        predicted = model.predict(current_sequence[np.newaxis, :, :])[0]
        predicted = predicted.reshape(-1, len(params.feature_cols))
        predicted_transformed = scaler.inverse_transform(predicted)

        for j, feature_name in enumerate(params.feature_cols):
            future_data[feature_name].extend(predicted_transformed[:, j])

        current_sequence = np.vstack((current_sequence[predicted.shape[0]:, :], predicted))

    return future_data
