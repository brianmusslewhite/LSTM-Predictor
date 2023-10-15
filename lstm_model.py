from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


def train_model(x_train, y_train, x_test=None, y_test=None, lstm_units=150, dropout_rate=0.1, batch_size=16, epochs=50, use_early_stopping=True, optimizer_name='nadam'):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=y_train.shape[1])
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


def predict_next_n_days(model, last_sequence, n_days):
    predictions = []

    input_sequence = last_sequence.copy()

    for _ in range(n_days):
        # Predict the next step (1 day in the future)
        predicted_step = model.predict(input_sequence)
        predictions.append(predicted_step[0])

        # Update the input_sequence with the predicted value
        input_sequence[:, :-1, :] = input_sequence[:, 1:, :]
        input_sequence[:, -1, :] = predicted_step

    return predictions
