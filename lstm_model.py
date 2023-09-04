import numpy as np

import tensorflow as tf
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
    if user_options.d_optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=user_options.d_learning_rate,
            beta_1=user_options.d_beta_1,
            beta_2=user_options.d_beta_2
        )
    elif user_options.d_optimizer == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax(
            learning_rate=user_options.d_learning_rate,
            beta_1=user_options.d_beta_1,
            beta_2=user_options.d_beta_2
        )
    elif user_options.d_optimizer == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=user_options.d_learning_rate,
            beta_1=user_options.d_beta_1,
            beta_2=user_options.d_beta_2
        )
    else:
        raise ValueError(f"Unknown optimizer {user_options.d_optimizer}")

    model = Sequential()

    if user_options.d_model_type == 'lstm':
        model.add(LSTM(units=user_options.d_lstm_units, return_sequences=True, input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])))
        model.add(Dropout(user_options.d_dropout_rate))

        for _ in range(1, user_options.d_model_layers):
            model.add(LSTM(units=user_options.d_lstm_units, return_sequences=True))
            model.add(Dropout(user_options.d_dropout_rate))

    elif user_options.d_model_type == 'lstmbidirectional':
        model.add(Bidirectional(LSTM(units=user_options.d_lstm_units, return_sequences=True, recurrent_regularizer='l1'), input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])))
        model.add(Dropout(user_options.d_dropout_rate))

        for _ in range(1, user_options.d_model_layers):
            model.add(Bidirectional(LSTM(units=user_options.d_lstm_units, return_sequences=True)))
            model.add(Dropout(user_options.d_dropout_rate))

    # Additional Dense Layer
    # model.add(TimeDistributed(Dense(units=20, activation='relu')))

    model.add(TimeDistributed(Dense(units=2)))
    model.add(Lambda(lambda x: x[:, -user_options.days_to_predict:, :]))

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def predict_future(model, last_sequence, n_future_predictions, scaler, num_features):
    future_predictions = np.zeros((n_future_predictions, num_features))
    input_data = last_sequence.copy()

    for i in range(n_future_predictions):
        model_input = np.expand_dims(input_data[-num_features:], axis=0)
        prediction = model.predict(model_input)[0, -1, :]
        future_predictions[i, :] = prediction
        input_data = np.delete(input_data, 0, axis=0)
        input_data = np.vstack([input_data, prediction])

    if scaler is not None:
        future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions
