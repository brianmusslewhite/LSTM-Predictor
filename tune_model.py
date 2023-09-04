from keras_tuner import BayesianOptimization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization, Bidirectional, Lambda
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_and_clean_data, process_for_model


def build_model_function(model_data):
    def build_model(hp):
        model = Sequential()

        # First BiLSTM layer with BatchNormalization and Dropout
        model.add(Bidirectional(LSTM(units=hp.Int('first_lstm_units', min_value=64, max_value=256, step=16),
                                     return_sequences=True,
                                     recurrent_regularizer=l1_l2(l1=hp.Float('first_lstm_l1', min_value=1e-6, max_value=1e-2, sampling='LOG'), 
                                                                 l2=hp.Float('first_lstm_l2', min_value=1e-6, max_value=1e-2, sampling='LOG'))),
                                input_shape=((model_data.x_train.shape[1], model_data.x_train.shape[2]))))  # Replace your_input_shape with actual shape
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.05)))

        # Second LSTM layer
        model.add(LSTM(units=hp.Int('second_lstm_units', min_value=32, max_value=128, step=8), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.05)))

        # Third LSTM layer
        model.add(LSTM(units=hp.Int('third_lstm_units', min_value=32, max_value=128, step=8), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.05)))

        # Fourth LSTM layer
        model.add(LSTM(units=hp.Int('fourth_lstm_units', min_value=16, max_value=64, step=4), return_sequences=True))

        # Fully connected layer
        model.add(TimeDistributed(Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=8), activation='relu')))
        model.add(Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.05)))

        # Output layer
        model.add(TimeDistributed(Dense(units=2, activation='linear')))  # Replace user_options.feature_cols with actual feature columns

        model.add(Lambda(lambda x: x[:, -5:, :]))

        # Add learning rate to the hyperparameters
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mean_squared_error')

        return model
    return build_model


if __name__ == '__main__':
    # Load and clean data
    df = load_and_clean_data('HistoricalData_1692981828643_GME_NASDAQ.csv', ['Close/Last', 'Volume'])
    model_data = process_for_model(df)

    tuner = BayesianOptimization(
        build_model_function(model_data),
        objective='val_loss',
        max_trials=100,
        num_initial_points=30,  # Number of randomly chosen hyperparameter sets to start off the search.
        directory='my_dir',
        project_name='bayesian_tuning3'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(model_data.x_train, model_data.y_train, epochs=10,
                 validation_data=(model_data.x_test, model_data.y_test),
                 callbacks=[early_stopping])
