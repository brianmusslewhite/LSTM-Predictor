from keras_tuner import BayesianOptimization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization, Bidirectional, Lambda, Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_and_clean_data, process_for_model


def build_cnn_lstm_model_function(model_data):
    def build_cnn_lstm_model(hp):
        model = Sequential()

        # Convolutional Layer
        model.add(Conv1D(filters=hp.Int('conv1d_filters', min_value=64, max_value=256, step=64),
                         kernel_size=hp.Int('conv1d_kernel_size', min_value=2, max_value=5, step=2),
                         activation='relu',
                         input_shape=(model_data.x_train.shape[1], model_data.x_train.shape[2])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=hp.Int('maxpooling1d_pool_size', min_value=2, max_value=4, step=2)))
        model.add(Dropout(hp.Float('conv_dropout', min_value=0, max_value=0.8, step=0.2)))

        # First Bidirectional LSTM Layer
        model.add(Bidirectional(LSTM(units=hp.Int('first_lstm_units', min_value=128, max_value=2048, step=256),
                                     return_sequences=True,
                                     recurrent_regularizer=l1_l2(l1=hp.Float('first_lstm_l1', min_value=1e-5, max_value=1e-1, sampling='LOG'),
                                                                 l2=hp.Float('first_lstm_l2', min_value=1e-5, max_value=1e-1, sampling='LOG')))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('first_lstm_dropout', min_value=0, max_value=0.8, step=0.2)))

        # Second and Third LSTM Layers
        model.add(LSTM(units=hp.Int('second_lstm_units', min_value=64, max_value=1024, step=128), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('second_lstm_dropout', min_value=0, max_value=0.8, step=0.2)))

        model.add(LSTM(units=hp.Int('third_lstm_units', min_value=64, max_value=1024, step=128), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('third_lstm_dropout', min_value=0, max_value=0.8, step=0.2)))

        # Fully Connected Layers
        model.add(TimeDistributed(Dense(units=hp.Int('first_dense_units', min_value=64, max_value=2048, step=256), activation='relu')))
        model.add(Dropout(hp.Float('first_dense_dropout', min_value=0, max_value=0.6, step=0.1)))

        model.add(TimeDistributed(Dense(units=hp.Int('second_dense_units', min_value=32, max_value=2048, step=256), activation='relu')))
        model.add(Dropout(hp.Float('second_dense_dropout', min_value=0, max_value=0.6, step=0.1)))

        # Output Layer
        model.add(TimeDistributed(Dense(units=2, activation='linear')))

        model.add(Lambda(lambda x: x[:, -3:, :]))

        # Add learning rate to the hyperparameters
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='mean_squared_error')

        return model
    return build_cnn_lstm_model


def build_model_function(model_data):
    def build_model(hp):
        model = Sequential()

        # First BiLSTM layer with BatchNormalization and Dropout
        model.add(Bidirectional(LSTM(units=hp.Int('first_lstm_units', min_value=512, max_value=2056, step=64),
                                     return_sequences=True,
                                     recurrent_regularizer=l1_l2(l1=hp.Float('first_lstm_l1', min_value=1e-6, max_value=1e-2, sampling='LOG'), 
                                                                 l2=hp.Float('first_lstm_l2', min_value=1e-6, max_value=1e-2, sampling='LOG'))),
                                input_shape=((model_data.x_train.shape[1], model_data.x_train.shape[2]))))  # Replace your_input_shape with actual shape
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.8, step=0.05)))

        # Second LSTM layer
        model.add(LSTM(units=hp.Int('second_lstm_units', min_value=32, max_value=512, step=16), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.8, step=0.05)))

        # Third LSTM layer
        model.add(LSTM(units=hp.Int('third_lstm_units', min_value=32, max_value=512, step=16), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.8, step=0.05)))

        # Fourth LSTM layer
        model.add(LSTM(units=hp.Int('fourth_lstm_units', min_value=16, max_value=512, step=16), return_sequences=True))

        # Fully connected layer
        model.add(TimeDistributed(Dense(units=hp.Int('dense_units', min_value=64, max_value=1024, step=32), activation='relu')))
        model.add(Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.8, step=0.05)))

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
        build_cnn_lstm_model_function(model_data),
        objective='val_loss',
        max_trials=1000,
        num_initial_points=100,  # Number of randomly chosen hyperparameter sets to start off the search.
        directory='my_dir',
        project_name='bayesian_tuning_cnnlstm_coarse2'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(model_data.x_train, model_data.y_train, epochs=30,
                 validation_data=(model_data.x_test, model_data.y_test),
                 callbacks=[early_stopping])
