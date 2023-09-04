from keras_tuner import BayesianOptimization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization, Bidirectional, Lambda
from tensorflow.keras.regularizers import l1_l2

from preprocessing import load_and_clean_data, process_for_model


def build_model_function(model_data):
    def build_model(hp):
        model = Sequential()

        # First BiLSTM layer with BatchNormalization and Dropout
        model.add(Bidirectional(LSTM(units=hp.Int('first_lstm_units', min_value=64, max_value=256, step=32),
                                     return_sequences=True,
                                     recurrent_regularizer=l1_l2(l1=hp.Float('first_lstm_l1', min_value=1e-6, max_value=1e-2, sampling='LOG'), 
                                                                 l2=hp.Float('first_lstm_l2', min_value=1e-6, max_value=1e-2, sampling='LOG'))),
                                input_shape=((model_data.x_train.shape[1], model_data.x_train.shape[2]))))  # Replace your_input_shape with actual shape
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

        # Second LSTM layer
        model.add(LSTM(units=hp.Int('second_lstm_units', min_value=32, max_value=128, step=16), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

        # Third LSTM layer
        model.add(LSTM(units=hp.Int('third_lstm_units', min_value=32, max_value=128, step=16), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))

        # Fourth LSTM layer
        model.add(LSTM(units=hp.Int('fourth_lstm_units', min_value=16, max_value=64, step=8), return_sequences=True))

        # Fully connected layer
        model.add(TimeDistributed(Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=16), activation='relu')))
        model.add(Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1)))

        # Output layer
        model.add(TimeDistributed(Dense(units=2, activation='linear')))  # Replace user_options.feature_cols with actual feature columns

        model.add(Lambda(lambda x: x[:, -5:, :]))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model
    return build_model


if __name__ == '__main__':
    # Load and clean data
    df = load_and_clean_data('HistoricalData_1692981828643_GME_NASDAQ.csv', ['Close/Last', 'Volume'])
    model_data = process_for_model(df)

    tuner = BayesianOptimization(
        build_model_function(model_data),
        objective='val_loss',
        max_trials=20,
        num_initial_points=5,  # Number of randomly chosen hyperparameter sets to start off the search.
        directory='my_dir',
        project_name='bayesian_tuning'
    )

    tuner.search(model_data.x_train, model_data.y_train, epochs=10, validation_data=(model_data.x_test, model_data.y_test))  # Replace x_train, y_train, x_val, y_val with your actual data
