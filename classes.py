class User_Options:
    def __init__(self, file_path, feature_cols, days_to_predict, perform_optimization, use_early_stopping,
                 d_scaling_method, d_sequence_length, d_epochs, d_train_size_ratio, d_lstm_units,
                 d_dropout_rate, d_batch_size, d_optimizer):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.days_to_predict = days_to_predict
        self.perform_optimization = perform_optimization
        self.use_early_stopping = use_early_stopping
        self.d_train_size_ratio = d_train_size_ratio
        self.d_scaling_method = d_scaling_method
        self.d_sequence_length = d_sequence_length
        self.d_epochs = d_epochs
        self.d_lstm_units = d_lstm_units
        self.d_dropout_rate = d_dropout_rate
        self.d_batch_size = d_batch_size
        self.d_optimizer = d_optimizer


class Optimization_Options:
    def __init__(self, scaling_method_options, sequence_length_options, epochs_options, train_size_ratio_options,
                 lstm_units_options, dropout_rate_options, batch_size_options, optimizer_options):
        self.scaling_method_options = scaling_method_options
        self.sequence_length_options = sequence_length_options
        self.epochs_options = epochs_options
        self.train_size_ratio_options = train_size_ratio_options
        self.lstm_units_options = lstm_units_options
        self.dropout_rate_options = dropout_rate_options
        self.batch_size_options = batch_size_options
        self.optimizer_options = optimizer_options