import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


class StockData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        self.raw_data = pd.read_csv(self.filepath)

    def preprocess_data(self):
        df = self.raw_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Replace $ and convert to float
        financial_cols = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
        for col in financial_cols:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

        # Handling missing values
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill if any NaNs remain

        # Create a complete date range
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(full_date_range, method='ffill')  # Reindex and forward fill missing dates

        # Feature Engineering
        df['Close/Last_pct_change'] = df['Close/Last'].pct_change()  # Percentage change
        df['Volume_pct_change'] = df['Volume'].pct_change()

        # Rolling Window Features (e.g., 7-day rolling mean)
        df['Close/Last_rolling_mean'] = df['Close/Last'].rolling(window=7).mean()

        # Dropping NaNs created by pct_change and rolling features
        df.dropna(inplace=True)

        self.processed_data = df

    def apply_differencing(self, column_name='Close/Last', period=1):
        if column_name not in self.processed_data.columns:
            raise ValueError(f"Column '{column_name}' not found in the data.")

        # Applying differencing
        self.processed_data[f'{column_name}_diff'] = self.processed_data[column_name].diff(periods=period)

        # Dropping NaN values that result from differencing
        self.processed_data.dropna(inplace=True)

    def apply_log_transformation(self, column_name='Close/Last'):
        """
        Apply log transformation to a specified column.
        """
        self.processed_data[f'{column_name}_log'] = np.log(self.processed_data[column_name])

    def apply_seasonal_decomposition(self, column_name='Close/Last', model='additive', period=5):
        """
        Apply seasonal decomposition to a specified column.
        """
        decomposition = seasonal_decompose(self.processed_data[column_name], model=model, period=period)
        self.processed_data[f'{column_name}_trend'] = decomposition.trend
        self.processed_data[f'{column_name}_seasonal'] = decomposition.seasonal
        self.processed_data[f'{column_name}_resid'] = decomposition.resid

    def add_time_features(self):
        """
        Add additional time features to the data.
        """
        self.processed_data['day_of_week'] = self.processed_data.index.dayofweek
        self.processed_data['month'] = self.processed_data.index.month
        self.processed_data['quarter'] = self.processed_data.index.quarter

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.processed_data


class ProphetModel:
    def __init__(self, params=None, perform_optimization=False):
        if params is None:
            params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            }
        self.params = params
        self.perform_optimization = perform_optimization
        self.model = None

    def optimize(self, training_data, validation_data):
        def objective(params):
            model = Prophet(**params, daily_seasonality=False)
            model.fit(training_data)
            future = model.make_future_dataframe(periods=len(validation_data), freq='B')
            forecast = model.predict(future)
            mse = mean_squared_error(validation_data['y'], forecast['yhat'][-len(validation_data):])
            return {'loss': mse, 'status': STATUS_OK}

        space = {
            'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.01, 0.5),
            'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 1, 10),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        self.params = {
            'changepoint_prior_scale': best['changepoint_prior_scale'],
            'seasonality_prior_scale': best['seasonality_prior_scale'],
            'seasonality_mode': ['additive', 'multiplicative'][best['seasonality_mode']]
        }

    def fit(self, training_data):
        if self.perform_optimization:
            self.optimize(training_data)
        self.model = Prophet(**self.params, daily_seasonality=False, optimized=False)
        self.model.fit(training_data)

    def predict(self, future_periods):
        future = self.model.make_future_dataframe(periods=future_periods, freq='B')
        forecast = self.model.predict(future)
        return forecast


class ARIMAModel:
    def __init__(self):
        self.model = None

    def fit(self, training_data):
        arima_order = auto_arima(training_data['y'], seasonal=False, stepwise=True).order
        self.model = ARIMA(training_data['y'], order=arima_order).fit()

    def predict(self, start, end):
        return self.model.predict(start=start, end=end)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)


class SESModel:
    def __init__(self):
        self.model = None

    def fit(self, training_data, smoothing_level=0.2):
        self.model = SimpleExpSmoothing(training_data['y']).fit(smoothing_level=smoothing_level)

    def predict(self, start, end):
        return self.model.predict(start=start, end=end)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)


class HoltModel:
    def __init__(self):
        self.model = None

    def fit(self, training_data, smoothing_level=0.2, smoothing_trend=0.1):
        self.model = Holt(training_data['y']).fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)

    def predict(self, start, end):
        return self.model.predict(start=start, end=end)

    def forecast(self, steps):
        return self.model.forecast(steps=steps)


def plot_forecasts(test_dates, test_forecasts, future_dates, future_forecast, model_name):
    # Define colors for each model
    model_colors = {
        'ARIMA': 'red',
        'SES': 'blue',
        'Holt': 'green',
        'Prophet': 'purple'
    }

    color = model_colors.get(model_name, 'black')

    # Plot Test Predictions for each model
    plt.plot(test_dates, test_forecasts[model_name], label=f'{model_name} Test Prediction', color=color, linestyle='--')

    # Check if future_forecast is an iterable
    if isinstance(future_forecast, (list, np.ndarray, pd.Series)):
        if len(future_dates) != len(future_forecast):
            print(f"Length mismatch: future_dates ({len(future_dates)}) vs future_forecast ({len(future_forecast)})")
            future_forecast = future_forecast[:len(future_dates)]
        plt.plot(future_dates, future_forecast, label=f'{model_name} Future Forecast', color=color)
    else:
        print(f"Invalid type for future_forecast: {type(future_forecast)}")


def main(perform_optimization=True):
    # Initialize and preprocess stock data
    stock_data = StockData('HistoricalData_1692981828643_GME_NASDAQ.csv')
    stock_data.load_data()
    stock_data.preprocess_data()

    # Apply transformations
    #  stock_data.apply_differencing(column_name='Close/Last', period=1)
    #  stock_data.apply_log_transformation(column_name='Close/Last')
    #  stock_data.apply_seasonal_decomposition(column_name='Close/Last', model='additive', period=5)
    #  stock_data.add_time_features()

    data = stock_data.get_processed_data()
    data = data.sort_index()  # Sort the data by date

    close_prices = data['Close/Last']
    close_prices_dates = data.index

    # Split the data into training and validation sets
    split_ratio = 0.8
    split_index = int(len(close_prices) * split_ratio)
    global training_data, validation_data
    training_data = data[:split_index].reset_index().rename(columns={'index': 'ds', 'Close/Last': 'y'})
    validation_data = data[split_index:].reset_index().rename(columns={'index': 'ds', 'Close/Last': 'y'})

    # Create model instances and fit them
    prophet_model = ProphetModel(perform_optimization=perform_optimization)
    prophet_model.fit(training_data)
    arima_model = ARIMAModel()
    arima_model.fit(training_data)
    ses_model = SESModel()
    ses_model.fit(training_data)
    holt_model = HoltModel()
    holt_model.fit(training_data)

    # Predict and forecast using each model
    prophet_forecast = prophet_model.predict(len(validation_data))
    arima_forecast = arima_model.forecast(steps=30)
    ses_forecast = ses_model.forecast(steps=30)
    holt_forecast = holt_model.forecast(steps=30)

    future_dates = pd.date_range(start=close_prices_dates[-1] + pd.Timedelta(days=1), periods=30, freq='B')

    # Prepare forecast data for plotting
    test_forecasts = {
        'Prophet': prophet_forecast.set_index('ds')['yhat'].loc[validation_data['ds']],
        'ARIMA': arima_model.predict(start=split_index, end=len(close_prices) - 1),
        'SES': ses_model.predict(start=split_index, end=len(close_prices) - 1),
        'Holt': holt_model.predict(start=split_index, end=len(close_prices) - 1)
    }

    future_forecasts = {
        'Prophet': prophet_forecast.set_index('ds')['yhat'][-len(validation_data):],
        'ARIMA': arima_forecast,
        'SES': ses_forecast,
        'Holt': holt_forecast
    }

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.plot(close_prices_dates, close_prices, label='Historical Data', color='grey', alpha=0.5)

    for model_name in ['Prophet', 'ARIMA', 'SES', 'Holt']:
        plot_forecasts(validation_data['ds'], test_forecasts, future_dates, future_forecasts[model_name], model_name)

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Forecasts')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(perform_optimization=False)
