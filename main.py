import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


class StockData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        self.close_prices = None
        self.close_prices_dates = None
        self.train = None
        self.test = None
        self.train_dates = None
        self.test_dates = None
        self.future_dates = None
        self.split_index = None

    def load_data(self):
        self.raw_data = pd.read_csv(self.filepath)

    def preprocess_data(self):
        df = self.raw_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        financial_cols = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
        for col in financial_cols:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
        df = df.sort_values(by='Date')
        if df.isnull().sum().sum() > 0:
            df = df.fillna(method='ffill')

        # Set 'Date' as the time index and set frequency
        df.set_index('Date', inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('B')  # 'B' for business day frequency

        self.close_prices = df['Close/Last']
        self.close_prices_dates = self.close_prices.index.to_timestamp()
        self.processed_data = df

    def split_data(self, split_ratio=0.8):
        self.split_index = int(len(self.close_prices) * split_ratio)
        self.train, self.test = self.close_prices[:self.split_index], self.close_prices[self.split_index:]
        self.train_dates, self.test_dates = self.close_prices_dates[:self.split_index], self.close_prices_dates[self.split_index:]

    def generate_future_dates(self, periods=31):
        if self.close_prices_dates is not None:
            self.future_dates = pd.date_range(start=self.close_prices_dates[-1], periods=periods, freq='B')[1:]
        else:
            print("Close prices dates not set. Please run preprocess_data first.")

    def add_time_features(self):
        self.processed_data['DayOfWeek'] = self.processed_data.index.dayofweek
        self.processed_data['Month'] = self.processed_data.index.month
        self.processed_data['Quarter'] = self.processed_data.index.quarter
        self.processed_data['Year'] = self.processed_data.index.year

    def add_lag_features(self, lags=5):
        for lag in range(1, lags + 1):
            self.processed_data[f'lag_{lag}'] = self.close_prices.shift(lag)

    def add_rolling_window_features(self, window_sizes=[7, 30]):
        for window in window_sizes:
            self.processed_data[f'rolling_mean_{window}'] = self.close_prices.rolling(window=window).mean()
            self.processed_data[f'rolling_std_{window}'] = self.close_prices.rolling(window=window).std()

    def scale_features(self):
        scaler = MinMaxScaler()
        self.processed_data = pd.DataFrame(scaler.fit_transform(self.processed_data), columns=self.processed_data.columns, index=self.processed_data.index)


class ARIMAModel:
    def __init__(self, stock_data):
        self.train_data = stock_data.train
        self.test_data = stock_data.test
        self.split_index = stock_data.split_index
        self.close_prices = stock_data.close_prices
        self.model = None

    def fit(self):
        arima_order = auto_arima(self.train_data, seasonal=False, stepwise=True).order
        self.model = ARIMA(self.train_data, order=arima_order).fit()

    def predict_test(self):
        return self.model.predict(start=self.split_index, end=len(self.close_prices) - 1)

    def predict_future(self, steps=30):
        return self.model.forecast(steps=steps)

    def objective(self, p, d, q):
        order = (int(p), int(d), int(q))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = ARIMA(self.train_data, order=order)
                # Updated fit method call
                fitted_model = model.fit()
            predictions = fitted_model.predict(start=self.split_index, end=len(self.close_prices) - 1)
            mse = np.mean((self.test_data - predictions)**2)
        except Exception as e:
            print(f"Error with order {order}: {e}")
            return 1e6
        return -mse

    def optimize_arima(self):
        pbounds = {'p': (0, 5), 'd': (0, 2), 'q': (0, 5)}

        optimizer = BayesianOptimization(
            f=partial(self.objective),  # Use functools.partial to bind 'self' to the objective function
            pbounds=pbounds,
            random_state=1,
        )

        optimizer.maximize(init_points=50, n_iter=500)

        best_params = optimizer.max['params']
        self.model = ARIMA(self.train_data, order=(int(best_params['p']), int(best_params['d']), int(best_params['q']))).fit()


class SESModel:
    def __init__(self, stock_data):
        self.train_data = stock_data.train
        self.split_index = stock_data.split_index
        self.close_prices = stock_data.close_prices
        self.model = None

    def fit(self, smoothing_level=0.2):
        self.model = SimpleExpSmoothing(self.train_data).fit(smoothing_level=smoothing_level)

    def predict_test(self):
        return self.model.predict(start=self.split_index, end=len(self.close_prices) - 1)

    def predict_future(self, steps=30):
        return self.model.forecast(steps=steps)


class HoltModel:
    def __init__(self, stock_data):
        self.train_data = stock_data.train
        self.split_index = stock_data.split_index
        self.close_prices = stock_data.close_prices
        self.model = None

    def fit(self, smoothing_level=0.2, smoothing_trend=0.1):
        self.model = Holt(self.train_data).fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)

    def predict_test(self):
        return self.model.predict(start=self.split_index, end=len(self.close_prices) - 1)

    def predict_future(self, steps=30):
        return self.model.forecast(steps=steps)


class ProphetModel:
    def __init__(self, train_data, split_index, close_prices, prophet_data, yearly_seasonality=True, daily_seasonality=True):
        self.train_data = train_data
        self.split_index = split_index
        self.close_prices = close_prices
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.prophet_data = prophet_data

    def fit(self):
        self.prophet_data['Date'] = self.prophet_data['Date'].dt.to_timestamp()  # Convert PeriodIndex to Timestamp
        self.prophet_data.rename(columns={'Date': 'ds', 'Close/Last': 'y'}, inplace=True)
        self.model = Prophet(yearly_seasonality=self.yearly_seasonality, daily_seasonality=self.daily_seasonality)
        self.model.fit(self.prophet_data)

    def get_forecast(self, steps=30):
        future = self.model.make_future_dataframe(steps, freq='B')
        prophet_forecast = self.model.predict(future)
        return prophet_forecast


class RandomForestModel:
    def __init__(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target
        self.model = RandomForestRegressor()

    def fit(self):
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.train_features_imputed = imputer.fit_transform(self.train_features)

        self.model.fit(self.train_features_imputed, self.train_target)

    def predict(self, features):
        # Apply the same imputation to the test features
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)
        return self.model.predict(features_imputed)

    def get_feature_importance(self):
        return self.model.feature_importances_


def plot_data(historical_dates, historical_data, test_dates, test_forecasts, future_dates, future_forecasts):
    plt.figure(figsize=(15, 10))

    # Define colors for each model
    model_colors = {
        'ARIMA': 'red',
        'SES': 'blue',
        'Holt': 'green',
        'Prophet': 'purple'
    }

    # Plot Historical Data
    plt.plot(historical_dates, historical_data, label='Historical Data', color='grey', alpha=0.5)

    # Plot Test Predictions for each model
    for model_name, test_forecast in test_forecasts.items():
        color = model_colors.get(model_name, 'black')  # Default to black if model name not found
        plt.plot(test_dates, test_forecast, label=f'{model_name} Test Prediction', color=color, linestyle='--')

    # Plot Future Forecasts for each model
    for model_name, future_forecast in future_forecasts.items():
        color = model_colors.get(model_name, 'black')  # Default to black if model name not found
        plt.plot(future_dates, future_forecast, label=f'{model_name} Future Forecast', color=color)

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Forecasts')
    plt.legend()
    plt.show()


def plot_feature_importances(feature_importances, feature_names):
    # Sort the feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_importances)), feature_importances[indices],
            color="r", align="center")
    plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.show()


def main():
    forecast_length = 365

    stock_data = StockData('HistoricalData_1692981828643_GME_NASDAQ.csv')
    stock_data.load_data()
    stock_data.preprocess_data()
    stock_data.add_time_features()         # Add time-related features
    stock_data.add_lag_features(lags=5)    # Add lag features
    stock_data.add_rolling_window_features(window_sizes=[7, 30])  # Add rolling window features
    stock_data.split_data()
    stock_data.generate_future_dates(periods=forecast_length + 1)

    if False:
        # Prepare features and target for RandomForest
        features = stock_data.processed_data.drop('Close/Last', axis=1)
        target = stock_data.close_prices
        train_features = features[:stock_data.split_index]
        train_target = target[:stock_data.split_index]

        # Initialize and fit RandomForest model
        rf_model = RandomForestModel(train_features, train_target)
        rf_model.fit()

        feature_importances = rf_model.get_feature_importance()
        plot_feature_importances(feature_importances, train_features.columns)

    # Fit models on training data and forecast
    test_forecasts = {}
    future_forecasts = {}

    # ARIMA Model (as before)
    arima_model = ARIMAModel(stock_data)
    # arima_model.optimize_arima()
    arima_model.fit()
    test_forecasts['ARIMA'] = arima_model.predict_test()
    future_forecasts['ARIMA'] = arima_model.predict_future(steps=forecast_length)

    # SES Model
    ses_model = SESModel(stock_data)
    ses_model.fit(smoothing_level=0.2)
    test_forecasts['SES'] = ses_model.predict_test()
    future_forecasts['SES'] = ses_model.predict_future(steps=forecast_length)

    # Holt's Model
    holt_model = HoltModel(stock_data)
    holt_model.fit(smoothing_level=0.2, smoothing_trend=0.1)
    test_forecasts['Holt'] = holt_model.predict_test()
    future_forecasts['Holt'] = holt_model.predict_future(steps=forecast_length)

    # Prophet Model
    prophet_data = stock_data.processed_data.reset_index()
    prophet_model = ProphetModel(stock_data.train, stock_data.split_index, stock_data.close_prices, prophet_data)
    prophet_model.fit()
    prophet_forecast = prophet_model.get_forecast(steps=forecast_length)

    test_forecasts['Prophet'] = prophet_forecast['yhat'][stock_data.split_index:len(stock_data.close_prices)]
    future_forecasts['Prophet'] = prophet_forecast['yhat'][-forecast_length:]

    # Call the plotting function
    plot_data(stock_data.close_prices_dates, stock_data.close_prices, stock_data.test_dates, test_forecasts,
              stock_data.future_dates, future_forecasts)


if __name__ == "__main__":
    main()
