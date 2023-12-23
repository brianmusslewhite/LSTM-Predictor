import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from prophet import Prophet


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

        # Feature Engineering
        df['Close/Last_pct_change'] = df['Close/Last'].pct_change()  # Percentage change
        df['Volume_pct_change'] = df['Volume'].pct_change()

        # Rolling Window Features (e.g., 7-day rolling mean)
        df['Close/Last_rolling_mean'] = df['Close/Last'].rolling(window=7).mean()

        # Dropping NaNs created by pct_change and rolling features
        df.dropna(inplace=True)

        self.processed_data = df

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.processed_data


def plot_forecasts(historical_dates, historical_data, test_dates, test_forecasts, future_dates, future_forecasts):
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


def main():
    stock_data = StockData('HistoricalData_1692981828643_GME_NASDAQ.csv')
    stock_data.load_data()
    stock_data.preprocess_data()
    data = stock_data.get_processed_data()

    close_prices = data['Close/Last']
    close_prices_dates = data.index  # Already a DatetimeIndex

    # Split the data into training and test sets
    split_ratio = 0.8
    split_index = int(len(close_prices) * split_ratio)
    train, test = close_prices[:split_index], close_prices[split_index:]
    train_dates, test_dates = close_prices_dates[:split_index], close_prices_dates[split_index:]

    # Generate future dates for forecasting
    future_dates = pd.date_range(start=close_prices_dates[-1], periods=31, freq='B')[1:]

    # Fit models on training data and forecast
    test_forecasts = {}
    future_forecasts = {}

    # ARIMA Model
    arima_order = auto_arima(train, seasonal=False, stepwise=True).order
    arima_model = ARIMA(train, order=arima_order).fit()
    test_forecasts['ARIMA'] = arima_model.predict(start=split_index, end=len(close_prices) - 1)
    future_forecasts['ARIMA'] = arima_model.forecast(steps=30)

    # SES Model
    ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
    test_forecasts['SES'] = ses_model.predict(start=split_index, end=len(close_prices) - 1)
    future_forecasts['SES'] = ses_model.forecast(30)

    # Holt's Model
    holt_model = Holt(train).fit(smoothing_level=0.2, smoothing_trend=0.1)
    test_forecasts['Holt'] = holt_model.predict(start=split_index, end=len(close_prices) - 1)
    future_forecasts['Holt'] = holt_model.forecast(30)

    # Prophet Model
    prophet_data = data.reset_index()  # The 'Date' column is now in 'ds' format suitable for Prophet
    prophet_data.rename(columns={'Date': 'ds', 'Close/Last': 'y'}, inplace=True)
    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=30, freq='B')
    prophet_forecast = prophet_model.predict(future)
    test_forecasts['Prophet'] = prophet_forecast['yhat'][split_index:len(close_prices)]
    future_forecasts['Prophet'] = prophet_forecast['yhat'][-30:]

    # Call the plotting function
    plot_forecasts(close_prices_dates, close_prices, test_dates, test_forecasts, future_dates, future_forecasts)


if __name__ == "__main__":
    main()
