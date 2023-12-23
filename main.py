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
    stock_data.split_data()
    stock_data.generate_future_dates()

    # Fit models on training data and forecast
    test_forecasts = {}
    future_forecasts = {}

    # ARIMA Model
    arima_order = auto_arima(stock_data.train, seasonal=False, stepwise=True).order
    arima_model = ARIMA(stock_data.train, order=arima_order).fit()
    test_forecasts['ARIMA'] = arima_model.predict(start=stock_data.split_index, end=len(stock_data.close_prices) - 1)
    future_forecasts['ARIMA'] = arima_model.forecast(steps=30)

    # SES Model
    ses_model = SimpleExpSmoothing(stock_data.train).fit(smoothing_level=0.2)
    test_forecasts['SES'] = ses_model.predict(start=stock_data.split_index, end=len(stock_data.close_prices) - 1)
    future_forecasts['SES'] = ses_model.forecast(30)

    # Holt's Model
    holt_model = Holt(stock_data.train).fit(smoothing_level=0.2, smoothing_trend=0.1)
    test_forecasts['Holt'] = holt_model.predict(start=stock_data.split_index, end=len(stock_data.close_prices) - 1)
    future_forecasts['Holt'] = holt_model.forecast(30)

    # Prophet Model
    prophet_data = stock_data.processed_data.reset_index()
    prophet_data['Date'] = prophet_data['Date'].dt.to_timestamp()  # Convert PeriodIndex to Timestamp
    prophet_data.rename(columns={'Date': 'ds', 'Close/Last': 'y'}, inplace=True)
    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=30, freq='B')
    prophet_forecast = prophet_model.predict(future)
    test_forecasts['Prophet'] = prophet_forecast['yhat'][stock_data.split_index:len(stock_data.close_prices)]
    future_forecasts['Prophet'] = prophet_forecast['yhat'][-30:]

    # Call the plotting function
    plot_forecasts(stock_data.close_prices_dates, stock_data.close_prices, stock_data.test_dates, test_forecasts, stock_data.future_dates, future_forecasts)


if __name__ == "__main__":
    main()