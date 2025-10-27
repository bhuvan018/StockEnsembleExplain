import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class StockDataFetcher:
    def __init__(self, ticker, start_date=None, end_date=None, lookback_days=365):
        self.ticker = ticker
        self.lookback_days = lookback_days
        
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=lookback_days)
        else:
            self.start_date = pd.to_datetime(start_date)
            
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        try:
            stock = yf.Ticker(self.ticker)
            self.raw_data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.raw_data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            return self.raw_data
        except Exception as e:
            raise Exception(f"Error fetching data for {self.ticker}: {str(e)}")
    
    def calculate_technical_indicators(self):
        if self.raw_data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        df = self.raw_data.copy()
        
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        
        rsi = ta.rsi(df['Close'], length=14)
        df['RSI'] = rsi
        
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        
        bbands = ta.bbands(df['Close'], length=20)
        if bbands is not None:
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['BB_middle'] = bbands['BBM_20_2.0']
            df['BB_lower'] = bbands['BBL_20_2.0']
        
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR'] = atr
        
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
        
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        df['Price_Change'] = df['Close'] - df['Close'].shift(1)
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['Target_Price'] = df['Close'].shift(-1)
        df['Target_Return'] = df['Returns'].shift(-1)
        
        df.dropna(inplace=True)
        
        self.processed_data = df
        return df
    
    def prepare_features(self, exclude_cols=None):
        if self.processed_data is None:
            raise ValueError("No processed data available. Call calculate_technical_indicators() first.")
        
        if exclude_cols is None:
            exclude_cols = ['Target', 'Target_Price', 'Target_Return', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        X = self.processed_data[feature_cols].copy()
        y_classification = self.processed_data['Target'].copy()
        y_regression = self.processed_data['Target_Price'].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X.fillna(method='ffill', inplace=True)
        X.fillna(0, inplace=True)
        
        return X, y_classification, y_regression, feature_cols
    
    def get_train_test_split(self, test_size=0.2):
        X, y_class, y_reg, feature_names = self.prepare_features()
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train_class = y_class.iloc[:split_idx]
        y_test_class = y_class.iloc[split_idx:]
        y_train_reg = y_reg.iloc[:split_idx]
        y_test_reg = y_reg.iloc[split_idx:]
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_class': y_train_class,
            'y_test_class': y_test_class,
            'y_train_reg': y_train_reg,
            'y_test_reg': y_test_reg,
            'feature_names': feature_names,
            'dates_train': self.processed_data.index[:split_idx],
            'dates_test': self.processed_data.index[split_idx:],
            'prices_train': self.processed_data['Close'].iloc[:split_idx],
            'prices_test': self.processed_data['Close'].iloc[split_idx:]
        }
    
    def get_data_summary(self):
        if self.processed_data is None:
            return None
            
        summary = {
            'ticker': self.ticker,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'total_records': len(self.processed_data),
            'features_count': len(self.processed_data.columns),
            'date_range': f"{self.processed_data.index[0].strftime('%Y-%m-%d')} to {self.processed_data.index[-1].strftime('%Y-%m-%d')}",
            'price_range': f"${self.processed_data['Close'].min():.2f} - ${self.processed_data['Close'].max():.2f}",
            'avg_volume': f"{self.processed_data['Volume'].mean():.0f}",
        }
        return summary

def get_multiple_stocks_data(tickers, start_date=None, end_date=None, lookback_days=365):
    all_data = {}
    
    for ticker in tickers:
        try:
            fetcher = StockDataFetcher(ticker, start_date, end_date, lookback_days)
            fetcher.fetch_data()
            fetcher.calculate_technical_indicators()
            all_data[ticker] = fetcher
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    return all_data
