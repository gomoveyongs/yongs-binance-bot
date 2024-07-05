import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    scaler = StandardScaler()
    data['RSI'] = compute_RSI(data['Close'])
    data['MFI'] = compute_MFI(data)
    data['Ichimoku'] = data['Close'].shift(-26)
    data['Close_26'] = data['Close'].shift(26)
    data['MACD'] = compute_MACD(data['Close'])
    data = data.dropna()
    normalized_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MFI', 'Ichimoku', 'Close_26', 'MACD']])
    return normalized_data

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def compute_MFI(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = (money_flow.where(typical_price > typical_price.shift(1), 0)).rolling(window=period).sum()
    negative_flow = (money_flow.where(typical_price < typical_price.shift(1), 0)).rolling(window=period).sum()
    MFI = 100 - (100 / (1 + (positive_flow / negative_flow)))
    return MFI

def compute_MACD(series, slow=26, fast=12, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal

def plot_trading_results(env, data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'].values, label='Close Price')
    
    long_positions = [i for i, x in enumerate(env.position_history) if x == 1]
    short_positions = [i for i, x in enumerate(env.position_history) if x == -1]
    
    plt.scatter(long_positions, data['Close'].values[long_positions], marker='^', color='g', label='Long')
    plt.scatter(short_positions, data['Close'].values[short_positions], marker='v', color='r', label='Short')
    
    plt.title('Trading Strategy Results')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot cumulative returns
    plt.figure(figsize=(14, 7))
    plt.plot(np.cumsum(env.reward_history), label='Cumulative Return')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()
