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

def plot_trading_results(env):
    data = env.data
    positions = env.positions

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data['Close'], label='Close Price')
    for position in positions:
        if position[1] == 'long':
            ax.plot(position[0], data['Close'].iloc[position[0]], '^', markersize=10, color='g', label='Long')
        elif position[1] == 'short':
            ax.plot(position[0], data['Close'].iloc[position[0]], 'v', markersize=10, color='r', label='Short')

    plt.title('Trading Positions')
    plt.legend()
    plt.show()

    cumulative_rewards = np.cumsum([reward for _, reward in env.positions])
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Reward')
    plt.show()
