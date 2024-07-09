import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocess_data(data):
    data['RSI'] = compute_rsi(data['Close'])
    data['MFI'] = compute_mfi(data)
    data['Ichimoku'] = data['Close'].shift(-26)
    data['Prev_Close'] = data['Close'].shift(26)
    data['MACD'] = compute_macd(data['Close'])

    data = data.dropna()
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])
    return data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    return mfi

def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
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
    plt.savefig('trading_positions.png')

    cumulative_rewards = np.cumsum([reward for _, reward in env.positions])
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Reward')
    plt.savefig('cmulative_reward.png')
