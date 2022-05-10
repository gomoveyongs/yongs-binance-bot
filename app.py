from flask import Flask, render_template, jsonify, request

import time
import pandas as pd
import ccxt
import numpy as np

binance = ccxt.binance()
app = Flask(__name__)

@app.route('/')
def home():   
   # render_template: template 파일을 읽어오기 위한 것
   return render_template('index.html')

## API 역할을 하는 부분
@app.route('/review', methods=['POST'])
def write_review():
    sample_receive = request.form['sample_give']
    print(sample_receive)
    return jsonify({'msg': '이 요청은 POST!'})

@app.route('/review', methods=['GET'])
def read_reviews():
    sample_receive = request.args.get('sample_give')
    print(sample_receive)

    df = get_binance_backtest_data()

    return jsonify({'msg': '이 요청은 GET!'})

def get_binance_backtest_data(strg):
    ohlcvs = binance.fetch_ohlcv('ETH/BTC', '1d')
    df = pd.DataFrame(ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)

    df = get_binance_target_price(strg, df)
    df = get_binance_ror(df)

    # 누적수익률
    ror = df['ror'].cumprod()[-2]
    
    # Save Backtesting Result to file
    df.to_excel('eth.xlsx')
    return ror

def get_binance_target_price(strg, df):
    if strg == 'volatility-breakout':
        df = get_binance_target_price_volatility_breakout(df)

    return df

def get_binance_ror(df):
    df['ror'] = np.where(df['high'] > df['target'], 
                    df['close'] / df['target'],
                    1)
    return df

def get_binance_target_price_volatility_breakout(df):
    k = 0.5
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)
    return df
    




# render_template: template 파일 중 index.html을 불러 옴 
if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)