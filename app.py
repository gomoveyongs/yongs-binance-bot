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

@app.route('/result', methods=['GET'])
def read_reviews():
    strg = request.args.get('strategy')
    k = request.args.get('k')
    ticker = request.args.get('ticker') 

    print(strg)

    # 바이낸스 접속 불가..
    #backtest_result = get_binance_backtest_data(strg, k, ticker)
    backtest_result = {'ROR': 1.102132, 'MDD' : 36}

    result_list = []
    result_list.append(backtest_result)

    return jsonify({'results': result_list})


def get_binance_backtest_data(strg, k, ticker):
    ohlcvs = binance.fetch_ohlcv(ticker, '1d')
    df = pd.DataFrame(ohlcvs, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)

    df = get_binance_target_price(strg, df, k)
    df = get_binance_ror_hpr_dd(df)

    # 누적수익률
    ror = df['ror'].cumprod()[-2]
    mdd = df['dd'].max()

    result = {'ROR': ror, 'MDD' : mdd}
    
    # Save Backtesting Result to file
    df.to_excel('eth.xlsx')
    return result

def get_binance_target_price(strg, df, k=0.5):
    if strg == 'volatility-breakout':
        df = get_binance_target_price_volatility_breakout(df, k)

    return df

def get_binance_ror_hpr_dd(df):
    df['ror'] = np.where(df['high'] > df['target'], 
                    df['close'] / df['target'],
                    1)
    
    df['hpr'] = df['ror'].cumprod()
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
    return df

def get_binance_target_price_volatility_breakout(df, k):
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)
    return df
    

# render_template: template 파일 중 index.html을 불러 옴 
if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)