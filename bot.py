import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= Secure Configuration (From GitHub Secrets) =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ================= Trading Settings (Edit directly here) =================
SYMBOL = 'BTC/USDT'   
TIMEFRAME = '15m'      # Options: 1m, 5m, 15m, 30m, 1h, 4h, 1d

class SqueezeMomentumIndicator:
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def true_range(self, high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def linear_regression(self, series, length):
        def linreg_single(x):
            if len(x) < length: return np.nan
            y = np.arange(len(x))
            slope, intercept, _, _, _ = stats.linregress(y, x)
            return slope * (len(x) - 1) + intercept
        return series.rolling(window=length).apply(linreg_single, raw=False)

    def calculate_indicators(self, df):
        data = df.copy()
        
        bb_basis = data['close'].rolling(window=self.bb_length).mean()
        bb_dev = self.bb_mult * data['close'].rolling(window=self.bb_length).std()
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev
        
        kc_ma = data['close'].rolling(window=self.kc_length).mean()
        tr = self.true_range(data['high'], data['low'], data['close'])
        range_ma = tr.rolling(window=self.kc_length).mean()
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult
        
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        highest_high = data['high'].rolling(window=self.kc_length).max()
        lowest_low = data['low'].rolling(window=self.kc_length).min()
        close_ma = data['close'].rolling(window=self.kc_length).mean()
        avg_val = ((highest_high + lowest_low) / 2 + close_ma) / 2
        momentum = self.linear_regression(data['close'] - avg_val, self.kc_length)
        
        data['squeeze_on'] = squeeze_on
        data['momentum'] = momentum
        data['momentum_increasing'] = momentum > momentum.shift(1)
        
        return data

    def generate_signals(self, df):
        data = self.calculate_indicators(df)
        data['signal'] = 0
        data['squeeze_release'] = data['squeeze_on'].shift(1) & ~data['squeeze_on']
        
        buy_cond = (data['squeeze_release']) & (data['momentum'] > 0) & (data['momentum_increasing'])
        sell_cond = (
            (data['momentum'] < 0) & (data['momentum'].shift(1) >= 0) |
            ((~data['momentum_increasing']) & (~data['momentum_increasing'].shift(1)) & (data['momentum'] > 0))
        )
        
        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1
        
        return data

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Error: TELEGRAM_TOKEN or CHANNEL_ID is missing!")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHANNEL_ID, 
        'text': message, 
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def get_mexc_data(symbol, timeframe, limit=100):
    exchange = ccxt.mexc({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def main():
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Environment not configured. Please add TELEGRAM_TOKEN and CHANNEL_ID to GitHub Secrets.")
        return

    print(f"[{datetime.now()}] Checking {SYMBOL} on {TIMEFRAME} timeframe via MEXC...")
    
    try:
        df = get_mexc_data(SYMBOL, TIMEFRAME)
        indicator = SqueezeMomentumIndicator()
        df_signals = indicator.generate_signals(df)
        
        latest_candle = df_signals.iloc[-2] 
        current_signal = latest_candle['signal']
        current_price = latest_candle['close']
        current_time = latest_candle.name
        
        if current_signal != 0:
            if current_signal == 1:
                msg = f"""🟢 <b>BUY Signal</b> 🟢\n\n📊 Asset: <b>{SYMBOL}</b>\n⏱️ Time: {current_time}\n💎 Price: <b>{current_price:.4f}</b>\n📈 Strategy: Squeeze Momentum\n\n⚠️ <i>Manage your risk</i>"""
            else:
                msg = f"""🔴 <b>SELL / Close Signal</b> 🔴\n\n📊 Asset: <b>{SYMBOL}</b>\n⏱️ Time: {current_time}\n💎 Price: <b>{current_price:.4f}</b>\n📉 Strategy: Squeeze Momentum\n\n⚠️ <i>Manage your risk</i>"""
            
            send_telegram_message(msg)
            print(f"Signal sent: {'BUY' if current_signal == 1 else 'SELL'}")
        else:
            print("No new signals at this time.")
            
    except Exception as e:
        print(f"Error fetching data from MEXC: {e}")

if __name__ == "__main__":
    main()