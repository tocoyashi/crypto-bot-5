import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import requests
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= Secure Configuration (From GitHub Secrets) =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ================= Trading Settings =================
TIMEFRAME = '15m'      
TOP_N_COINS = 50       
STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']

# ================= Risk Management Settings =================
LEVERAGE = 10
TP1_PERC = 0.6
TP2_PERC = 1.5
TP3_PERC = 2.4
TP4_PERC = 5.0
TP5_PERC = 7.0
TP6_PERC = 9.0
SL_PERC = 6.0

def calculate_targets(entry_price, signal_type):
    """Calculates TP and SL prices based on signal type (1 = BUY, -1 = SELL)"""
    if signal_type == 1:  # BUY (Long)
        tp1 = entry_price * (1 + TP1_PERC / 100)
        tp2 = entry_price * (1 + TP2_PERC / 100)
        tp3 = entry_price * (1 + TP3_PERC / 100)
        tp4 = entry_price * (1 + TP4_PERC / 100)
        tp5 = entry_price * (1 + TP5_PERC / 100)
        tp6 = entry_price * (1 + TP6_PERC / 100)
        sl  = entry_price * (1 - SL_PERC / 100)
    else:  # SELL (Short)
        tp1 = entry_price * (1 - TP1_PERC / 100)
        tp2 = entry_price * (1 - TP2_PERC / 100)
        tp3 = entry_price * (1 - TP3_PERC / 100)
        tp4 = entry_price * (1 - TP4_PERC / 100)
        tp5 = entry_price * (1 - TP5_PERC / 100)
        tp6 = entry_price * (1 - TP6_PERC / 100)
        sl  = entry_price * (1 + SL_PERC / 100)

    p = 6 
    targets_text = f"""⭐ <b>Leverage:</b> {LEVERAGE}x

🎯 <b>Take Profits:</b>
TP1: <code>{tp1:.{p}f}</code>
TP2: <code>{tp2:.{p}f}</code>
TP3: <code>{tp3:.{p}f}</code>
TP4: <code>{tp4:.{p}f}</code>
TP5: <code>{tp5:.{p}f}</code>
TP6: <code>{tp6:.{p}f}</code> 🚀 Boom

🛑 <b>Stop Loss:</b> <code>{sl:.{p}f}</code>"""
    
    return targets_text

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
        
        squeeze_on_safe = data['squeeze_on'].fillna(False).astype(bool)
        mom_inc_safe = data['momentum_increasing'].fillna(False).astype(bool)
        
        data['squeeze_release'] = (squeeze_on_safe.shift(1) == True) & (squeeze_on_safe == False)
        
        buy_cond = (
            (data['squeeze_release'] == True) & 
            (data['momentum'] > 0) & 
            (mom_inc_safe == True)
        )
        
        sell_cond = (
            ((data['momentum'] < 0) & (data['momentum'].shift(1).fillna(0) >= 0)) |
            ((mom_inc_safe == False) & (mom_inc_safe.shift(1).fillna(True) == False) & (data['momentum'] > 0))
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

def get_top_mexc_coins(limit=50):
    """Fetches top N coins sorted by 24h volume in USDT"""
    print(f"Fetching top {limit} coins by volume from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 1000000: 
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})
        
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]
        print(f"Successfully fetched: {top_coins[:5]} ... (and {len(top_coins)-5} more)")
        return top_coins
    except Exception as e:
        print(f"Error fetching top coins list: {e}")
        return []

def main():
    print("=== Running Multi-Coin Scalping Bot ===")
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Environment not configured.")
        return

    print(f"[{datetime.now()}] Starting scan for TOP {TOP_N_COINS} coins on {TIMEFRAME} timeframe...")
    
    top_coins = get_top_mexc_coins(TOP_N_COINS)
    
    if not top_coins:
        print("Failed to get coin list. Aborting run.")
        return

    indicator = SqueezeMomentumIndicator()
    signals_found = 0

    for symbol in top_coins:
        try:
            time.sleep(0.5) 
            
            df = get_mexc_data(symbol, TIMEFRAME)
            df_signals = indicator.generate_signals(df)
            
            latest_candle = df_signals.iloc[-2] 
            current_signal = latest_candle['signal']
            current_price = latest_candle['close']
            current_time = latest_candle.name
            
            if current_signal != 0:
                signals_found += 1
                targets_str = calculate_targets(current_price, current_signal)
                
                if current_signal == 1:
                    msg = f"""🟢 <b>BUY Signal (Long)</b> 🟢
                    
📊 Asset: <b>{symbol}</b>
⏱️ Time: {current_time}
💎 Entry Price: <code>{current_price:.6f}</code>
📈 Strategy: Scalping

{targets_str}

⚠️ <i>Manage your risk</i>"""
                else:
                    msg = f"""🔴 <b>SELL Signal (Short)</b> 🔴
                    
📊 Asset: <b>{symbol}</b>
⏱️ Time: {current_time}
💎 Entry Price: <code>{current_price:.6f}</code>
📉 Strategy: Scalping

{targets_str}

⚠️ <i>Manage your risk</i>"""
                
                send_telegram_message(msg)
                print(f"-> Signal sent for {symbol}: {'BUY' if current_signal == 1 else 'SELL'}")
                
        except Exception as e:
            pass 

    print(f"[{datetime.now()}] Scan finished. Total signals found: {signals_found}")

if __name__ == "__main__":
    main()
