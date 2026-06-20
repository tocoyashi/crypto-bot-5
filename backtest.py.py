import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import requests
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================= Secure Configuration =================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ================= Backtest Settings =================
TIMEFRAME = '15m'
TOP_N_COINS = 20  # We test top 20 to save time during the report generation
DAYS_BACK = 60    # Test the last 60 days (2 months)

# ================= Strategy Settings =================
LEVERAGE = 12
TP1_PERC = 0.6
TP3_PERC = 2.4
TP6_PERC = 9.0
SL_PERC = 6.0

STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']

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
        
        buy_cond = (data['squeeze_release'] == True) & (data['momentum'] > 0) & (mom_inc_safe == True)
        sell_cond = (
            ((data['momentum'] < 0) & (data['momentum'].shift(1).fillna(0) >= 0)) |
            ((mom_inc_safe == False) & (mom_inc_safe.shift(1).fillna(True) == False) & (data['momentum'] > 0))
        )
        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1
        return data

def get_historical_data(symbol, timeframe, days):
    """Fetches historical data in chunks to bypass exchange limits"""
    exchange = ccxt.mexc({'enableRateLimit': True})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_data = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            # Update 'since' to the timestamp of the last fetched candle
            since = ohlcv[-1][0] + 1 
            time.sleep(0.2) # Be nice to the exchange API
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Remove duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_top_mexc_coins(limit=20):
    print(f"Fetching top {limit} coins...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 1000000: usdt_pairs.append({'symbol': symbol, 'volume': vol})
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        return [pair['symbol'] for pair in usdt_pairs[:limit]]
    except Exception as e:
        print(f"Error fetching coins list: {e}")
        return []

def run_backtest():
    print(f"=== Starting {DAYS_BACK}-Day Backtest ===")
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Missing Telegram credentials.")
        return

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins: return

    indicator = SqueezeMomentumIndicator()
    
    total_trades = 0
    total_tp1 = 0
    total_tp3 = 0
    total_tp6 = 0
    total_sl = 0

    for symbol in top_coins:
        print(f"Backtesting {symbol}...")
        df = get_historical_data(symbol, TIMEFRAME, DAYS_BACK)
        if df.empty or len(df) < 25: continue
        
        df_signals = indicator.generate_signals(df)
        
        # Find where signals occurred
        signal_indices = df_signals[df_signals['signal'] != 0].index
        
        for sig_time in signal_indices:
            # Get the location of the signal in the dataframe
            loc = df_signals.index.get_loc(sig_time)
            
            # Ensure we have enough candles ahead to check the outcome (next 20 candles)
            if loc + 20 >= len(df_signals): continue
            
            signal_type = df_signals.loc[sig_time, 'signal']
            entry_price = df_signals.loc[sig_time, 'close']
            
            # Calculate TP/SL prices
            if signal_type == 1: # BUY
                sl_price = entry_price * (1 - SL_PERC / 100)
                tp1_price = entry_price * (1 + TP1_PERC / 100)
                tp3_price = entry_price * (1 + TP3_PERC / 100)
                tp6_price = entry_price * (1 + TP6_PERC / 100)
            else: # SELL
                sl_price = entry_price * (1 + SL_PERC / 100)
                tp1_price = entry_price * (1 - TP1_PERC / 100)
                tp3_price = entry_price * (1 - TP3_PERC / 100)
                tp6_price = entry_price * (1 - TP6_PERC / 100)

            # Check future candles (max 20 candles forward)
            future_df = df_signals.iloc[loc+1 : loc+21]
            
            hit_sl = False
            hit_tp1 = False
            hit_tp3 = False
            hit_tp6 = False

            for i, row in future_df.iterrows():
                if not hit_tp6 and not hit_sl:
                    if signal_type == 1: # Checking Long
                        if row['low'] <= sl_price: hit_sl = True; break
                        if row['high'] >= tp6_price: hit_tp6 = True
                        elif row['high'] >= tp3_price: hit_tp3 = True
                        elif row['high'] >= tp1_price: hit_tp1 = True
                    else: # Checking Short
                        if row['high'] >= sl_price: hit_sl = True; break
                        if row['low'] <= tp6_price: hit_tp6 = True
                        elif row['low'] <= tp3_price: hit_tp3 = True
                        elif row['low'] <= tp1_price: hit_tp1 = True

            total_trades += 1
            if hit_sl: total_sl += 1
            elif hit_tp6: total_tp6 += 1
            elif hit_tp3: total_tp3 += 1
            elif hit_tp1: total_tp1 += 1

    # Calculate Results
    win_rate_tp1 = ((total_tp1 + total_tp3 + total_tp6) / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (total_sl / total_trades * 100) if total_trades > 0 else 0

    msg = f"""📊 <b>Scalping Strategy Backtest Report</b> ⏱️ {DAYS_BACK} Days
                         
🔍 <b>Tested Assets:</b> Top {TOP_N_COINS} MEXC Coins
📈 <b>Timeframe:</b> {TIMEFRAME}
⭐ <b>Leverage:</b> {LEVERAGE}x

━━━━━━━━━━━━━━━━━━━━
<b>📊 Total Signals Found:</b> {total_trades}

🟢 <b>Winners (Hit TP before SL):</b>
• Hit TP1 ({TP1_PERC}%): {total_tp1}
• Hit TP3 ({TP3_PERC}%): {total_tp3}
• Hit TP6 ({TP6_PERC}%) 🚀 Boom: {total_tp6}

🔴 <b>Losers (Hit Stop Loss {SL_PERC}%):</b>
• Stopped Out: {total_sl}

━━━━━━━━━━━━━━━━━━━━
🏆 <b>Win Rate (Reaching at least TP1):</b> <code>{win_rate_tp1:.1f}%</code>
💀 <b>Loss Rate:</b> <code>{loss_rate:.1f}%</code>
━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Note: Slippage and fees are not included in this mathematical model.</i>"""

    print("Sending report to Telegram...")
    send_telegram_message(msg)
    print("Done!")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHANNEL_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

if __name__ == "__main__":
    run_backtest()