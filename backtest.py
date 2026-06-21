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
TOP_N_COINS = 50
DAYS_BACK = 60

# ================= Strategy Settings =================
LEVERAGE = 12
TP1_PERC = 0.8    
SL_PERC = 3.6     
TP3_PERC = 2.4
TP6_PERC = 9.0

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
    exchange = ccxt.mexc({'enableRateLimit': True})
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    all_data = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1 
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_top_mexc_coins(limit=50):
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
    print(f"=== Running 15m Backtest with Move To Break Even ===")
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Missing Telegram credentials.")
        return

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins: return

    indicator = SqueezeMomentumIndicator()
    total_trades = 0
    total_tp3 = 0
    total_tp6 = 0
    total_sl = 0
    total_be = 0  # New Counter for Break Even

    for symbol in top_coins:
        print(f"Backtesting {symbol}...")
        df = get_historical_data(symbol, TIMEFRAME, DAYS_BACK)
        if df.empty or len(df) < 25: continue
        df_signals = indicator.generate_signals(df)
        signal_indices = df_signals[df_signals['signal'] != 0].index
        
        for sig_time in signal_indices:
            loc = df_signals.index.get_loc(sig_time)
            if loc + 20 >= len(df_signals): continue
            
            signal_type = df_signals.loc[sig_time, 'signal']
            entry_price = df_signals.loc[sig_time, 'close']
            
            if signal_type == 1:
                sl_price = entry_price * (1 - SL_PERC / 100)
                tp1_price = entry_price * (1 + TP1_PERC / 100)
                tp3_price = entry_price * (1 + TP3_PERC / 100)
                tp6_price = entry_price * (1 + TP6_PERC / 100)
            else:
                sl_price = entry_price * (1 + SL_PERC / 100)
                tp1_price = entry_price * (1 - TP1_PERC / 100)
                tp3_price = entry_price * (1 - TP3_PERC / 100)
                tp6_price = entry_price * (1 - TP6_PERC / 100)

            future_df = df_signals.iloc[loc+1 : loc+21]
            
            # New Dynamic Logic Variables
            current_sl = sl_price
            tp1_hit = False
            final_result = "open" # can be 'sl', 'be', 'tp3', 'tp6'

            for i, row in future_df.iterrows():
                if signal_type == 1: # LONG
                    # Check Stop Loss First (Conservative approach)
                    if row['low'] <= current_sl:
                        if tp1_hit:
                            final_result = "be" # Hit Break Even
                        else:
                            final_result = "sl" # Hit original Stop Loss
                        break
                    
                    # Check Take Profits
                    if row['high'] >= tp6_price:
                        final_result = "tp6"; break
                    if row['high'] >= tp3_price:
                        final_result = "tp3"; break
                    
                    # Check TP1 to activate Break Even
                    if not tp1_hit and row['high'] >= tp1_price:
                        tp1_hit = True
                        current_sl = entry_price # MOVE SL TO ENTRY PRICE

                else: # SHORT
                    if row['high'] >= current_sl:
                        if tp1_hit:
                            final_result = "be"
                        else:
                            final_result = "sl"
                        break
                    
                    if row['low'] <= tp6_price:
                        final_result = "tp6"; break
                    if row['low'] <= tp3_price:
                        final_result = "tp3"; break
                    
                    if not tp1_hit and row['low'] <= tp1_price:
                        tp1_hit = True
                        current_sl = entry_price

            total_trades += 1
            if final_result == "sl": total_sl += 1
            elif final_result == "be": total_be += 1
            elif final_result == "tp6": total_tp6 += 1
            elif final_result == "tp3": total_tp3 += 1

    # Win rate is now only TP3 and TP6 (Since TP1 is no longer a final exit)
    win_rate = ((total_tp3 + total_tp6) / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (total_sl / total_trades * 100) if total_trades > 0 else 0
    be_rate = (total_be / total_trades * 100) if total_trades > 0 else 0

    msg = f"""📊 <b>Advanced Backtest (Move To Break Even)</b> ⏱️ {DAYS_BACK} Days
                         
🔍 <b>Tested Assets:</b> Top {TOP_N_COINS} MEXC Coins
📈 <b>Timeframe:</b> {TIMEFRAME}
⭐ <b>Leverage:</b> {LEVERAGE}x
🛑 <b>Initial Stop Loss:</b> {SL_PERC}%
🎯 <b>Target 1 (Triggers BE):</b> {TP1_PERC}%

━━━━━━━━━━━━━━━━━━━━
📊 <b>Total Signals Found:</b> {total_trades}

🟢 <b>Full Winners:</b>
• Hit TP3 ({TP3_PERC}%): {total_tp3}
• Hit TP6 ({TP6_PERC}%) 🚀 Boom: {total_tp6}

🟡 <b>Break Even (Hit TP1 then stopped at Entry):</b>
• Saved from Loss: {total_be}

🔴 <b>Losers (Hit SL before TP1):</b>
• Stopped Out: {total_sl}

━━━━━━━━━━━━━━━━━━━━
🏆 <b>Win Rate (TP3 + TP6):</b> <code>{win_rate:.1f}%</code>
🛡️ <b>Break Even Rate:</b> <code>{be_rate:.1f}%</code>
💀 <b>Loss Rate:</b> <code>{loss_rate:.1f}%</code>
━━━━━━━━━━━━━━━━━━━━

⚠️ <i>Logic: TP1 is no longer an exit, it's a trigger to protect capital.</i>"""

    print("Sending Advanced report to Telegram...")
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