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
SL_PERC = 3.6      # Initial Stop Loss
TP1_PERC = 0.8     # Trigger: Close 50%
TP2_PERC = 2.4     # Close 20%
TP3_PERC = 3.5     # Close 20%
TP4_PERC = 7.0     # Close 10%
TRAIL_BE_PERC = 0.25 # Move SL to Entry + 0.25%

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
    print(f"=== Running Partial Close Backtest ===")
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        print("Missing Telegram credentials.")
        return

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins: return

    indicator = SqueezeMomentumIndicator()
    total_trades = 0
    total_sl = 0
    total_trail_sl = 0
    
    # Track exact Profit & Loss as % of initial capital
    total_pnl_points = 0
    gross_profit_points = 0
    gross_loss_points = 0

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
            
            if signal_type == 1: # LONG
                sl_price = entry_price * (1 - SL_PERC / 100)
                tp1_price = entry_price * (1 + TP1_PERC / 100)
                tp2_price = entry_price * (1 + TP2_PERC / 100)
                tp3_price = entry_price * (1 + TP3_PERC / 100)
                tp4_price = entry_price * (1 + TP4_PERC / 100)
            else: # SHORT
                sl_price = entry_price * (1 + SL_PERC / 100)
                tp1_price = entry_price * (1 - TP1_PERC / 100)
                tp2_price = entry_price * (1 - TP2_PERC / 100)
                tp3_price = entry_price * (1 - TP3_PERC / 100)
                tp4_price = entry_price * (1 - TP4_PERC / 100)

            future_df = df_signals.iloc[loc+1 : loc+21]
            
            # Wallet Logic
            remaining_size = 1.0 # 100%
            trade_pnl = 0.0
            current_sl = sl_price
            tp1_hit = False
            tp2_hit = False
            tp3_hit = False

            for i, row in future_df.iterrows():
                if signal_type == 1: # LONG
                    # 1. Check Stop Loss First (Strict)
                    if row['low'] <= current_sl:
                        if not tp1_hit:
                            trade_pnl -= remaining_size * SL_PERC * LEVERAGE
                            total_sl += 1
                        else:
                            # Trailing SL hit (Locks in +0.25% profit for remaining)
                            trade_pnl += remaining_size * TRAIL_BE_PERC * LEVERAGE
                            total_trail_sl += 1
                        break
                    
                    # 2. Check Take Profits (Highest to lowest)
                    if row['high'] >= tp4_price:
                        trade_pnl += remaining_size * TP4_PERC * LEVERAGE
                        remaining_size = 0
                        break
                    elif not tp3_hit and row['high'] >= tp3_price:
                        trade_pnl += 0.2 * TP3_PERC * LEVERAGE
                        remaining_size -= 0.2
                        tp3_hit = True
                    elif not tp2_hit and row['high'] >= tp2_price:
                        trade_pnl += 0.2 * TP2_PERC * LEVERAGE
                        remaining_size -= 0.2
                        tp2_hit = True
                    elif not tp1_hit and row['high'] >= tp1_price:
                        trade_pnl += 0.5 * TP1_PERC * LEVERAGE
                        remaining_size -= 0.5
                        tp1_hit = True
                        # Move SL to Entry + 0.25%
                        current_sl = entry_price * (1 + TRAIL_BE_PERC / 100)

                else: # SHORT
                    if row['high'] >= current_sl:
                        if not tp1_hit:
                            trade_pnl -= remaining_size * SL_PERC * LEVERAGE
                            total_sl += 1
                        else:
                            trade_pnl += remaining_size * TRAIL_BE_PERC * LEVERAGE
                            total_trail_sl += 1
                        break
                    
                    if row['low'] <= tp4_price:
                        trade_pnl += remaining_size * TP4_PERC * LEVERAGE
                        remaining_size = 0
                        break
                    elif not tp3_hit and row['low'] <= tp3_price:
                        trade_pnl += 0.2 * TP3_PERC * LEVERAGE
                        remaining_size -= 0.2
                        tp3_hit = True
                    elif not tp2_hit and row['low'] <= tp2_price:
                        trade_pnl += 0.2 * TP2_PERC * LEVERAGE
                        remaining_size -= 0.2
                        tp2_hit = True
                    elif not tp1_hit and row['low'] <= tp1_price:
                        trade_pnl += 0.5 * TP1_PERC * LEVERAGE
                        remaining_size -= 0.5
                        tp1_hit = True
                        current_sl = entry_price * (1 - TRAIL_BE_PERC / 100)

            total_trades += 1
            total_pnl_points += trade_pnl
            if trade_pnl > 0:
                gross_profit_points += trade_pnl
            else:
                gross_loss_points += abs(trade_pnl)

    # Calculate final stats
    net_profit_trades = total_trades - total_sl - total_trail_sl
    win_rate = (net_profit_trades / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (total_sl / total_trades * 100) if total_trades > 0 else 0
    trail_rate = (total_trail_sl / total_trades * 100) if total_trades > 0 else 0
    
    # Net Capital Growth in % (Assuming 1000$ initial per trade, simplified to points)
    expected_capital_growth = (total_pnl_points / total_trades) if total_trades > 0 else 0

    msg = f"""📊 <b>Institutional Partial Close Backtest</b> ⏱️ {DAYS_BACK} Days
                         
🔍 <b>Tested Assets:</b> Top {TOP_N_COINS} MEXC Coins
📈 <b>Timeframe:</b> {TIMEFRAME}
⭐ <b>Leverage:</b> {LEVERAGE}x

⚙️ <b>Execution Logic:</b>
• TP1 ({TP1_PERC}%): Close 50%
• TP2 ({TP2_PERC}%): Close 20%
• TP3 ({TP3_PERC}%): Close 20%
• TP4 ({TP4_PERC}%): Close 10%
• SL ({SL_PERC}%): Initial Risk
• Trail SL: Entry + {TRAIL_BE_PERC}% after TP1

━━━━━━━━━━━━━━━━━━━━
📊 <b>Total Signals:</b> {total_trades}

🟢 <b>Full Winners (Hit TP3/TP4):</b> {net_profit_trades} (<code>{win_rate:.1f}%</code>)
🟡 <b>Trail SL (Hit TP1, stopped safe):</b> {total_trail_sl} (<code>{trail_rate:.1f}%</code>)
🔴 <b>Full Losers (Hit initial SL):</b> {total_sl} (<code>{loss_rate:.1f}%</code>)

━━━━━━━━━━━━━━━━━━━━
💰 <b>Mathematical Performance:</b>
• Total Gross Profit: <code>{gross_profit_points:,.0f}</code> pts
• Total Gross Loss: <code>{gross_loss_points:,.0f}</code> pts
• Net PnL Points: <code>{total_pnl_points:,.0f}</code> pts

📈 <b>Avg Capital Change per Trade:</b> <code>{expected_capital_growth:.2f}%</code>
{'🟢 <b>VERDICT: PROFITABLE SYSTEM</b>' if total_pnl_points > 0 else '🔴 <b>VERDICT: NEEDS OPTIMIZATION</b>'}
━━━━━━━━━━━━━━━━━━━━"""

    print("Sending Institutional report to Telegram...")
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