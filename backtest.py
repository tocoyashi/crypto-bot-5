import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import requests
import os
import time
from datetime import datetime, timedelta
import warnings
import json
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# ========================= BACKTEST CONFIGURATION ==========================
# ============================================================================

TIMEFRAME = '15m'
TOP_N_COINS = 50          # Number of top coins to scan
DAYS_BACK = 90            # 3 months backtest period
INITIAL_BALANCE = 1000    # Starting capital in USDT
RISK_PER_TRADE = 0.02     # 2% risk per trade
LEVERAGE = 10

# ========================= TAKE PROFIT / STOP LOSS ==========================
TP1_PERC = 0.6
TP2_PERC = 1.5
TP3_PERC = 2.4
TP4_PERC = 5.0
TP5_PERC = 7.0
TP6_PERC = 9.0
SL_PERC = 6.0

# ========================= EXCLUDED COINS ==========================
# Stablecoins (always excluded)
STABLECOINS = [
    'USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 
    'USDP/USDT', 'PYUSD/USDT', 'UST/USDT', 'BUSD/USDT'
]

# User requested exclusions
EXCLUDED_COINS = [
    'SPACEX(PRE)/USDT', 'RAIN/USDT', 'TOYL/USDT', 'WXT/USDT',
    'UPC/USDT', 'DN/USDT', 'AIXPLAY/USDT', 'MBG/USDT',
    'KAZAR/USDT', 'STAR/USDT'
]

ALL_EXCLUSIONS = set(STABLECOINS + EXCLUDED_COINS)

# ========================= TELEGRAM (Optional) ==========================
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')

# ============================================================================
# ========================= SQUEEZE MOMENTUM INDICATOR ========================
# ============================================================================

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
            if len(x) < length:
                return np.nan
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


# ============================================================================
# ========================= BACKTEST ENGINE ================================
# ============================================================================

class BacktestEngine:
    def __init__(self, initial_balance=1000, risk_per_trade=0.02, leverage=10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.trades = []
        self.equity_curve = []

    def calculate_targets(self, entry_price, signal_type):
        """Calculate TP and SL prices"""
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

        return (tp1, tp2, tp3, tp4, tp5, tp6, sl)

    def simulate_trade(self, entry_price, signal_type, entry_time, symbol, df_future):
        """
        Simulate a trade from entry to exit.
        Checks each future candle for TP or SL hit.
        """
        tp1, tp2, tp3, tp4, tp5, tp6, sl = self.calculate_targets(entry_price, signal_type)

        # Position size based on risk
        risk_amount = self.balance * self.risk_per_trade

        if signal_type == 1:  # Long
            price_risk_pct = (entry_price - sl) / entry_price
            if price_risk_pct == 0:
                return None
            position_size = risk_amount / price_risk_pct

            for idx, row in df_future.iterrows():
                high, low = row['high'], row['low']

                # Check SL first
                if low <= sl:
                    pnl = -risk_amount * self.leverage
                    self.balance += pnl
                    return {
                        'symbol': symbol, 'type': 'LONG', 'entry': entry_price,
                        'exit': sl, 'exit_type': 'SL', 'pnl': pnl,
                        'pnl_pct': (pnl / self.initial_balance) * 100,
                        'entry_time': entry_time, 'exit_time': idx,
                        'duration_candles': len(df_future.loc[:idx]),
                        'tp_target': None
                    }

                # Check TPs in order (highest first for long)
                for tp_name, tp_price in [('TP6', tp6), ('TP5', tp5), ('TP4', tp4),
                                           ('TP3', tp3), ('TP2', tp2), ('TP1', tp1)]:
                    if high >= tp_price:
                        pnl = risk_amount * (abs(tp_price - entry_price) / abs(entry_price - sl)) * self.leverage
                        self.balance += pnl
                        return {
                            'symbol': symbol, 'type': 'LONG', 'entry': entry_price,
                            'exit': tp_price, 'exit_type': tp_name, 'pnl': pnl,
                            'pnl_pct': (pnl / self.initial_balance) * 100,
                            'entry_time': entry_time, 'exit_time': idx,
                            'duration_candles': len(df_future.loc[:idx]),
                            'tp_target': tp_name
                        }

        else:  # Short
            price_risk_pct = (sl - entry_price) / entry_price
            if price_risk_pct == 0:
                return None
            position_size = risk_amount / price_risk_pct

            for idx, row in df_future.iterrows():
                high, low = row['high'], row['low']

                # Check SL first
                if high >= sl:
                    pnl = -risk_amount * self.leverage
                    self.balance += pnl
                    return {
                        'symbol': symbol, 'type': 'SHORT', 'entry': entry_price,
                        'exit': sl, 'exit_type': 'SL', 'pnl': pnl,
                        'pnl_pct': (pnl / self.initial_balance) * 100,
                        'entry_time': entry_time, 'exit_time': idx,
                        'duration_candles': len(df_future.loc[:idx]),
                        'tp_target': None
                    }

                # Check TPs in order (lowest first for short)
                for tp_name, tp_price in [('TP6', tp6), ('TP5', tp5), ('TP4', tp4),
                                           ('TP3', tp3), ('TP2', tp2), ('TP1', tp1)]:
                    if low <= tp_price:
                        pnl = risk_amount * (abs(entry_price - tp_price) / abs(sl - entry_price)) * self.leverage
                        self.balance += pnl
                        return {
                            'symbol': symbol, 'type': 'SHORT', 'entry': entry_price,
                            'exit': tp_price, 'exit_type': tp_name, 'pnl': pnl,
                            'pnl_pct': (pnl / self.initial_balance) * 100,
                            'entry_time': entry_time, 'exit_time': idx,
                            'duration_candles': len(df_future.loc[:idx]),
                            'tp_target': tp_name
                        }

        # Trade didn't close within available data
        return None

    def run_backtest(self, all_signals_data):
        """Run backtest on all signals"""
        print(f"\n🏃 Simulating {len(all_signals_data)} trades...")

        for i, signal in enumerate(all_signals_data):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(all_signals_data)} trades...")

            result = self.simulate_trade(
                signal['entry_price'], signal['signal_type'],
                signal['signal_time'], signal['symbol'], signal['df_future']
            )
            if result:
                self.trades.append(result)
                self.equity_curve.append({
                    'time': result['exit_time'],
                    'balance': self.balance
                })

        return self.generate_report()

    def generate_report(self):
        if not self.trades:
            return {"error": "No trades executed"}

        df_trades = pd.DataFrame(self.trades)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = df_trades['pnl'].sum()
        avg_pnl = df_trades['pnl'].mean()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        max_drawdown = self.calculate_max_drawdown()

        profit_factor = abs(df_trades[df_trades['pnl'] > 0]['pnl'].sum() /
                           df_trades[df_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')

        tp_distribution = df_trades['exit_type'].value_counts().to_dict()

        # Calculate expectancy
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_usdt': self.balance - self.initial_balance,
            'total_return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'tp_distribution': tp_distribution,
            'trades': df_trades.to_dict('records')
        }

    def calculate_max_drawdown(self):
        if not self.equity_curve:
            return 0

        balances = [self.initial_balance] + [e['balance'] for e in self.equity_curve]
        peak = balances[0]
        max_dd = 0

        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd


# ============================================================================
# ========================= DATA FETCHING ====================================
# ============================================================================

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
            since = ohlcv[-1][0] + 1
            time.sleep(0.3)
        except Exception as e:
            print(f"    ⚠️ Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


def get_top_mexc_coins(limit=50):
    """Fetches top N coins sorted by 24h volume in USDT"""
    print(f"📥 Fetching top {limit} coins from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []

        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in ALL_EXCLUSIONS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 500000:  # Minimum 500K USDT volume
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})

        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]

        print(f"✅ Fetched {len(top_coins)} coins")
        print(f"🚫 Excluded: {len(EXCLUDED_COINS)} user-specified + {len(STABLECOINS)} stablecoins")
        return top_coins
    except Exception as e:
        print(f"❌ Error fetching coins list: {e}")
        return []


# ============================================================================
# ========================= TELEGRAM REPORTING ==============================
# ============================================================================

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHANNEL_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram error: {e}")


def format_report_telegram(report, days_back, top_n):
    if 'error' in report:
        return f"❌ Backtest Error: {report['error']}"

    tp_dist = report['tp_distribution']
    tp6 = tp_dist.get('TP6', 0)
    tp5 = tp_dist.get('TP5', 0)
    tp4 = tp_dist.get('TP4', 0)
    tp3 = tp_dist.get('TP3', 0)
    tp2 = tp_dist.get('TP2', 0)
    tp1 = tp_dist.get('TP1', 0)
    sl = tp_dist.get('SL', 0)

    win_rate = report['win_rate']
    total_wins = report['winning_trades']
    total_loss = report['losing_trades']

    msg = f"""📊 <b>Backtest Report — {days_back} Days</b>

🔍 <b>Assets:</b> Top {top_n} MEXC Coins
⏱️ <b>Timeframe:</b> {TIMEFRAME}
⭐ <b>Leverage:</b> {LEVERAGE}x
💰 <b>Initial:</b> {report['initial_balance']} USDT
📈 <b>Final:</b> {report['final_balance']:.2f} USDT

━━━━━━━━━━━━━━━━━━━━
<b>📊 Total Signals:</b> {report['total_trades']}

🟢 <b>Winners ({win_rate:.1f}%):</b> {total_wins}
• TP1 ({TP1_PERC}%): {tp1}
• TP2 ({TP2_PERC}%): {tp2}
• TP3 ({TP3_PERC}%): {tp3}
• TP4 ({TP4_PERC}%): {tp4}
• TP5 ({TP5_PERC}%): {tp5}
• TP6 ({TP6_PERC}%) 🚀: {tp6}

🔴 <b>Losers ({100-win_rate:.1f}%):</b> {total_loss}
• SL ({SL_PERC}%): {sl}

━━━━━━━━━━━━━━━━━━━━
💵 <b>Total Return:</b> <code>{report['total_return_pct']:+.2f}%</code>
📉 <b>Max Drawdown:</b> <code>{report['max_drawdown_pct']:.2f}%</code>
⚖️ <b>Profit Factor:</b> <code>{report['profit_factor']:.2f}</code>
🎯 <b>Expectancy:</b> <code>{report['expectancy']:.2f} USDT</code>

⚠️ <i>Fees/slippage not included.</i>"""

    return msg


# ============================================================================
# ========================= MAIN BACKTEST ====================================
# ============================================================================

def run_full_backtest():
    print("=" * 70)
    print("🚀  MEXC SQUEEZE MOMENTUM — 3 MONTH BACKTEST ENGINE")
    print("=" * 70)
    print(f"📅 Period: Last {DAYS_BACK} days")
    print(f"💰 Capital: {INITIAL_BALANCE} USDT | Risk: {RISK_PER_TRADE*100}% | Leverage: {LEVERAGE}x")
    print(f"🎯 TPs: {TP1_PERC}% / {TP3_PERC}% / {TP6_PERC}% | SL: {SL_PERC}%")
    print(f"🚫 Excluded: {len(EXCLUDED_COINS)} coins")
    print("=" * 70)

    # Get top coins
    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins:
        print("❌ Failed to get coin list. Aborting.")
        return None

    print(f"\n🪙 Coins to test: {', '.join(top_coins[:5])}... ({len(top_coins)} total)")

    indicator = SqueezeMomentumIndicator()
    all_signals = []

    print(f"\n📥 Fetching historical data ({DAYS_BACK} days, {TIMEFRAME})...")
    print("-" * 70)

    for i, symbol in enumerate(top_coins):
        print(f"[{i+1:2d}/{len(top_coins)}] {symbol:20s} — ", end="", flush=True)

        try:
            df = get_historical_data(symbol, TIMEFRAME, DAYS_BACK)

            if df.empty or len(df) < 50:
                print(f"❌ Insufficient data ({len(df)} candles)")
                continue

            print(f"✅ {len(df):5d} candles — ", end="", flush=True)

            # Generate signals
            df_signals = indicator.generate_signals(df)

            # Find all signals
            signal_mask = df_signals['signal'] != 0
            signal_rows = df_signals[signal_mask].copy()

            if len(signal_rows) == 0:
                print("📭 No signals")
                continue

            print(f"🔔 {len(signal_rows):3d} signals")

            for idx, row in signal_rows.iterrows():
                # Get future candles for trade simulation
                future_mask = df_signals.index > idx
                df_future = df_signals[future_mask].copy()

                if len(df_future) < 10:
                    continue

                signal_type = int(row['signal'])
                entry_price = float(row['close'])

                all_signals.append({
                    'symbol': symbol,
                    'signal_time': idx,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'df_future': df_future.head(200)  # Limit for performance
                })

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print(f"\n{'=' * 70}")
    print(f"📊 TOTAL SIGNALS COLLECTED: {len(all_signals)}")
    print(f"{'=' * 70}")

    if len(all_signals) == 0:
        print("❌ No signals to backtest!")
        return None

    # Run backtest
    engine = BacktestEngine(INITIAL_BALANCE, RISK_PER_TRADE, LEVERAGE)
    report = engine.run_backtest(all_signals)

    # Print detailed report
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total Trades:       {report['total_trades']}")
    print(f"Winning Trades:     {report['winning_trades']} ({report['win_rate']:.1f}%)")
    print(f"Losing Trades:      {report['losing_trades']} ({100-report['win_rate']:.1f}%)")
    print(f"Initial Balance:    {report['initial_balance']:.2f} USDT")
    print(f"Final Balance:      {report['final_balance']:.2f} USDT")
    print(f"Total Return:       {report['total_return_pct']:+.2f}%")
    print(f"Total PnL:          {report['total_pnl']:+.2f} USDT")
    print(f"Avg PnL/Trade:      {report['avg_pnl']:+.2f} USDT")
    print(f"Avg Win:            {report['avg_win']:+.2f} USDT")
    print(f"Avg Loss:           {report['avg_loss']:+.2f} USDT")
    print(f"Max Drawdown:       {report['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor:      {report['profit_factor']:.2f}")
    print(f"Expectancy:         {report['expectancy']:.2f} USDT")
    print(f"\nTP Distribution:")
    for tp, count in report['tp_distribution'].items():
        print(f"  {tp}: {count}")
    print("=" * 70)

    # Save report to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"backtest_report_{timestamp}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Report saved to: {report_filename}")

    # Send to Telegram if configured
    if TELEGRAM_TOKEN and CHANNEL_ID:
        print("📤 Sending report to Telegram...")
        msg = format_report_telegram(report, DAYS_BACK, len(top_coins))
        send_telegram_message(msg)
        print("✅ Telegram sent!")

    return report, engine


if __name__ == "__main__":
    run_full_backtest()
