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

# ================= BACKTEST CONFIGURATION =================
TIMEFRAME = '15m'
TOP_N_COINS = 50
DAYS_BACK = 30
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.02
LEVERAGE = 10

TP1_PERC = 0.8
TP2_PERC = 1.6
TP3_PERC = 3.0
TP4_PERC = 6.0
SL_PERC = 6.0
SL_MOVE_AFTER_TP1 = 0.10

TP1_SIZE = 0.50
TP2_SIZE = 0.25
TP3_SIZE = 0.10
TP4_SIZE = 0.15

FEE_RATE = 0.0006

STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT',
    'USDP/USDT', 'PYUSD/USDT', 'UST/USDT', 'BUSD/USDT']
EXCLUDED_COINS = [
    'SPACEX(PRE)/USDT', 'RAIN/USDT', 'TOYL/USDT', 'WXT/USDT',
    'UPC/USDT', 'DN/USDT', 'AIXPLAY/USDT', 'MBG/USDT',
    'KAZAR/USDT', 'STAR/USDT']
ALL_EXCLUSIONS = set(STABLECOINS + EXCLUDED_COINS)

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
CHANNEL_ID = os.environ.get('CHANNEL_ID')


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
            (mom_inc_safe == True))
        sell_cond = (
            ((data['momentum'] < 0) & (data['momentum'].shift(1).fillna(0) >= 0)) |
            ((mom_inc_safe == False) & (mom_inc_safe.shift(1).fillna(True) == False) & (data['momentum'] > 0)))
        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1
        return data


class AdvancedBacktestEngine:
    def __init__(self, initial_balance=1000, risk_per_trade=0.02, leverage=10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.trades = []
        self.equity_curve = []

    def calculate_targets(self, entry_price, signal_type):
        if signal_type == 1:
            tp1 = entry_price * (1 + TP1_PERC / 100)
            tp2 = entry_price * (1 + TP2_PERC / 100)
            tp3 = entry_price * (1 + TP3_PERC / 100)
            tp4 = entry_price * (1 + TP4_PERC / 100)
            sl = entry_price * (1 - SL_PERC / 100)
            sl_after_tp1 = entry_price * (1 + SL_MOVE_AFTER_TP1 / 100)
        else:
            tp1 = entry_price * (1 - TP1_PERC / 100)
            tp2 = entry_price * (1 - TP2_PERC / 100)
            tp3 = entry_price * (1 - TP3_PERC / 100)
            tp4 = entry_price * (1 - TP4_PERC / 100)
            sl = entry_price * (1 + SL_PERC / 100)
            sl_after_tp1 = entry_price * (1 - SL_MOVE_AFTER_TP1 / 100)
        return {'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'tp4': tp4,
                'sl_initial': sl, 'sl_after_tp1': sl_after_tp1}

    def simulate_trade(self, entry_price, signal_type, entry_time, symbol, df_future):
        targets = self.calculate_targets(entry_price, signal_type)
        risk_amount = self.balance * self.risk_per_trade
        if signal_type == 1:
            price_risk_pct = (entry_price - targets['sl_initial']) / entry_price
        else:
            price_risk_pct = (targets['sl_initial'] - entry_price) / entry_price
        if price_risk_pct == 0:
            return None
        position_notional = risk_amount / price_risk_pct
        entry_fee = position_notional * FEE_RATE
        remaining_size = 1.0
        realized_pnl = 0.0
        total_fees = entry_fee
        current_sl = targets['sl_initial']
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        tp4_hit = False
        exits = []

        for idx, row in df_future.iterrows():
            high, low = row['high'], row['low']
            if signal_type == 1:
                if low <= current_sl and remaining_size > 0:
                    close_size = remaining_size
                    pnl_pct = (current_sl - entry_price) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size = 0
                    exits.append({'type': 'SL', 'price': current_sl, 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                    break
                if not tp1_hit and high >= targets['tp1'] and remaining_size > 0:
                    close_size = TP1_SIZE
                    pnl_pct = (targets['tp1'] - entry_price) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp1_hit = True
                    current_sl = targets['sl_after_tp1']
                    exits.append({'type': 'TP1', 'price': targets['tp1'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp2_hit and high >= targets['tp2'] and remaining_size > 0:
                    close_size = TP2_SIZE
                    pnl_pct = (targets['tp2'] - entry_price) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp2_hit = True
                    exits.append({'type': 'TP2', 'price': targets['tp2'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp3_hit and high >= targets['tp3'] and remaining_size > 0:
                    close_size = TP3_SIZE
                    pnl_pct = (targets['tp3'] - entry_price) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp3_hit = True
                    exits.append({'type': 'TP3', 'price': targets['tp3'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp4_hit and high >= targets['tp4'] and remaining_size > 0:
                    close_size = TP4_SIZE
                    pnl_pct = (targets['tp4'] - entry_price) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp4_hit = True
                    exits.append({'type': 'TP4', 'price': targets['tp4'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                    break
            else:
                if high >= current_sl and remaining_size > 0:
                    close_size = remaining_size
                    pnl_pct = (entry_price - current_sl) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size = 0
                    exits.append({'type': 'SL', 'price': current_sl, 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                    break
                if not tp1_hit and low <= targets['tp1'] and remaining_size > 0:
                    close_size = TP1_SIZE
                    pnl_pct = (entry_price - targets['tp1']) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp1_hit = True
                    current_sl = targets['sl_after_tp1']
                    exits.append({'type': 'TP1', 'price': targets['tp1'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp2_hit and low <= targets['tp2'] and remaining_size > 0:
                    close_size = TP2_SIZE
                    pnl_pct = (entry_price - targets['tp2']) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp2_hit = True
                    exits.append({'type': 'TP2', 'price': targets['tp2'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp3_hit and low <= targets['tp3'] and remaining_size > 0:
                    close_size = TP3_SIZE
                    pnl_pct = (entry_price - targets['tp3']) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp3_hit = True
                    exits.append({'type': 'TP3', 'price': targets['tp3'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                if not tp4_hit and low <= targets['tp4'] and remaining_size > 0:
                    close_size = TP4_SIZE
                    pnl_pct = (entry_price - targets['tp4']) / entry_price
                    pnl = position_notional * close_size * pnl_pct * LEVERAGE
                    exit_fee = position_notional * close_size * FEE_RATE
                    realized_pnl += pnl
                    total_fees += exit_fee
                    remaining_size -= close_size
                    tp4_hit = True
                    exits.append({'type': 'TP4', 'price': targets['tp4'], 'size': close_size,
                                  'pnl': pnl, 'fee': exit_fee, 'time': idx})
                    break

        if remaining_size > 0:
            return None
        net_pnl = realized_pnl - total_fees
        self.balance += net_pnl
        final_exit = exits[-1]['type'] if exits else 'UNKNOWN'
        return {
            'symbol': symbol, 'type': 'LONG' if signal_type == 1 else 'SHORT',
            'entry': entry_price, 'entry_time': entry_time,
            'exit_time': exits[-1]['time'] if exits else None,
            'final_exit': final_exit, 'net_pnl': net_pnl,
            'gross_pnl': realized_pnl, 'total_fees': total_fees,
            'pnl_pct': (net_pnl / self.initial_balance) * 100,
            'exits': exits, 'position_notional': position_notional,
            'sl_initial': targets['sl_initial'], 'sl_after_tp1': targets['sl_after_tp1']}

    def run_backtest(self, all_signals_data):
        print("Simulating " + str(len(all_signals_data)) + " trades...")
        for i, signal in enumerate(all_signals_data):
            if (i + 1) % 50 == 0:
                print("  Processed " + str(i+1) + "/" + str(len(all_signals_data)) + " trades...")
            result = self.simulate_trade(
                signal['entry_price'], signal['signal_type'],
                signal['signal_time'], signal['symbol'], signal['df_future'])
            if result:
                self.trades.append(result)
                self.equity_curve.append({'time': result['exit_time'], 'balance': self.balance})
        return self.generate_report()

    def generate_report(self):
        if not self.trades:
            return {"error": "No trades executed"}
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['net_pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['net_pnl'] < 0])
        breakeven = total_trades - winning_trades - losing_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_gross_pnl = sum(t['gross_pnl'] for t in self.trades)
        total_fees = sum(t['total_fees'] for t in self.trades)
        total_net_pnl = sum(t['net_pnl'] for t in self.trades)
        avg_net_pnl = total_net_pnl / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['net_pnl'] for t in self.trades if t['net_pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['net_pnl'] for t in self.trades if t['net_pnl'] < 0]) if losing_trades > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        profit_factor = abs(sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0) /
                           sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0)) if losing_trades > 0 else float('inf')
        exit_dist = defaultdict(int)
        for t in self.trades:
            exit_dist[t['final_exit']] += 1
        tp1_hits = sum(1 for t in self.trades if any(e['type'] == 'TP1' for e in t['exits']))
        tp2_hits = sum(1 for t in self.trades if any(e['type'] == 'TP2' for e in t['exits']))
        tp3_hits = sum(1 for t in self.trades if any(e['type'] == 'TP3' for e in t['exits']))
        tp4_hits = sum(1 for t in self.trades if any(e['type'] == 'TP4' for e in t['exits']))
        sl_hits = sum(1 for t in self.trades if t['final_exit'] == 'SL')
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0
        return {
            'total_trades': total_trades, 'winning_trades': winning_trades,
            'losing_trades': losing_trades, 'breakeven': breakeven, 'win_rate': win_rate,
            'initial_balance': self.initial_balance, 'final_balance': self.balance,
            'total_return_usdt': self.balance - self.initial_balance,
            'total_return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'total_gross_pnl': total_gross_pnl, 'total_fees': total_fees,
            'total_net_pnl': total_net_pnl, 'avg_net_pnl': avg_net_pnl,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor, 'expectancy': expectancy,
            'exit_distribution': dict(exit_dist),
            'tp_hits': {'TP1': tp1_hits, 'TP2': tp2_hits, 'TP3': tp3_hits, 'TP4': tp4_hits, 'SL': sl_hits},
            'trades': self.trades}

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


def get_historical_data(symbol, timeframe, days):
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
            print("    Error fetching " + symbol + ": " + str(e))
            break
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_top_mexc_coins(limit=50):
    print("Fetching top " + str(limit) + " coins from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in ALL_EXCLUSIONS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 500000:
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]
        print("Fetched " + str(len(top_coins)) + " coins")
        print("Excluded: " + str(len(EXCLUDED_COINS)) + " user-specified + " + str(len(STABLECOINS)) + " stablecoins")
        return top_coins
    except Exception as e:
        print("Error fetching coins list: " + str(e))
        return []

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        return
    url = "https://api.telegram.org/bot" + TELEGRAM_TOKEN + "/sendMessage"
    payload = {'chat_id': CHANNEL_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram error: " + str(e))

def format_report_telegram(report, days_back, top_n):
    if 'error' in report:
        return "Backtest Error: " + report['error']
    tp = report['tp_hits']
    exit_dist = report['exit_distribution']
    msg = ("Backtest Report - " + str(days_back) + " Days

"
           "Assets: Top " + str(top_n) + " MEXC Coins
"
           "Timeframe: " + TIMEFRAME + "
"
           "Leverage: " + str(LEVERAGE) + "x
"
           "Initial: " + str(report['initial_balance']) + " USDT
"
           "Final: " + str(round(report['final_balance'], 2)) + " USDT

"
           "Total Trades: " + str(report['total_trades']) + "

"
           "Winners: " + str(report['winning_trades']) + " (" + str(round(report['win_rate'], 1)) + "%)
"
           "Losers: " + str(report['losing_trades']) + " (" + str(round(100-report['win_rate'], 1)) + "%)

"
           "TP Hits:
"
           "TP1 (" + str(TP1_PERC) + "%): " + str(tp['TP1']) + "
"
           "TP2 (" + str(TP2_PERC) + "%): " + str(tp['TP2']) + "
"
           "TP3 (" + str(TP3_PERC) + "%): " + str(tp['TP3']) + "
"
           "TP4 (" + str(TP4_PERC) + "%): " + str(tp['TP4']) + "

"
           "SL Hits: " + str(tp['SL']) + "

"
           "Total Gross PnL: " + ("+" if report['total_gross_pnl'] >= 0 else "") + str(round(report['total_gross_pnl'], 2)) + " USDT
"
           "Total Fees: " + str(round(report['total_fees'], 2)) + " USDT
"
           "Net Return: " + ("+" if report['total_return_pct'] >= 0 else "") + str(round(report['total_return_pct'], 2)) + "%
"
           "Max Drawdown: " + str(round(report['max_drawdown_pct'], 2)) + "%
"
           "Profit Factor: " + str(round(report['profit_factor'], 2)) + "
"
           "Expectancy: " + str(round(report['expectancy'], 2)) + " USDT

"
           "Fees: MEXC Futures API 0.06% (taker)")
    return msg


def run_full_backtest():
    print("=" * 70)
    print("ADVANCED SQUEEZE MOMENTUM - 1 MONTH BACKTEST")
    print("=" * 70)
    print("Period: Last " + str(DAYS_BACK) + " days")
    print("Capital: " + str(INITIAL_BALANCE) + " USDT | Risk: " + str(RISK_PER_TRADE*100) + "% | Leverage: " + str(LEVERAGE) + "x")
    print("TP1: " + str(TP1_PERC) + "% (50%) | TP2: " + str(TP2_PERC) + "% (25%) | TP3: " + str(TP3_PERC) + "% (10%) | TP4: " + str(TP4_PERC) + "% (15%)")
    print("SL: " + str(SL_PERC) + "% | After TP1: SL->+" + str(SL_MOVE_AFTER_TP1) + "%")
    print("Fee: " + str(FEE_RATE*100) + "% per side (MEXC API Futures)")
    print("Excluded: " + str(len(EXCLUDED_COINS)) + " coins")
    print("=" * 70)

    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins:
        print("Failed to get coin list. Aborting.")
        return None

    print("")
    print("Coins to test: " + ", ".join(top_coins[:5]) + "... (" + str(len(top_coins)) + " total)")

    indicator = SqueezeMomentumIndicator()
    all_signals = []

    print("")
    print("Fetching historical data (" + str(DAYS_BACK) + " days, " + TIMEFRAME + ")...")
    print("-" * 70)

    for i, symbol in enumerate(top_coins):
        print("[" + str(i+1).rjust(2) + "/" + str(len(top_coins)) + "] " + symbol.ljust(20) + " - ", end="", flush=True)
        try:
            df = get_historical_data(symbol, TIMEFRAME, DAYS_BACK)
            if df.empty or len(df) < 50:
                print("Insufficient data (" + str(len(df)) + " candles)")
                continue
            print("OK " + str(len(df)).rjust(5) + " candles - ", end="", flush=True)
            df_signals = indicator.generate_signals(df)
            signal_mask = df_signals['signal'] != 0
            signal_rows = df_signals[signal_mask].copy()
            if len(signal_rows) == 0:
                print("No signals")
                continue
            print(str(len(signal_rows)).rjust(3) + " signals")
            for idx, row in signal_rows.iterrows():
                future_mask = df_signals.index > idx
                df_future = df_signals[future_mask].copy()
                if len(df_future) < 10:
                    continue
                signal_type = int(row['signal'])
                entry_price = float(row['close'])
                all_signals.append({
                    'symbol': symbol, 'signal_time': idx,
                    'signal_type': signal_type, 'entry_price': entry_price,
                    'df_future': df_future.head(300)})
        except Exception as e:
            print("Error: " + str(e))
            continue

    print("")
    print("=" * 70)
    print("TOTAL SIGNALS COLLECTED: " + str(len(all_signals)))
    print("=" * 70)

    if len(all_signals) == 0:
        print("No signals to backtest!")
        return None

    engine = AdvancedBacktestEngine(INITIAL_BALANCE, RISK_PER_TRADE, LEVERAGE)
    report = engine.run_backtest(all_signals)

    print("")
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print("Total Trades:       " + str(report['total_trades']))
    print("Winning Trades:     " + str(report['winning_trades']) + " (" + str(round(report['win_rate'], 1)) + "%)")
    print("Losing Trades:      " + str(report['losing_trades']) + " (" + str(round(100-report['win_rate'], 1)) + "%)")
    print("Initial Balance:    " + str(round(report['initial_balance'], 2)) + " USDT")
    print("Final Balance:      " + str(round(report['final_balance'], 2)) + " USDT")
    print("Gross PnL:          " + ("+" if report['total_gross_pnl'] >= 0 else "") + str(round(report['total_gross_pnl'], 2)) + " USDT")
    print("Total Fees:         " + str(round(report['total_fees'], 2)) + " USDT")
    print("Net Return:         " + ("+" if report['total_return_pct'] >= 0 else "") + str(round(report['total_return_pct'], 2)) + "%")
    print("Avg Net PnL/Trade:  " + ("+" if report['avg_net_pnl'] >= 0 else "") + str(round(report['avg_net_pnl'], 2)) + " USDT")
    print("Avg Win:            " + ("+" if report['avg_win'] >= 0 else "") + str(round(report['avg_win'], 2)) + " USDT")
    print("Avg Loss:           " + ("+" if report['avg_loss'] >= 0 else "") + str(round(report['avg_loss'], 2)) + " USDT")
    print("Max Drawdown:       " + str(round(report['max_drawdown_pct'], 2)) + "%")
    print("Profit Factor:      " + str(round(report['profit_factor'], 2)))
    print("Expectancy:         " + str(round(report['expectancy'], 2)) + " USDT")
    print("")
    print("Exit Distribution:")
    for exit_type, count in report['exit_distribution'].items():
        print("  " + exit_type + ": " + str(count))
    print("")
    print("TP Hit Rates:")
    for tp, count in report['tp_hits'].items():
        print("  " + tp + ": " + str(count))
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = "backtest_advanced_" + timestamp + ".json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("")
    print("Report saved to: " + report_filename)

    if TELEGRAM_TOKEN and CHANNEL_ID:
        print("Sending report to Telegram...")
        msg = format_report_telegram(report, DAYS_BACK, len(top_coins))
        send_telegram_message(msg)
        print("Telegram sent!")

    return report, engine

if __name__ == "__main__":
    run_full_backtest()
