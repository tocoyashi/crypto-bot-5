"""
Backtest Script - Squeeze Momentum Strategy (v6)
- Synced with bot.py (exact same signal logic)
- Cooldown support (4h per coin)
- Top 25 coins
- CSV output
"""

import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ================= Settings (synced with bot.py) =================
TIMEFRAME = '15m'
TOP_N_COINS = 25
LEVERAGE = 10

# MEXC Futures Taker Fee
MEXC_TAKER_FEE = 0.0002  # 0.02%

# Excluded coins
EXCLUDED_COINS = [
    'SPACEX(PRE)/USDT', 'RAIN/USDT', 'TOYL/USDT', 'WXT/USDT',
    'UPC/USDT', 'DN/USDT', 'AIXPLAY/USDT', 'MBG/USDT',
    'KAZAR/USDT', 'STAR/USDT'
]

STABLECOINS = ['USDC/USDT', 'TUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'USDP/USDT', 'PYUSD/USDT']

# TP levels with partial close percentages (same logic as bot.py TP targets)
TP_LEVELS = [
    {'name': 'TP1', 'perc': 0.6,  'close_pct': 0.30},
    {'name': 'TP2', 'perc': 1.5,  'close_pct': 0.25},
    {'name': 'TP3', 'perc': 2.4,  'close_pct': 0.20},
    {'name': 'TP4', 'perc': 5.0,  'close_pct': 0.15},
    {'name': 'TP5', 'perc': 7.0,  'close_pct': 0.05},
    {'name': 'TP6', 'perc': 9.0,  'close_pct': 0.05},
]

SL_PERC = 6.0

# ================= Cooldown Settings (synced with bot.py) =================
COOLDOWN_HOURS = 4

# Backtest period
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=30)

_exchange = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.mexc({'enableRateLimit': True})
    return _exchange


# ================= Same indicator class as bot.py =================
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


def fetch_ohlcv(symbol, start_ms, end_ms):
    exchange = get_exchange()
    all_data = []
    since = start_ms

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= since:
                break
            since = last_ts + 1
        except Exception:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    start_dt = pd.to_datetime(start_ms, unit='ms')
    end_dt = pd.to_datetime(end_ms, unit='ms')
    df = df.loc[start_dt:end_dt]
    return df


def fetch_all_data_parallel(coins, start_ms, end_ms, max_workers=5):
    results = {}
    print(f"  Fetching data for {len(coins)} coins (parallel, {max_workers} threads)...")

    def _fetch(sym):
        try:
            df = fetch_ohlcv(sym, start_ms, end_ms)
            return sym, df
        except Exception:
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, s): s for s in coins}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, df = future.result()
            results[sym] = df
            if done % 10 == 0:
                print(f"    Fetched {done}/{len(coins)}...")

    valid = {k: v for k, v in results.items() if len(v) >= 100}
    print(f"  Valid data: {len(valid)}/{len(coins)} coins")
    return valid


def simulate_trade(entry_price, signal_type, highs, lows):
    direction = 1 if signal_type == 1 else -1

    tp_prices = np.array([
        entry_price * (1 + direction * tp['perc'] / 100) for tp in TP_LEVELS
    ])

    sl_price = entry_price * (1 - direction * SL_PERC / 100)

    position_usdt = 100.0
    remaining_pct = 1.0
    open_fee = position_usdt * MEXC_TAKER_FEE
    total_pnl = 0.0
    total_fees = open_fee
    tp_hits = [False] * len(TP_LEVELS)
    closed_amounts = []
    exit_reason = "Timeout"
    exit_idx = len(highs) - 1

    for i in range(len(highs)):
        # Check TPs
        for j in range(len(TP_LEVELS)):
            if tp_hits[j]:
                continue
            hit = (direction == 1 and highs[i] >= tp_prices[j]) or \
                  (direction == -1 and lows[i] <= tp_prices[j])
            if hit:
                tp_hits[j] = True
                cp = TP_LEVELS[j]['close_pct']
                cu = position_usdt * cp
                pnl = cu * direction * (tp_prices[j] - entry_price) / entry_price
                cf = cu * MEXC_TAKER_FEE
                total_fees += cf
                total_pnl += pnl - cf
                remaining_pct -= cp
                closed_amounts.append({
                    'tp': TP_LEVELS[j]['name'], 'price': tp_prices[j],
                    'pct_closed': cp, 'pnl': pnl - cf, 'fee': cf, 'idx': i
                })
                if remaining_pct <= 0.001:
                    exit_reason = f"TP{j+1} (All closed)"
                    exit_idx = i
                    return _result(exit_reason, i, total_pnl, total_fees, tp_hits, False, closed_amounts)

        # Check SL
        sl_hit = (direction == 1 and lows[i] <= sl_price) or \
                 (direction == -1 and highs[i] >= sl_price)
        if sl_hit and remaining_pct > 0.001:
            cu = position_usdt * remaining_pct
            pnl = cu * direction * (sl_price - entry_price) / entry_price
            cf = cu * MEXC_TAKER_FEE
            total_fees += cf
            total_pnl += pnl - cf
            closed_amounts.append({
                'tp': 'SL', 'price': sl_price,
                'pct_closed': remaining_pct, 'pnl': pnl - cf, 'fee': cf, 'idx': i
            })
            return _result("Stop Loss", i, total_pnl, total_fees, tp_hits, True, closed_amounts)

    # Timeout - close remaining at last price
    if remaining_pct > 0.001:
        last_p = (highs[-1] + lows[-1]) / 2
        cu = position_usdt * remaining_pct
        pnl = cu * direction * (last_p - entry_price) / entry_price
        cf = cu * MEXC_TAKER_FEE
        total_fees += cf
        total_pnl += pnl - cf
        closed_amounts.append({
            'tp': 'TIMEOUT', 'price': last_p,
            'pct_closed': remaining_pct, 'pnl': pnl - cf, 'fee': cf, 'idx': exit_idx
        })

    return _result(exit_reason, exit_idx, total_pnl, total_fees, tp_hits, False, closed_amounts)


def _result(reason, idx, pnl, fees, tp_hits, sl_hit, details):
    return {
        'exit_reason': reason, 'exit_idx': idx,
        'total_pnl': pnl, 'total_fees': fees, 'net_pnl': pnl,
        'tp_hits': sum(tp_hits), 'sl_hit': sl_hit,
        'tp_details': details
    }


def process_coin(symbol, df, indicator):
    trades = []
    try:
        df_sig = indicator.generate_signals(df)
        signals = df_sig[df_sig['signal'] != 0].index

        if len(df_sig) < 60:
            return trades

        cutoff = df_sig.index[-30]
        signals = signals[signals < cutoff]

        cooldown_until = None
        cooldown_skipped = 0

        for sig_time in signals:
            # Cooldown check
            if cooldown_until is not None and sig_time < cooldown_until:
                cooldown_skipped += 1
                continue

            sig_idx = df_sig.index.get_loc(sig_time)
            sig_type = df_sig.loc[sig_time, 'signal']
            entry = df_sig.loc[sig_time, 'close']

            if sig_idx + 60 >= len(df_sig):
                continue

            future = df_sig.iloc[sig_idx + 1: sig_idx + 600]
            if len(future) < 60:
                continue

            highs = future['high'].values
            lows = future['low'].values

            result = simulate_trade(entry, sig_type, highs, lows)

            # Set cooldown after taking a trade
            cooldown_until = sig_time + timedelta(hours=COOLDOWN_HOURS)

            trade = {
                'symbol': symbol,
                'signal_type': 'BUY' if sig_type == 1 else 'SELL',
                'entry_timestamp': str(sig_time),
                'entry_price': entry,
                'exit_timestamp': str(future.index[min(result['exit_idx'], len(future)-1)]),
                'exit_reason': result['exit_reason'],
                'tp_hits': result['tp_hits'],
                'sl_hit': result['sl_hit'],
                'total_pnl_pct': (result['net_pnl'] / 100) * 100 * LEVERAGE,
                'total_fees': result['total_fees'],
                'net_pnl': result['net_pnl'],
            }

            for k in range(len(TP_LEVELS)):
                tp_name = TP_LEVELS[k]['name']
                tp_match = [d for d in result['tp_details'] if d['tp'] == tp_name]
                trade[f'{tp_name}_price'] = tp_match[0]['price'] if tp_match else None
                trade[f'{tp_name}_pnl'] = tp_match[0]['pnl'] if tp_match else None

            trades.append(trade)

    except Exception as e:
        print(f"    [WARN] {symbol}: {e}")

    return trades


def run_backtest():
    t0 = time.time()
    print("=" * 60)
    print("  BACKTEST v6 - Synced with bot.py")
    print("=" * 60)
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x")
    print(f"  Coins: Top {TOP_N_COINS}")
    print(f"  SL: {SL_PERC}% (matches bot.py)")
    tp_info = ", ".join([f"{tp['name']}({tp['perc']}%/{int(tp['close_pct']*100)}%)" for tp in TP_LEVELS])
    print(f"  TP: {tp_info}")
    print(f"  MEXC Fee: {MEXC_TAKER_FEE*100:.3f}% (taker)")
    print(f"  Cooldown: {COOLDOWN_HOURS}h per coin (matches bot.py)")
    print(f"  Signal logic: Exact copy from bot.py (no extra filters)")
    print(f"  Excluded: {len(EXCLUDED_COINS)} coins")
    print("=" * 60)

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)

    print("\n[1/3] Fetching top coins...")
    exchange = get_exchange()
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS and symbol not in EXCLUDED_COINS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 1000000:
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})
        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [p['symbol'] for p in usdt_pairs[:TOP_N_COINS]]
        print(f"  Found {len(top_coins)} coins")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    print(f"\n[2/3] Fetching OHLCV data (parallel)...")
    data_map = fetch_all_data_parallel(top_coins, start_ms, end_ms, max_workers=8)
    if not data_map:
        print("No valid data. Aborting.")
        return

    print(f"\n[3/3] Running signals & simulation...")
    indicator = SqueezeMomentumIndicator()
    all_trades = []
    processed = 0

    for symbol, df in data_map.items():
        processed += 1
        coin_trades = process_coin(symbol, df, indicator)
        all_trades.extend(coin_trades)
        if processed % 10 == 0 or processed == len(data_map):
            print(f"  Processed {processed}/{len(data_map)} | Signals: {len(all_trades)}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    if not all_trades:
        print("\nNo trades found.")
        return

    df_trades = pd.DataFrame(all_trades)

    total_trades = len(df_trades)
    winning = len(df_trades[df_trades['net_pnl'] > 0])
    losing = len(df_trades[df_trades['net_pnl'] <= 0])
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    total_pnl = df_trades['net_pnl'].sum()
    total_fees = df_trades['total_fees'].sum()
    avg_pnl = df_trades['net_pnl'].mean()
    avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning > 0 else 0
    avg_loss = df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean() if losing > 0 else 0
    max_win = df_trades['net_pnl'].max()
    max_loss = df_trades['net_pnl'].min()
    gross_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_win / gross_loss if gross_loss != 0 else float('inf')

    exit_counts = df_trades['exit_reason'].value_counts()
    sl_count = int(df_trades['sl_hit'].sum())

    buy_t = df_trades[df_trades['signal_type'] == 'BUY']
    sell_t = df_trades[df_trades['signal_type'] == 'SELL']

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Trades:        {total_trades}")
    print(f"  Winning:             {winning}  |  Losing: {losing}")
    print(f"  Win Rate:            {win_rate:.1f}%")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Total Net PnL:       ${total_pnl:.2f}")
    print(f"  Total Fees:          ${total_fees:.2f}")
    print(f"  Avg PnL/Trade:       ${avg_pnl:.2f}")
    print(f"  Avg Win:             ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print(f"  Max Win:             ${max_win:.2f}  |  Max Loss: ${max_loss:.2f}")
    print(f"  ROI (100$/trade):    {total_pnl / (total_trades * 100) * 100:.2f}%")
    print(f"  ROI (leveraged):     {total_pnl / (total_trades * 100) * 100 * LEVERAGE:.2f}%")
    print("-" * 60)
    for k in range(len(TP_LEVELS)):
        tp_name = TP_LEVELS[k]['name']
        tp_hits = len(df_trades[df_trades['tp_hits'] >= k + 1])
        print(f"  {tp_name} ({TP_LEVELS[k]['perc']}%):  {tp_hits}/{total_trades} ({tp_hits/total_trades*100:.1f}%)")
    print(f"  Stop Loss:           {sl_count}/{total_trades} ({sl_count/total_trades*100:.1f}%)")
    print("-" * 60)
    print(f"  BUY:  {len(buy_t)} trades (Avg: ${buy_t['net_pnl'].mean():.2f})" if len(buy_t) else "  BUY:  0 trades")
    print(f"  SELL: {len(sell_t)} trades (Avg: ${sell_t['net_pnl'].mean():.2f})" if len(sell_t) else "  SELL: 0 trades")
    print("-" * 60)
    print("  Exit Reasons:")
    for r, c in exit_counts.items():
        print(f"    {r}: {c}")
    print("=" * 60)

    output_path = 'backtest_results.csv'
    df_trades.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    run_backtest()
