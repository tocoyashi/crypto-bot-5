"""
Backtest Script - Squeeze Momentum Strategy (Optimized v2)
- Fixed datetime slicing
- Filtered false signals (volume, momentum strength, squeeze duration, ATR)
- Parallel data fetching for speed
- Vectorized trade simulation
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

# ================= Settings =================
TIMEFRAME = '15m'
TOP_N_COINS = 50
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

# TP levels with partial close percentages
TP_LEVELS = [
    {'name': 'TP1', 'perc': 0.8,  'close_pct': 0.50},
    {'name': 'TP2', 'perc': 1.6,  'close_pct': 0.25},
    {'name': 'TP3', 'perc': 3.0,  'close_pct': 0.10},
    {'name': 'TP4', 'perc': 6.0,  'close_pct': 0.15},
]

SL_PERC = 6.0
SL_AFTER_TP1 = 0.10

# ================= False Signal Filters =================
MIN_SQUEEZE_BARS = 5          # Minimum squeeze duration (candles) before release
MIN_MOMENTUM_STRENGTH = 0.0   # Minimum |momentum| value to accept signal
MIN_VOLUME_RATIO = 1.2        # Signal candle volume must be > X * avg volume(20)
MIN_ATR_PERCENTILE = 10       # ATR must be above this percentile (avoid dead markets)

# Backtest period
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=30)

# Exchange instance (reuse for all calls)
_exchange = None


def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.mexc({'enableRateLimit': True})
    return _exchange


class SqueezeMomentumIndicator:
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def calculate_indicators(self, df):
        data = df.copy()
        c, h, l = data['close'], data['high'], data['low']

        # Bollinger Bands
        bb_basis = c.rolling(self.bb_length).mean()
        bb_dev = self.bb_mult * c.rolling(self.bb_length).std()
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev

        # Keltner Channels
        kc_ma = c.rolling(self.kc_length).mean()
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        range_ma = tr.rolling(self.kc_length).mean()
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult

        # Squeeze
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

        # Squeeze duration: count consecutive squeeze bars
        squeeze_duration = (~squeeze_on).cumsum()
        squeeze_bars = squeeze_on.groupby(squeeze_duration).cumsum()

        # Momentum (linear regression on close - avg)
        hh = h.rolling(self.kc_length).max()
        ll = l.rolling(self.kc_length).min()
        cm = c.rolling(self.kc_length).mean()
        avg_val = ((hh + ll) / 2 + cm) / 2
        diff = c - avg_val

        # Vectorized linear regression
        def _linreg(s, w):
            x = np.arange(w)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            def _apply(y):
                if len(y) < w or np.any(np.isnan(y)):
                    return np.nan
                y_c = y - y.mean()
                return np.dot(x - x_mean, y_c) / x_var * (w - 1) + y_c[-1]
            return s.rolling(w).apply(_apply, raw=True)

        momentum = _linreg(diff, self.kc_length)

        # ATR
        atr = tr.rolling(14).mean()
        atr_pct = atr / c * 100  # ATR as % of price

        # Volume ratio
        vol_ma = data['volume'].rolling(20).mean()
        vol_ratio = data['volume'] / vol_ma

        data['squeeze_on'] = squeeze_on
        data['squeeze_bars'] = squeeze_bars
        data['squeeze_release'] = squeeze_on.shift(1).fillna(False) & ~squeeze_on
        data['momentum'] = momentum
        data['momentum_increasing'] = momentum > momentum.shift(1)
        data['atr_pct'] = atr_pct
        data['atr_pct_rank'] = atr_pct.rolling(100).rank(pct=True)
        data['vol_ratio'] = vol_ratio

        return data

    def generate_signals(self, df):
        data = self.calculate_indicators(df)
        data['signal'] = 0

        sq = data['squeeze_release']
        mom = data['momentum']
        mi = data['momentum_increasing'].fillna(False)
        sq_bars = data['squeeze_bars'].shift(1).fillna(0)
        atr_rank = data['atr_pct_rank']
        vol_r = data['vol_ratio']

        # BUY: squeeze release + positive momentum + increasing + filters
        buy_cond = (
            sq &
            (mom > MIN_MOMENTUM_STRENGTH) &
            mi &
            (sq_bars >= MIN_SQUEEZE_BARS) &
            (atr_rank > MIN_ATR_PERCENTILE / 100) &
            (vol_r > MIN_VOLUME_RATIO)
        )

        # SELL: momentum crosses below zero OR momentum stops increasing (with filters)
        sell_cond = (
            (
                (mom < -MIN_MOMENTUM_STRENGTH) & (mom.shift(1).fillna(0) >= 0)
            ) |
            (
                ~mi & ~mi.shift(1).fillna(True) & (mom > 0)
            )
        ) & (atr_rank > MIN_ATR_PERCENTILE / 100) & (vol_r > MIN_VOLUME_RATIO)

        data.loc[buy_cond, 'signal'] = 1
        data.loc[sell_cond, 'signal'] = -1
        return data


def fetch_ohlcv(symbol, start_ms, end_ms):
    """Fetch all candles between start and end, paginating."""
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
    """Fetch data for all coins in parallel."""
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


def simulate_trade_vectorized(entry_price, signal_type, highs, lows):
    """Fast vectorized trade simulation using numpy arrays."""
    direction = 1 if signal_type == 1 else -1

    tp_prices = np.array([
        entry_price * (1 + direction * tp['perc'] / 100) for tp in TP_LEVELS
    ])

    sl_price = entry_price * (1 - direction * SL_PERC / 100)
    new_sl = entry_price * (1 + direction * SL_AFTER_TP1 / 100)

    position_usdt = 100.0
    remaining_pct = 1.0
    open_fee = position_usdt * MEXC_TAKER_FEE
    total_pnl = 0.0
    total_fees = open_fee
    tp_hits = [False] * 4
    sl_moved = False
    closed_amounts = []
    exit_reason = "Timeout"
    exit_idx = len(highs) - 1

    for i in range(len(highs)):
        # Check TPs (order matters)
        for j in range(4):
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
                if j == 0 and not sl_moved:
                    sl_moved = True
                    sl_price = new_sl
                if remaining_pct <= 0.001:
                    exit_reason = f"TP{j+1} (All closed)"
                    exit_idx = i
                    return _result(exit_reason, i, total_pnl, total_fees, tp_hits, False, sl_moved, closed_amounts)

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
            return _result("Stop Loss", i, total_pnl, total_fees, tp_hits, True, sl_moved, closed_amounts)

    # Timeout
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

    return _result(exit_reason, exit_idx, total_pnl, total_fees, tp_hits, False, sl_moved, closed_amounts)


def _result(reason, idx, pnl, fees, tp_hits, sl_hit, sl_moved, details):
    return {
        'exit_reason': reason, 'exit_idx': idx,
        'total_pnl': pnl, 'total_fees': fees, 'net_pnl': pnl,
        'tp_hits': sum(tp_hits), 'sl_hit': sl_hit,
        'sl_moved': sl_moved, 'tp_details': details
    }


def process_coin(symbol, df, indicator, all_trades):
    """Process a single coin and return trades."""
    trades = []
    try:
        df_sig = indicator.generate_signals(df)
        signals = df_sig[df_sig['signal'] != 0].index

        # Exclude last 30 candles (not enough room for trade)
        if len(df_sig) < 60:
            return trades

        cutoff = df_sig.index[-30]
        signals = signals[signals < cutoff]

        last_trade_time = pd.Timestamp.min

        for sig_time in signals:
            sig_idx = df_sig.index.get_loc(sig_time)
            sig_type = df_sig.loc[sig_time, 'signal']
            entry = df_sig.loc[sig_time, 'close']

            # Skip if too close to last trade on this coin (min 30 candles = 7.5h)
            if (sig_time - last_trade_time).total_seconds() < 30 * 15 * 60:
                continue

            # Need at least 60 future candles
            if sig_idx + 60 >= len(df_sig):
                continue

            future = df_sig.iloc[sig_idx + 1: sig_idx + 600]  # max ~6 days look-ahead
            if len(future) < 60:
                continue

            highs = future['high'].values
            lows = future['low'].values

            result = simulate_trade_vectorized(entry, sig_type, highs, lows)

            trade = {
                'symbol': symbol,
                'signal_type': 'BUY' if sig_type == 1 else 'SELL',
                'entry_timestamp': sig_time,
                'entry_price': entry,
                'exit_timestamp': future.index[min(result['exit_idx'], len(future)-1)],
                'exit_reason': result['exit_reason'],
                'tp_hits': result['tp_hits'],
                'sl_hit': result['sl_hit'],
                'sl_moved': result['sl_moved'],
                'total_pnl_pct': (result['net_pnl'] / 100) * 100 * LEVERAGE,
                'total_fees': result['total_fees'],
                'net_pnl': result['net_pnl'],
            }

            for k in range(4):
                tp_name = TP_LEVELS[k]['name']
                tp_match = [d for d in result['tp_details'] if d['tp'] == tp_name]
                trade[f'{tp_name}_price'] = tp_match[0]['price'] if tp_match else None
                trade[f'{tp_name}_pnl'] = tp_match[0]['pnl'] if tp_match else None

            trades.append(trade)
            last_trade_time = sig_time

    except Exception as e:
        print(f"    [WARN] {symbol}: {e}")

    return trades


def run_backtest():
    t0 = time.time()
    print("=" * 60)
    print("  BACKTEST v2 - Squeeze Momentum (Optimized)")
    print("=" * 60)
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x")
    print(f"  TP: 0.8%(50%), 1.6%(25%), 3%(10%), 6%(15%)")
    print(f"  SL: {SL_PERC}% -> +{SL_AFTER_TP1}% after TP1")
    print(f"  MEXC Fee: {MEXC_TAKER_FEE*100:.3f}% (taker)")
    print(f"  Filters: squeeze>={MIN_SQUEEZE_BARS}bars, vol>{MIN_VOLUME_RATIO}x, ATR>{MIN_ATR_PERCENTILE}%ile")
    print(f"  Excluded: {len(EXCLUDED_COINS)} coins")
    print("=" * 60)

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)

    # 1) Get top coins
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

    # 2) Fetch all data in parallel
    print(f"\n[2/3] Fetching OHLCV data (parallel)...")
    data_map = fetch_all_data_parallel(top_coins, start_ms, end_ms, max_workers=8)
    if not data_map:
        print("No valid data. Aborting.")
        return

    # 3) Generate signals and simulate trades
    print(f"\n[3/3] Running signals & simulation...")
    indicator = SqueezeMomentumIndicator()
    all_trades = []
    processed = 0

    for symbol, df in data_map.items():
        processed += 1
        coin_trades = process_coin(symbol, df, indicator, all_trades)
        all_trades.extend(coin_trades)
        if processed % 10 == 0 or processed == len(data_map):
            print(f"  Processed {processed}/{len(data_map)} | Signals so far: {len(all_trades)}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    if not all_trades:
        print("\nNo trades found.")
        return

    # ==================== Results ====================
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
    tp1_hits = len(df_trades[df_trades['tp_hits'] >= 1])
    tp2_hits = len(df_trades[df_trades['tp_hits'] >= 2])
    tp3_hits = len(df_trades[df_trades['tp_hits'] >= 3])
    tp4_hits = len(df_trades[df_trades['tp_hits'] >= 4])
    sl_count = int(df_trades['sl_hit'].sum())

    buy_t = df_trades[df_trades['signal_type'] == 'BUY']
    sell_t = df_trades[df_trades['signal_type'] == 'SELL']

    # ==================== Print ====================
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
    print(f"  TP1 (0.8%):          {tp1_hits}/{total_trades} ({tp1_hits/total_trades*100:.1f}%)")
    print(f"  TP2 (1.6%):          {tp2_hits}/{total_trades} ({tp2_hits/total_trades*100:.1f}%)")
    print(f"  TP3 (3.0%):          {tp3_hits}/{total_trades} ({tp3_hits/total_trades*100:.1f}%)")
    print(f"  TP4 (6.0%):          {tp4_hits}/{total_trades} ({tp4_hits/total_trades*100:.1f}%)")
    print(f"  Stop Loss:           {sl_count}/{total_trades} ({sl_count/total_trades*100:.1f}%)")
    print("-" * 60)
    print(f"  BUY:  {len(buy_t)} trades (Avg: ${buy_t['net_pnl'].mean():.2f})")
    print(f"  SELL: {len(sell_t)} trades (Avg: ${sell_t['net_pnl'].mean():.2f})")
    print("-" * 60)
    print("  Exit Reasons:")
    for r, c in exit_counts.items():
        print(f"    {r}: {c}")
    print("=" * 60)

    # ==================== Save Excel ====================
    output_path = 'backtest_results.xlsx'

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        t_out = df_trades.copy()
        t_out['entry_timestamp'] = t_out['entry_timestamp'].astype(str)
        t_out['exit_timestamp'] = t_out['exit_timestamp'].astype(str)
        t_out.to_excel(writer, sheet_name='Trades', index=False)

        summary = pd.DataFrame({
            'Metric': [
                'Period', 'Timeframe', 'Leverage',
                'Total Trades', 'Winning', 'Losing', 'Win Rate (%)',
                'Profit Factor', 'Net PnL ($)', 'Fees ($)',
                'Avg PnL ($)', 'Avg Win ($)', 'Avg Loss ($)',
                'Max Win ($)', 'Max Loss ($)',
                'ROI (%)', 'ROI Leverage (%)',
                'TP1 Hit', 'TP2 Hit', 'TP3 Hit', 'TP4 Hit', 'SL Hit',
                'BUY Trades', 'SELL Trades', 'BUY Avg ($)', 'SELL Avg ($)',
                'Fee (%)', 'Speed (s)',
                'Filter: Min Squeeze Bars', 'Filter: Min Vol Ratio',
                'Filter: Min ATR Percentile',
            ],
            'Value': [
                f"{START_DATE.strftime('%Y-%m-%d')} ~ {END_DATE.strftime('%Y-%m-%d')}",
                TIMEFRAME, f"{LEVERAGE}x",
                total_trades, winning, losing, f"{win_rate:.1f}",
                f"{profit_factor:.2f}", f"${total_pnl:.2f}", f"${total_fees:.2f}",
                f"${avg_pnl:.2f}", f"${avg_win:.2f}", f"${avg_loss:.2f}",
                f"${max_win:.2f}", f"${max_loss:.2f}",
                f"{total_pnl/(total_trades*100)*100:.2f}",
                f"{total_pnl/(total_trades*100)*100*LEVERAGE:.2f}",
                f"{tp1_hits}/{total_trades} ({tp1_hits/total_trades*100:.1f}%)",
                f"{tp2_hits}/{total_trades} ({tp2_hits/total_trades*100:.1f}%)",
                f"{tp3_hits}/{total_trades} ({tp3_hits/total_trades*100:.1f}%)",
                f"{tp4_hits}/{total_trades} ({tp4_hits/total_trades*100:.1f}%)",
                f"{sl_count}/{total_trades} ({sl_count/total_trades*100:.1f}%)",
                len(buy_t), len(sell_t),
                f"${buy_t['net_pnl'].mean():.2f}" if len(buy_t) else "$0",
                f"${sell_t['net_pnl'].mean():.2f}" if len(sell_t) else "$0",
                f"{MEXC_TAKER_FEE*100:.3f}%", f"{elapsed:.1f}",
                MIN_SQUEEZE_BARS, MIN_VOLUME_RATIO, MIN_ATR_PERCENTILE,
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

        pd.DataFrame(
            exit_counts.reset_index().values,
            columns=['Exit Reason', 'Count']
        ).to_excel(writer, sheet_name='Exit Reasons', index=False)

        pd.DataFrame({
            'Setting': [
                'TP1 (%)', 'TP1 Close', 'TP2 (%)', 'TP2 Close',
                'TP3 (%)', 'TP3 Close', 'TP4 (%)', 'TP4 Close',
                'SL (%)', 'SL After TP1 (%)', 'Leverage', 'Fee (%)',
                'Position ($)', 'Min Squeeze Bars', 'Min Vol Ratio', 'Min ATR %ile',
            ],
            'Value': [
                '0.8', '50%', '1.6', '25%', '3.0', '10%', '6.0', '15%',
                SL_PERC, SL_AFTER_TP1, f"{LEVERAGE}x", f"{MEXC_TAKER_FEE*100:.3f}",
                '100', MIN_SQUEEZE_BARS, MIN_VOLUME_RATIO, MIN_ATR_PERCENTILE,
            ]
        }).to_excel(writer, sheet_name='Settings', index=False)

    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    run_backtest()
