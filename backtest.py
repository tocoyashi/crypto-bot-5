"""
Backtest Script - Squeeze Momentum Strategy
Modified: 4 TP levels, partial closes, trailing SL after TP1, MEXC fees
Period: 1 month (15m timeframe)
"""

import pandas as pd
import numpy as np
from scipy import stats
import ccxt
import time
from datetime import datetime, timedelta
import warnings
import os
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

SL_PERC = 6.0          # Initial stop loss %
SL_AFTER_TP1 = 0.10    # Move SL to entry + 0.10% after TP1 hit

# Backtest period: last 1 month
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=30)


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


def fetch_ohlcv(symbol, timeframe, start_ms, end_ms):
    """Fetch all candles between start and end, paginating."""
    exchange = ccxt.mexc({'enableRateLimit': True})
    all_data = []
    since = start_ms
    limit = 1000

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= since:
                break
            since = last_ts + 1
            time.sleep(0.15)
        except Exception as e:
            print(f"  [ERROR] Fetching {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df = df.loc[start_ms:end_ms]
    return df


def get_top_mexc_coins(limit=50):
    """Fetch top N coins by 24h volume."""
    print(f"Fetching top {limit} coins by volume from MEXC...")
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and symbol not in STABLECOINS and symbol not in EXCLUDED_COINS:
                vol = ticker.get('quoteVolume') or 0
                if vol > 1000000:
                    usdt_pairs.append({'symbol': symbol, 'volume': vol})

        usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_coins = [pair['symbol'] for pair in usdt_pairs[:limit]]
        print(f"  Found {len(top_coins)} coins (excluded {len(EXCLUDED_COINS)} coins)")
        return top_coins
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []


def simulate_trade(entry_price, signal_type, future_candles):
    """
    Simulate a single trade with partial closes and MEXC fees.
    Returns trade result dict.
    """
    direction = 1 if signal_type == 1 else -1  # 1=long, -1=short

    # Calculate TP and SL prices
    tp_prices = []
    for tp in TP_LEVELS:
        if direction == 1:
            tp_prices.append(entry_price * (1 + tp['perc'] / 100))
        else:
            tp_prices.append(entry_price * (1 - tp['perc'] / 100))

    # Initial SL
    if direction == 1:
        sl_price = entry_price * (1 - SL_PERC / 100)
    else:
        sl_price = entry_price * (1 + SL_PERC / 100)

    # New SL after TP1 (entry + 0.10% in profit direction)
    if direction == 1:
        new_sl = entry_price * (1 + SL_AFTER_TP1 / 100)
    else:
        new_sl = entry_price * (1 - SL_AFTER_TP1 / 100)

    # Assume 100 USDT position per trade
    position_usdt = 100.0
    remaining_pct = 1.0  # 100% remaining

    # Opening fee
    open_fee = position_usdt * MEXC_TAKER_FEE

    total_pnl = 0.0
    total_fees = open_fee
    tp_hits = [False, False, False, False]
    sl_moved = False
    exit_reason = ""
    exit_price = 0.0
    closed_amounts = []

    for i, candle in future_candles.iterrows():
        high = candle['high']
        low = candle['low']
        close = candle['close']
        timestamp = i

        # Check each TP level
        for j in range(4):
            if tp_hits[j]:
                continue

            tp_hit = False
            if direction == 1:
                if high >= tp_prices[j]:
                    tp_hit = True
                    exit_price = tp_prices[j]
            else:
                if low <= tp_prices[j]:
                    tp_hit = True
                    exit_price = tp_prices[j]

            if tp_hit:
                tp_hits[j] = True
                close_pct = TP_LEVELS[j]['close_pct']
                closed_usdt = position_usdt * close_pct

                # PnL for this close
                if direction == 1:
                    pnl = closed_usdt * ((exit_price - entry_price) / entry_price)
                else:
                    pnl = closed_usdt * ((entry_price - exit_price) / entry_price)

                # Close fee
                close_fee = closed_usdt * MEXC_TAKER_FEE
                total_fees += close_fee
                total_pnl += pnl - close_fee
                remaining_pct -= close_pct
                closed_amounts.append({
                    'tp': TP_LEVELS[j]['name'],
                    'price': exit_price,
                    'pct_closed': close_pct,
                    'pnl': pnl - close_fee,
                    'fee': close_fee,
                    'timestamp': timestamp
                })

                # Move SL after TP1
                if j == 0 and not sl_moved:
                    sl_moved = True
                    sl_price = new_sl

                if remaining_pct <= 0.001:
                    exit_reason = f"TP{TP_LEVELS[j]['name'][-1]} (All closed)"
                    return {
                        'exit_reason': exit_reason,
                        'exit_timestamp': timestamp,
                        'total_pnl': total_pnl,
                        'total_fees': total_fees,
                        'net_pnl': total_pnl - 0,  # open fee already deducted
                        'tp_hits': sum(tp_hits),
                        'sl_hit': False,
                        'tp_details': closed_amounts
                    }

        # Check SL
        sl_hit = False
        if direction == 1:
            if low <= sl_price:
                sl_hit = True
                exit_price = sl_price
        else:
            if high >= sl_price:
                sl_hit = True
                exit_price = sl_price

        if sl_hit and remaining_pct > 0.001:
            closed_usdt = position_usdt * remaining_pct
            if direction == 1:
                pnl = closed_usdt * ((exit_price - entry_price) / entry_price)
            else:
                pnl = closed_usdt * ((entry_price - exit_price) / entry_price)

            close_fee = closed_usdt * MEXC_TAKER_FEE
            total_fees += close_fee
            total_pnl += pnl - close_fee
            closed_amounts.append({
                'tp': 'SL',
                'price': exit_price,
                'pct_closed': remaining_pct,
                'pnl': pnl - close_fee,
                'fee': close_fee,
                'timestamp': timestamp
            })

            exit_reason = "Stop Loss"
            return {
                'exit_reason': exit_reason,
                'exit_timestamp': timestamp,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'net_pnl': total_pnl,
                'tp_hits': sum(tp_hits),
                'sl_hit': True,
                'sl_moved': sl_moved,
                'tp_details': closed_amounts
            }

    # Trade didn't close within data - force close at last price
    if remaining_pct > 0.001:
        last_price = future_candles.iloc[-1]['close']
        closed_usdt = position_usdt * remaining_pct
        if direction == 1:
            pnl = closed_usdt * ((last_price - entry_price) / entry_price)
        else:
            pnl = closed_usdt * ((entry_price - last_price) / entry_price)

        close_fee = closed_usdt * MEXC_TAKER_FEE
        total_fees += close_fee
        total_pnl += pnl - close_fee
        closed_amounts.append({
            'tp': 'TIMEOUT',
            'price': last_price,
            'pct_closed': remaining_pct,
            'pnl': pnl - close_fee,
            'fee': close_fee,
            'timestamp': future_candles.index[-1]
        })
        exit_reason = "Timeout (force close)"

    return {
        'exit_reason': exit_reason,
        'exit_timestamp': future_candles.index[-1],
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'net_pnl': total_pnl,
        'tp_hits': sum(tp_hits),
        'sl_hit': False,
        'sl_moved': sl_moved,
        'tp_details': closed_amounts
    }


def run_backtest():
    print("=" * 60)
    print("  BACKTEST - Squeeze Momentum Strategy (1 Month)")
    print("=" * 60)
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x")
    print(f"  TP: 0.8%(50%), 1.6%(25%), 3%(10%), 6%(15%)")
    print(f"  SL: {SL_PERC}% -> +{SL_AFTER_TP1}% after TP1")
    print(f"  MEXC Fee: {MEXC_TAKER_FEE*100:.3f}% (taker)")
    print(f"  Excluded: {EXCLUDED_COINS}")
    print("=" * 60)

    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)

    # Get top coins
    top_coins = get_top_mexc_coins(TOP_N_COINS)
    if not top_coins:
        print("No coins found. Aborting.")
        return

    indicator = SqueezeMomentumIndicator()
    all_trades = []
    processed = 0

    for symbol in top_coins:
        processed += 1
        print(f"  [{processed}/{len(top_coins)}] Processing {symbol}...", end=" ", flush=True)

        try:
            df = fetch_ohlcv(symbol, TIMEFRAME, start_ms, end_ms)
            if len(df) < 100:
                print("Not enough data")
                continue

            df_signals = indicator.generate_signals(df)

            # Find all signals (excluding last 20 candles to allow trade room)
            signal_indices = df_signals[df_signals['signal'] != 0].index[:-20]

            for sig_time in signal_indices:
                sig_idx = df_signals.index.get_loc(sig_time)
                signal_type = df_signals.loc[sig_time, 'signal']
                entry_price = df_signals.loc[sig_time, 'close']

                # Future candles after signal
                future = df_signals.iloc[sig_idx + 1:]

                if len(future) < 10:
                    continue

                # Check we're not already in a trade on this coin
                # (simple: skip if signal within 20 candles of last trade on this coin)
                too_close = False
                for t in all_trades:
                    if t['symbol'] == symbol:
                        time_diff = abs((sig_time - t['entry_timestamp']).total_seconds())
                        if time_diff < 20 * 15 * 60:  # 20 candles * 15min
                            too_close = True
                            break
                if too_close:
                    continue

                result = simulate_trade(entry_price, signal_type, future)

                trade = {
                    'symbol': symbol,
                    'signal_type': 'BUY' if signal_type == 1 else 'SELL',
                    'entry_timestamp': sig_time,
                    'entry_price': entry_price,
                    'exit_timestamp': result['exit_timestamp'],
                    'exit_reason': result['exit_reason'],
                    'tp_hits': result['tp_hits'],
                    'sl_hit': result['sl_hit'],
                    'sl_moved': result.get('sl_moved', False),
                    'total_pnl_pct': (result['net_pnl'] / 100) * 100 * LEVERAGE,  # leveraged %
                    'total_fees': result['total_fees'],
                    'net_pnl': result['net_pnl'],
                }

                # Add TP details
                for k in range(4):
                    tp_name = TP_LEVELS[k]['name']
                    if k < len(result['tp_details']) and result['tp_details'][k]['tp'] == tp_name:
                        trade[f'{tp_name}_price'] = result['tp_details'][k]['price']
                        trade[f'{tp_name}_pnl'] = result['tp_details'][k]['pnl']
                    else:
                        trade[f'{tp_name}_price'] = None
                        trade[f'{tp_name}_pnl'] = None

                all_trades.append(trade)

            print(f"Done (signals found so far: {len(all_trades)})")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        time.sleep(0.3)

    if not all_trades:
        print("\nNo trades found in the backtest period.")
        return

    # Build results
    df_trades = pd.DataFrame(all_trades)

    # ==================== Summary Statistics ====================
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
    losing_trades = len(df_trades[df_trades['net_pnl'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_pnl = df_trades['net_pnl'].sum()
    total_fees_paid = df_trades['total_fees'].sum()
    avg_pnl = df_trades['net_pnl'].mean()
    avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
    max_win = df_trades['net_pnl'].max()
    max_loss = df_trades['net_pnl'].min()
    profit_factor = abs(df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum() / df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum()) if df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum() != 0 else float('inf')

    # Exit reason stats
    exit_counts = df_trades['exit_reason'].value_counts()

    # TP hit stats
    tp1_hits = len(df_trades[df_trades['tp_hits'] >= 1])
    tp2_hits = len(df_trades[df_trades['tp_hits'] >= 2])
    tp3_hits = len(df_trades[df_trades['tp_hits'] >= 3])
    tp4_hits = len(df_trades[df_trades['tp_hits'] >= 4])
    sl_count = df_trades['sl_hit'].sum()

    # By signal type
    buy_trades = df_trades[df_trades['signal_type'] == 'BUY']
    sell_trades = df_trades[df_trades['signal_type'] == 'SELL']

    # ==================== Print Summary ====================
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:          {total_trades}")
    print(f"  Winning Trades:        {winning_trades}")
    print(f"  Losing Trades:         {losing_trades}")
    print(f"  Win Rate:              {win_rate:.1f}%")
    print(f"  Profit Factor:         {profit_factor:.2f}")
    print(f"  Total Net PnL:         ${total_pnl:.2f}")
    print(f"  Total Fees Paid:       ${total_fees_paid:.2f}")
    print(f"  Avg PnL per Trade:     ${avg_pnl:.2f}")
    print(f"  Avg Win:               ${avg_win:.2f}")
    print(f"  Avg Loss:              ${avg_loss:.2f}")
    print(f"  Max Win:               ${max_win:.2f}")
    print(f"  Max Loss:              ${max_loss:.2f}")
    print(f"  ROI (on 100$/trade):   {total_pnl / (total_trades * 100) * 100:.2f}%")
    print(f"  ROI (leveraged):       {total_pnl / (total_trades * 100) * 100 * LEVERAGE:.2f}%")
    print("-" * 60)
    print(f"  TP1 Hit (0.8%):        {tp1_hits}/{total_trades} ({tp1_hits/total_trades*100:.1f}%)")
    print(f"  TP2 Hit (1.6%):        {tp2_hits}/{total_trades} ({tp2_hits/total_trades*100:.1f}%)")
    print(f"  TP3 Hit (3.0%):        {tp3_hits}/{total_trades} ({tp3_hits/total_trades*100:.1f}%)")
    print(f"  TP4 Hit (6.0%):        {tp4_hits}/{total_trades} ({tp4_hits/total_trades*100:.1f}%)")
    print(f"  Stop Loss Hit:         {sl_count}/{total_trades} ({sl_count/total_trades*100:.1f}%)")
    print("-" * 60)
    print(f"  BUY Trades:            {len(buy_trades)} (Avg PnL: ${buy_trades['net_pnl'].mean():.2f})")
    print(f"  SELL Trades:           {len(sell_trades)} (Avg PnL: ${sell_trades['net_pnl'].mean():.2f})")
    print("-" * 60)
    print("  Exit Reasons:")
    for reason, count in exit_counts.items():
        print(f"    {reason}: {count}")
    print("=" * 60)

    # ==================== Save to Excel ====================
    output_path = '/home/z/my-project/download/backtest_results.xlsx'

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Trades sheet
        trades_out = df_trades.copy()
        trades_out['entry_timestamp'] = trades_out['entry_timestamp'].astype(str)
        trades_out['exit_timestamp'] = trades_out['exit_timestamp'].astype(str)
        trades_out.to_excel(writer, sheet_name='Trades', index=False)

        # Summary sheet
        summary_data = {
            'Metric': [
                'Backtest Period', 'Timeframe', 'Leverage',
                'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate (%)',
                'Profit Factor', 'Total Net PnL ($)', 'Total Fees Paid ($)',
                'Avg PnL per Trade ($)', 'Avg Win ($)', 'Avg Loss ($)',
                'Max Win ($)', 'Max Loss ($)',
                'ROI (%) (unleveraged)', 'ROI (%) (leveraged)',
                'TP1 Hit (0.8%)', 'TP2 Hit (1.6%)', 'TP3 Hit (3.0%)', 'TP4 Hit (6.0%)',
                'Stop Loss Hit', 'SL Moved to BE+0.1%',
                'BUY Trades', 'SELL Trades',
                'BUY Avg PnL ($)', 'SELL Avg PnL ($)',
                'MEXC Taker Fee (%)',
            ],
            'Value': [
                f"{START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}",
                TIMEFRAME, f"{LEVERAGE}x",
                total_trades, winning_trades, losing_trades, round(win_rate, 1),
                round(profit_factor, 2), round(total_pnl, 2), round(total_fees_paid, 2),
                round(avg_pnl, 2), round(avg_win, 2), round(avg_loss, 2),
                round(max_win, 2), round(max_loss, 2),
                round(total_pnl / (total_trades * 100) * 100, 2),
                round(total_pnl / (total_trades * 100) * 100 * LEVERAGE, 2),
                f"{tp1_hits}/{total_trades} ({tp1_hits/total_trades*100:.1f}%)",
                f"{tp2_hits}/{total_trades} ({tp2_hits/total_trades*100:.1f}%)",
                f"{tp3_hits}/{total_trades} ({tp3_hits/total_trades*100:.1f}%)",
                f"{tp4_hits}/{total_trades} ({tp4_hits/total_trades*100:.1f}%)",
                f"{sl_count}/{total_trades} ({sl_count/total_trades*100:.1f}%)",
                f"{df_trades['sl_moved'].sum()}/{total_trades}",
                len(buy_trades), len(sell_trades),
                round(buy_trades['net_pnl'].mean(), 2) if len(buy_trades) > 0 else 0,
                round(sell_trades['net_pnl'].mean(), 2) if len(sell_trades) > 0 else 0,
                f"{MEXC_TAKER_FEE*100:.3f}%",
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Exit reasons sheet
        df_exit = exit_counts.reset_index()
        df_exit.columns = ['Exit Reason', 'Count']
        df_exit.to_excel(writer, sheet_name='Exit Reasons', index=False)

        # Settings sheet
        settings_data = {
            'Setting': [
                'TP1 (%)', 'TP1 Close %', 'TP2 (%)', 'TP2 Close %',
                'TP3 (%)', 'TP3 Close %', 'TP4 (%)', 'TP4 Close %',
                'Initial SL (%)', 'SL after TP1 (%)', 'Leverage',
                'MEXC Fee (%)', 'Position Size ($)',
                'Excluded Coins'
            ],
            'Value': [
                '0.8', '50%', '1.6', '25%',
                '3.0', '10%', '6.0', '15%',
                f'{SL_PERC}', f'{SL_AFTER_TP1}', f'{LEVERAGE}x',
                f'{MEXC_TAKER_FEE*100:.3f}%', '100',
                ', '.join(EXCLUDED_COINS)
            ]
        }
        df_settings = pd.DataFrame(settings_data)
        df_settings.to_excel(writer, sheet_name='Settings', index=False)

    print(f"\n  Results saved to: {output_path}")
    print(f"  Sheets: Trades, Summary, Exit Reasons, Settings")

    # Update worklog
    worklog_path = '/home/z/my-project/worklog.md'
    log_entry = f"""
---
Task ID: 1
Agent: Main Agent
Task: Create 1-month backtest with modified TP/SL levels, partial closes, MEXC fees

Work Log:
- Read original bot.py and extracted Squeeze Momentum strategy
- Created backtest.py with 4 TP levels (0.8%/50%, 1.6%/25%, 3%/10%, 6%/15%)
- Implemented SL trailing: after TP1, move SL to entry +0.10%
- Added MEXC taker fee calculation (0.02% per trade)
- Excluded 10 coins: SPACEX(PRE), RAIN, TOYL, WXT, UPC, DN, AIXPLAY, MBG, KAZAR, STAR
- Fetched top 50 coins by volume, ran backtest on 1 month of 15m data
- Saved results to Excel with 4 sheets

Stage Summary:
- Total trades: {total_trades}
- Win rate: {win_rate:.1f}%
- Total PnL: ${total_pnl:.2f}
- Results saved: {output_path}
"""
    with open(worklog_path, 'a') as f:
        f.write(log_entry)

    return df_trades


if __name__ == "__main__":
    run_backtest()