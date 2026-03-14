"""
Generic script to analyze Nepal stocks with EMA crossover detection and visualization.
Usage: python nepal.py <symbol> [--ema-short 9] [--ema-long 18] [--output-dir results/nepal/<symbol>/in_out]
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import requests
import argparse
from code.data_fetchers.nepal import NepalStockService


def _to_float(value):
    if value is None:
        return np.nan
    s = str(value).replace(",", "").strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _to_int(value, default=0):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def load_stock_data(
    symbol: str,
    data_dir: str = 'data/nepal',
    refresh_mode: str = 'auto',
    max_stale_days: int = 0
) -> pd.DataFrame:
    """Load stock data via the unified NepalStockService."""
    service = NepalStockService(data_dir=data_dir, max_stale_days=max_stale_days)
    return service.load_data(symbol, refresh_mode=refresh_mode)


def calculate_ema_cross(df: pd.DataFrame, short_span: int = 9, long_span: int = 18, threshold: float = 0.0) -> pd.DataFrame:
    """
    Calculate EMA crossovers and add columns:
    - ema_short, ema_long: the two moving averages
    - ema_cross: +1 (bullish), -1 (bearish), 0 (none)
    - threshold can be used to filter noise; cross only triggers if diff > threshold
    """
    df = df.copy()
    
    # Calculate EMAs on Close price
    df['ema_short'] = df['Close'].ewm(span=short_span, adjust=False).mean()
    df['ema_long']  = df['Close'].ewm(span=long_span,  adjust=False).mean()
    df['ema_diff']  = df['ema_short'] - df['ema_long']
    
    # Detect crossovers using hysteresis/threshold
    df['ema_cross'] = 0
    df['ema_diff'] = df['ema_short'] - df['ema_long']
    
    if threshold == 0.0:
        df.loc[(df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1)), 'ema_cross'] = 1
        df.loc[(df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1)), 'ema_cross'] = -1
        return df

    # We use a state-based approach to handle the threshold correctly
    # Finding points where bull/bear is confirmed
    active_bull = df['ema_diff'] > threshold
    active_bear = df['ema_diff'] < -threshold
    
    # 1. Capture signs (-1, 0, 1) based on threshold
    signs = np.zeros(len(df))
    signs[df['ema_diff'] > threshold] = 1
    signs[df['ema_diff'] < -threshold] = -1
    
    # 2. Fill zeros with previous non-zero state (preserving the trend until it breaks threshold)
    series_signs = pd.Series(signs, index=df.index)
    filled_signs = series_signs.replace(0, np.nan).ffill().fillna(0)
    
    # 3. Detect changes in the filled state
    diff_signs = filled_signs.diff()
    
    # If it goes from 0 to 1, or -1 to 1 => Bullish Cross (+1)
    df.loc[((filled_signs == 1) & (filled_signs.shift(1) != 1)).to_numpy(), 'ema_cross'] = 1
    # If it goes from 0 to -1, or 1 to -1 => Bearish Cross (-1)
    df.loc[((filled_signs == -1) & (filled_signs.shift(1) != -1)).to_numpy(), 'ema_cross'] = -1

    return df


def detect_swing_points(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Detect swing highs and lows using a rolling window.
    Returns two lists: indices of highs and lows.
    """
    highs = []
    lows = []
    
    for i in range(window, len(df) - window):
        close_price = df.iloc[i]['Close']
        left_prices = df.iloc[i-window:i]['Close'].values
        right_prices = df.iloc[i+1:i+1+window]['Close'].values
        
        # Check if it's a swing high
        if close_price >= left_prices.max() and close_price >= right_prices.max():
            highs.append(i)
        # Check if it's a swing low
        elif close_price <= left_prices.min() and close_price <= right_prices.min():
            lows.append(i)
    
    return highs, lows


def extract_zigzag_points(df: pd.DataFrame) -> list:
    """
    Extract meaningful swing points anchored to EMA crossovers.

    Rules implemented:
    - If there is at least one crossover, start with one INITIAL point from BEFORE
      the first crossover:
        first bullish crossover -> initial LOW before crossover
        first bearish crossover -> initial HIGH before crossover
    - Each crossover then gets exactly one assigned point in its FORWARD segment
      [cross_i, cross_{i+1}) (or to end for last crossover).
    - Swing type alternates globally:
      HIGH -> LOW -> HIGH -> LOW ...
      This keeps the zigzag consistent and avoids duplicate "jobs" per crossover.

    Returns a list of dicts with:
    - index, date, price
    - type ('high'/'low')
    - cross_type ('bull'/'bear') of the crossover owning that point
    """
    points = []

    # Find all crossover rows
    crosses = df[df['ema_cross'] != 0].copy()
    if len(crosses) == 0:
        return points

    cross_positions = list(crosses.index)
    # Skip startup crossovers that occur too early in the dataset; they often
    # don't have enough pre-history to form a meaningful "before crossover" anchor.
    min_pre_bars_for_seed = 20
    while cross_positions and cross_positions[0] < min_pre_bars_for_seed:
        cross_positions.pop(0)
    if not cross_positions:
        return points

    def _nearest_local_extreme_before(end_idx: int, extreme_type: str) -> int | None:
        """
        Find nearest local extreme before end_idx.
        extreme_type: 'high' or 'low'
        Returns absolute df index or None if not found.
        """
        if end_idx <= 2:
            return None
        # local high/low with 1-bar neighborhood
        if extreme_type == 'high':
            s = df['High'].iloc[:end_idx]
            candidates = s[(s >= s.shift(1)) & (s > s.shift(-1))].index.tolist()
        else:
            s = df['Low'].iloc[:end_idx]
            candidates = s[(s <= s.shift(1)) & (s < s.shift(-1))].index.tolist()
        return int(candidates[-1]) if candidates else None

    # 1) Initial anchor point from before first crossover.
    first_cross_idx = cross_positions[0]
    first_cross_val = int(df.loc[first_cross_idx, 'ema_cross'])
    pre_window = df.iloc[0:first_cross_idx]
    last_point_type = None
    if len(pre_window) > 0:
        # Direction-forced seed:
        # bearish first crossover => start from pre-crossover HIGH
        # bullish first crossover => start from pre-crossover LOW
        initial_type = 'high' if first_cross_val == -1 else 'low'
        
        # Use simple absolute extreme for the seed to ensure we start from the true origin
        if initial_type == 'high':
            init_idx = int(pre_window['High'].idxmax())
        else:
            init_idx = int(pre_window['Low'].idxmin())
            
        points.append({

            'index': init_idx,
            'date': df.loc[init_idx, 'Date'],
            'date_label': df.loc[init_idx, 'Date'].strftime('%d %b %Y') if hasattr(df.loc[init_idx, 'Date'], 'strftime') else str(df.loc[init_idx, 'Date']),
            'price': float(df.loc[init_idx, 'High' if initial_type == 'high' else 'Low']),
            'type': initial_type,
            'cross_type': 'seed',
        })
        last_point_type = initial_type

    # 2) One point per crossover in forward segment.
    for i, cross_idx in enumerate(cross_positions):
        cross_val = int(df.loc[cross_idx, 'ema_cross'])  # 1 bull, -1 bear
        left = cross_idx
        if i < len(cross_positions) - 1:
            right = cross_positions[i + 1] - 1
        else:
            right = len(df) - 1
        if left > right:
            continue

        window = df.iloc[left:right + 1]
        if len(window) == 0:
            continue

        # Alternate from previous chosen point.
        if last_point_type is None:
            # Fallback for edge case (no pre-window): use crossover direction.
            target_type = 'high' if cross_val == 1 else 'low'
        else:
            target_type = 'low' if last_point_type == 'high' else 'high'

        if target_type == 'high':
            point_idx = window['High'].idxmax()
        else:
            point_idx = window['Low'].idxmin()

        points.append({
            'index': int(point_idx),
            'date': df.loc[point_idx, 'Date'],
            'date_label': df.loc[point_idx, 'Date'].strftime('%d %b %Y') if hasattr(df.loc[point_idx, 'Date'], 'strftime') else str(df.loc[point_idx, 'Date']),
            'price': float(df.loc[point_idx, 'High' if target_type == 'high' else 'Low']),
            'type': target_type,
            'cross_type': 'bull' if cross_val == 1 else 'bear',
        })
        last_point_type = target_type

    return points


def plot_ema_chart(df: pd.DataFrame, symbol: str, short_span: int = 9, long_span: int = 18) -> None:
    """
    Plot the price chart with EMA lines and crossover markers.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Plot price and EMAs
    ax.plot(df['Date'], df['Close'], label='Close Price', color='#c9d1d9', linewidth=2, alpha=0.85)
    ax.plot(df['Date'], df['ema_short'], label=f'{short_span}-EMA', color='#29b6f6', linewidth=2, alpha=0.9)
    ax.plot(df['Date'], df['ema_long'],  label=f'{long_span}-EMA',  color='#ffa726', linewidth=2, alpha=0.9)

    # Mark bullish crosses
    bull_mask = df['ema_cross'] == 1
    ax.scatter(df[bull_mask]['Date'], df[bull_mask]['Close'],
               marker='^', color='#00e676', s=150, label='Bullish Cross', zorder=5)

    # Mark bearish crosses
    bear_mask = df['ema_cross'] == -1
    ax.scatter(df[bear_mask]['Date'], df[bear_mask]['Close'],
               marker='v', color='#ef5350', s=150, label='Bearish Cross', zorder=5)

    ax.set_xlabel('Date', fontsize=14, color='#c9d1d9')
    ax.set_ylabel('Price', fontsize=14, color='#c9d1d9')
    ax.set_title(f'{symbol} - EMA Crossover Analysis ({short_span}/{long_span})',
                 fontsize=16, fontweight='bold', color='#f0f6fc')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.12, linestyle='--', color='#30363d')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_highs_lows_after_cross(
    df: pd.DataFrame,
    symbol: str,
    short_span: int = 9,
    long_span: int = 18,
    points: list | None = None,
) -> None:
    """
    Plot a simplified chart showing significant highs and lows tied to EMA crosses.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    if points is None:
        points = extract_zigzag_points(df)

    if len(points) == 0:
        print("  No zigzag points found")
        return fig

    points_sorted = sorted(points, key=lambda p: p['index'])
    dates = [p['date'] for p in points_sorted]
    prices = [p['price'] for p in points_sorted]

    # Draw the zigzag line - White and opaque for clarity
    ax.plot(dates, prices, label='Price Action Zigzag', color='#ffffff',
            linewidth=1.8, marker='o', markersize=4, alpha=0.9, zorder=3)

    price_range = max(prices) - min(prices) if len(prices) > 1 else 1
    # Plot points: Uniform neutral color for identification
    for idx_e, point in enumerate(points_sorted):
        is_seed = (point['cross_type'] == 'seed')
        # Use white/gold consistently to avoid red/green clutter
        pt_color = '#ffeb3b' if is_seed else '#ffffff'
        
        # Identification dots
        ax.scatter(point['date'], point['price'], color=pt_color, s=70,
                   zorder=5, alpha=0.9, edgecolors='#0d1117', linewidth=0.8)
        
        # Improved labeling: vertical dates, high points above, low points below
        d_obj = pd.to_datetime(point['date'])
        label_offsets = [0.02, 0.05, 0.08, 0.11]
        dist = price_range * label_offsets[idx_e % 4]
        
        is_high = (point['type'] == 'high')
        y_pos = point['price'] + dist if is_high else point['price'] - dist
        va = 'bottom' if is_high else 'top'
        
        label_text = d_obj.strftime('%d %b %y')
        if is_seed: label_text = f"SEED: {label_text}"
            
        ax.text(point['date'], y_pos, label_text,
                fontsize=7.5, color=pt_color if is_seed else '#e2e2e2', 
                ha='center', va=va, rotation=90, fontweight='bold', alpha=1.0)

    ax.set_xlabel('Date', fontsize=14, color='#c9d1d9')
    ax.set_ylabel('Price', fontsize=14, color='#c9d1d9')
    ax.set_title(f'{symbol} - Price Zigzag: Highs/Lows After EMA Crosses ({short_span}/{long_span})',
                 fontsize=16, fontweight='bold', color='#f0f6fc')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.12, linestyle='--', color='#30363d')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()

    return fig



def save_results(df: pd.DataFrame, fig, symbol: str, output_dir: str, short_span: int = 9, long_span: int = 18, points: list = None):
    """
    Save the chart as an image and the annotated data as CSV.
    Also save the zigzag points as a separate CSV for pattern detection.
    """
    csv_dir = os.path.join(output_dir, "csv")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # Save main EMA chart
    img_name = f'ema_crossover_{short_span}_{long_span}.png'
    img_path = os.path.join(png_dir, img_name)
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✓ EMA chart saved: {img_path}")
    
    # Save highs/lows chart
    img_name_hl = f'highs_lows_pattern_{short_span}_{long_span}.png'
    img_path_hl = os.path.join(png_dir, img_name_hl)
    # Note: fig_hl is generated in main, we'll save it there
    print(f"✓ Highs/lows pattern chart saved: {img_path_hl}")
    
    # Save data
    csv_name = f'ema_analysis_{short_span}_{long_span}.csv'
    csv_path = os.path.join(csv_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"✓ Data saved: {csv_path}")
    
    # Save zigzag points for pattern detection
    if points:
        zigzag_csv_name = f'highs_lows_pattern_{short_span}_{long_span}.csv'
        zigzag_csv_path = os.path.join(csv_dir, zigzag_csv_name)
        zigzag_df = pd.DataFrame(points)
        zigzag_df.to_csv(zigzag_csv_path, index=False)
        print(f"✓ Zigzag CSV saved: {zigzag_csv_path}")
    
    # Print summary
    bull_count = (df['ema_cross'] == 1).sum()
    bear_count = (df['ema_cross'] == -1).sum()
    print(f"\nSummary for {symbol}:")
    print(f"  Bullish crosses: {bull_count}")
    print(f"  Bearish crosses: {bear_count}")
    print(f"  Total rows: {len(df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze Nepal stock with EMA crossover + highs/lows extraction."
    )
    parser.add_argument("symbol", help="Stock symbol, e.g. NICA, HIDCLP")
    parser.add_argument("--ema-short", type=int, default=9)
    parser.add_argument("--ema-long", type=int, default=18)
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum EMA diff to trigger crossover (filters noise)")
    parser.add_argument(
        "--refresh",
        choices=["auto", "always", "never"],
        default="auto",
        help="Data refresh mode for existing local CSV (default: auto)",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=7,
        help="Used when --refresh auto; refresh if local data older than this (default: 7)",
    )
    parser.add_argument("--market", default="nepal", help="Market name for path organization")
    parser.add_argument("--strategy", default="in_out", help="Strategy identifier (e.g. in_out, structural_v2, strategy1)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--years", type=int, default=None, help="Limit history context (years)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    short_span = args.ema_short
    long_span = args.ema_long
    strategy = args.strategy.lower()
    output_dir = args.output_dir or f'results/{args.market.lower()}/{symbol}/{strategy}'
    
    try:
        print(f"Loading {symbol}...")
        df = load_stock_data(
            symbol,
            refresh_mode=args.refresh,
            max_stale_days=args.max_stale_days
        )
        
        if args.years:
            from datetime import datetime as _dt, timedelta as _td
            cutoff = _dt.today() - _td(days=args.years * 365)
            df = df[df['Date'] >= cutoff].reset_index(drop=True)
            print(f"  Filtering to last {args.years} years of history...")
        print(f"  Loaded {len(df)} rows")
        
        print(f"Calculating {short_span}/{long_span} EMA crossovers (threshold: {args.threshold})...")
        df = calculate_ema_cross(df, short_span, long_span, threshold=args.threshold)
        
        # Compute once, then reuse everywhere.
        points = extract_zigzag_points(df)
        points_sorted = sorted(points, key=lambda p: p['index'])

        print("Generating chart...")
        fig = plot_ema_chart(df, symbol, short_span, long_span)
        
        print("Generating highs/lows chart...")
        fig_hl = plot_highs_lows_after_cross(df, symbol, short_span, long_span, points=points_sorted)
        
        print(f"Saving results to {output_dir}...")
        save_results(df, fig, symbol, output_dir, short_span, long_span, points=points_sorted)
        
        # Save the highs/lows chart
        img_name_hl = f'highs_lows_pattern_{short_span}_{long_span}.png'
        img_path_hl = os.path.join(output_dir, "png", img_name_hl)
        fig_hl.savefig(img_path_hl, dpi=300, bbox_inches='tight')
        print(f"✓ Highs/lows pattern chart saved: {img_path_hl}")
        
        print("\n✓ Done!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
