"""
Generic script to analyze Nepal stocks with EMA crossover detection and visualization.
Usage: python nepal.py <symbol> [--ema-short 9] [--ema-long 18] [--output-dir results/nepal/<symbol>/in_out]
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import sys
from pathlib import Path
import requests
from datetime import datetime, UTC
import re
import argparse


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


def _fetch_from_github_company_wise(symbol: str, output_path: str) -> bool:
    """Primary provider: Aabishkar2 company-wise CSV."""
    url = f"https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise/{symbol}.csv"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and response.text.strip() and "404: Not Found" not in response.text:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"  Downloaded {symbol} from GitHub company-wise source")
            return True
        print(f"  GitHub company-wise source miss for {symbol}: HTTP {response.status_code}")
    except Exception as e:
        print(f"  GitHub company-wise source error for {symbol}: {e}")
    return False


def _fetch_from_sharesansar_price_history(symbol: str, output_path: str) -> bool:
    """
    Secondary provider: Sharesansar company-price-history DataTable endpoint.
    Provides longer history for symbols that may be missing in GitHub source.
    """
    company_url = f"https://www.sharesansar.com/company/{symbol}"
    history_url = "https://www.sharesansar.com/company-price-history"
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        page = session.get(company_url, timeout=20, headers=headers)
        if page.status_code != 200:
            print(f"  Sharesansar company page miss for {symbol}: HTTP {page.status_code}")
            return False

        html = page.text
        company_match = re.search(r"id=['\"]companyid['\"][^>]*>([^<]+)<", html)
        token_match = re.search(r"name=['\"]_token['\"][^>]*content=['\"]([^'\"]+)", html)
        if not company_match or not token_match:
            print(f"  Sharesansar price-history metadata missing for {symbol}")
            return False
        company_id = company_match.group(1).strip()
        csrf_token = token_match.group(1)

        post_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": company_url,
            "Origin": "https://www.sharesansar.com",
            "X-CSRF-Token": csrf_token,
            "X-Requested-With": "XMLHttpRequest",
        }

        # First page to get recordsTotal
        page_size = 50
        first_payload = {"company": company_id, "draw": 1, "start": 0, "length": page_size}
        first = session.post(history_url, data=first_payload, headers=post_headers, timeout=20)
        if first.status_code != 200:
            print(f"  Sharesansar price-history miss for {symbol}: HTTP {first.status_code}")
            return False
        first_json = first.json()
        total = int(first_json.get("recordsTotal", 0))
        data_rows = list(first_json.get("data", []))
        if total == 0 or len(data_rows) == 0:
            print(f"  Sharesansar price-history has no rows for {symbol}")
            return False

        # Paginate if needed
        start = page_size
        draw = 2
        while start < total:
            payload = {"company": company_id, "draw": draw, "start": start, "length": page_size}
            resp = session.post(history_url, data=payload, headers=post_headers, timeout=20)
            if resp.status_code != 200:
                break
            js = resp.json()
            chunk = js.get("data", [])
            if not chunk:
                break
            data_rows.extend(chunk)
            start += page_size
            draw += 1

        # Normalize rows (source is newest-first)
        cleaned = []
        for r in data_rows:
            cleaned.append(
                {
                    "published_date": str(r.get("published_date", "")).strip(),
                    "open": _to_float(r.get("open")),
                    "high": _to_float(r.get("high")),
                    "low": _to_float(r.get("low")),
                    "close": _to_float(r.get("close")),
                    "per_change": _to_float(r.get("per_change")),
                    "traded_quantity": _to_float(r.get("traded_quantity")),
                    "traded_amount": _to_float(r.get("traded_amount")),
                    "status": _to_int(r.get("status"), default=0),
                }
            )

        df = pd.DataFrame(cleaned)
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
        df = df.dropna(subset=["published_date", "close"]).sort_values("published_date").copy()
        if len(df) == 0:
            print(f"  Sharesansar price-history normalization failed for {symbol}")
            return False
        df["published_date"] = df["published_date"].dt.strftime("%Y-%m-%d")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(
            f"  Downloaded {symbol} from Sharesansar price-history source "
            f"({df['published_date'].iloc[0]} to {df['published_date'].iloc[-1]}, {len(df)} rows)"
        )
        return True
    except Exception as e:
        print(f"  Sharesansar price-history source error for {symbol}: {e}")
        return False


def _fetch_from_sharesansar_chart_api(symbol: str, output_path: str) -> bool:
    """
    Secondary provider: Sharesansar TradingView-UDF history endpoint.
    Uses countback API to obtain recent OHLCV if primary source misses a symbol.
    """
    url = "https://www.sharesansar.com/company-chart/history"
    params = {
        "symbol": symbol,
        "resolution": "1D",
        "from": 0,
        "to": 9999999999,
        "countback": 5000,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://www.sharesansar.com/company/{symbol}",
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"  Sharesansar source miss for {symbol}: HTTP {response.status_code}")
            return False
        payload = response.json()
        if payload.get("s") != "ok" or not payload.get("t"):
            print(f"  Sharesansar source has no usable OHLC data for {symbol}")
            return False

        t = payload.get("t", [])
        o = payload.get("o", [])
        h = payload.get("h", [])
        l = payload.get("l", [])
        c = payload.get("c", [])
        v = payload.get("v", [])
        n = min(len(t), len(o), len(h), len(l), len(c), len(v))
        if n == 0:
            print(f"  Sharesansar source returned empty arrays for {symbol}")
            return False

        dates = [datetime.fromtimestamp(ts, UTC).date().isoformat() for ts in t[:n]]
        close_series = pd.Series(c[:n], dtype=float)
        per_change = close_series.pct_change() * 100.0
        traded_amount = pd.Series(v[:n], dtype=float) * close_series

        df = pd.DataFrame(
            {
                "published_date": dates,
                "open": pd.Series(o[:n], dtype=float),
                "high": pd.Series(h[:n], dtype=float),
                "low": pd.Series(l[:n], dtype=float),
                "close": close_series,
                "per_change": per_change.round(4),
                "traded_quantity": pd.Series(v[:n], dtype=float),
                "traded_amount": traded_amount.round(2),
                "status": 0,
            }
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(
            f"  Downloaded {symbol} from Sharesansar chart source "
            f"({dates[0]} to {dates[-1]}, {n} rows)"
        )
        return True
    except Exception as e:
        print(f"  Sharesansar source error for {symbol}: {e}")
    return False


def fetch_nepse_csv(symbol: str, output_path: str) -> bool:
    """Download NEPSE CSV for symbol using fallback providers."""
    if _fetch_from_github_company_wise(symbol, output_path):
        return True
    if _fetch_from_sharesansar_price_history(symbol, output_path):
        return True
    if _fetch_from_sharesansar_chart_api(symbol, output_path):
        return True
    return False


def _read_latest_local_date(csv_path: str):
    """Best-effort latest date from local CSV."""
    try:
        df = pd.read_csv(csv_path)
        lower_to_actual = {c.strip().lower(): c for c in df.columns}
        for key in ['date', 'datetime', 'published_date']:
            if key in lower_to_actual:
                d = pd.to_datetime(df[lower_to_actual[key]], errors='coerce').dropna()
                if len(d) > 0:
                    return d.max().date()
    except Exception:
        return None
    return None


def load_stock_data(
    symbol: str,
    data_dir: str = 'data/nepal',
    refresh_mode: str = 'auto',
    max_stale_days: int = 7
) -> pd.DataFrame:
    """Load stock CSV data for a given symbol from the data directory."""
    symbol = symbol.upper()
    csv_path = os.path.join(data_dir, f'{symbol}.csv')
    local_exists = os.path.exists(csv_path)

    if not local_exists:
        print(f"  Local data not found: {csv_path}")
        print(f"  Attempting to fetch {symbol} from NEPSE source...")
        fetched = fetch_nepse_csv(symbol, csv_path)
        if not fetched:
            raise FileNotFoundError(
                f"Data file not found and auto-fetch failed for symbol {symbol}: {csv_path}"
            )
    else:
        should_refresh = False
        if refresh_mode == 'always':
            should_refresh = True
        elif refresh_mode == 'auto':
            latest_local_date = _read_latest_local_date(csv_path)
            if latest_local_date is None:
                should_refresh = True
            else:
                age_days = (datetime.now(UTC).date() - latest_local_date).days
                should_refresh = age_days > max_stale_days
                if should_refresh:
                    print(
                        f"  Local data looks stale ({age_days} days old, latest {latest_local_date}). "
                        f"Attempting refresh..."
                    )
        if should_refresh:
            fetched = fetch_nepse_csv(symbol, csv_path)
            if not fetched:
                print("  Refresh failed; continuing with existing local file.")
    
    df = pd.read_csv(csv_path)

    # Normalize common column variants (case-insensitive)
    lower_to_actual = {c.strip().lower(): c for c in df.columns}
    if 'close' in lower_to_actual and 'Close' not in df.columns:
        df.rename(columns={lower_to_actual['close']: 'Close'}, inplace=True)

    # Parse date column (handles case variants)
    date_key_candidates = ['date', 'datetime', 'published_date']
    date_col = None
    for key in date_key_candidates:
        if key in lower_to_actual:
            date_col = lower_to_actual[key]
            break
    if date_col is None:
        raise KeyError(
            f"No date column found in {csv_path}. Expected one of: Date/Datetime/Published_date"
        )
    df['Date'] = pd.to_datetime(df[date_col])

    if 'Close' not in df.columns:
        raise KeyError(f"No close column found in {csv_path}. Expected a Close/close column.")
    
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def calculate_ema_cross(df: pd.DataFrame, short_span: int = 9, long_span: int = 18) -> pd.DataFrame:
    """
    Calculate EMA crossovers and add columns:
    - ema_short, ema_long: the two moving averages
    - ema_cross: +1 (bullish), -1 (bearish), 0 (none)
    """
    df = df.copy()
    
    # Calculate EMAs on Close price
    df['ema_short'] = df['Close'].ewm(span=short_span, adjust=False).mean()
    df['ema_long']  = df['Close'].ewm(span=long_span,  adjust=False).mean()
    df['ema_diff']  = df['ema_short'] - df['ema_long']
    
    # Detect crossovers
    df['ema_cross'] = 0
    df.loc[(df['ema_diff'] > 0) & (df['ema_diff'].shift(1) <= 0), 'ema_cross'] = 1
    df.loc[(df['ema_diff'] < 0) & (df['ema_diff'].shift(1) >= 0), 'ema_cross'] = -1
    
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
        s = df['Close'].iloc[:end_idx]
        # local high/low with 1-bar neighborhood
        if extreme_type == 'high':
            candidates = s[(s >= s.shift(1)) & (s > s.shift(-1))].index.tolist()
        else:
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
        init_idx = _nearest_local_extreme_before(first_cross_idx, initial_type)
        if init_idx is None:
            if initial_type == 'high':
                init_idx = int(pre_window['Close'].idxmax())
            else:
                init_idx = int(pre_window['Close'].idxmin())
        points.append({
            'index': init_idx,
            'date': df.loc[init_idx, 'Date'],
            'price': float(df.loc[init_idx, 'Close']),
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
            point_idx = window['Close'].idxmax()
        else:
            point_idx = window['Close'].idxmin()

        points.append({
            'index': int(point_idx),
            'date': df.loc[point_idx, 'Date'],
            'price': float(df.loc[point_idx, 'Close']),
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

    # Draw the zigzag line
    ax.plot(dates, prices, label='Price Action Zigzag', color='#8b949e',
            linewidth=2, marker='o', markersize=8, alpha=0.7)

    # Color code points: bull cross = neon green, bear cross = bright red
    for point in points_sorted:
        color = '#00e676' if point['cross_type'] == 'bull' else '#ef5350'
        ax.scatter(point['date'], point['price'], color=color, s=220,
                   zorder=5, alpha=0.85, edgecolors='white', linewidth=1)

    ax.set_xlabel('Date', fontsize=14, color='#c9d1d9')
    ax.set_ylabel('Price', fontsize=14, color='#c9d1d9')
    ax.set_title(f'{symbol} - Price Zigzag: Highs/Lows After EMA Crosses ({short_span}/{long_span})',
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
    args = parser.parse_args()

    symbol = args.symbol.upper()
    short_span = args.ema_short
    long_span = args.ema_long
    output_dir = f'results/nepal/{symbol}/in_out'
    
    try:
        print(f"Loading {symbol}...")
        df = load_stock_data(
            symbol,
            refresh_mode=args.refresh,
            max_stale_days=args.max_stale_days
        )
        print(f"  Loaded {len(df)} rows")
        
        print(f"Calculating {short_span}/{long_span} EMA crossovers...")
        df = calculate_ema_cross(df, short_span, long_span)
        
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
