"""
Global pattern runner for any stock symbol across markets.

Runs, in order:
1) code/algorithms/ema_viz.py
2) code/algorithms/in_out/detector.py
3) code/algorithms/in_out/analyzer.py

Usage:
  ./venv/bin/python code/pipelines/global_pipeline.py NICA --market nepal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path

def run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "." # Ensure project root is in path
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

def get_last_data_date(symbol: str, market: str) -> str:
    """Read the raw data CSV to find the latest date it contains."""
    csv_path = Path(f"data/{market}/{symbol.upper()}.csv")
    if not csv_path.exists():
        return ""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=100000) # Read enough but don't blow up
        # Try to find a date column
        for col in ['published_date', 'Date', 'date', 'time']:
            if col in df.columns:
                return str(pd.to_datetime(df[col]).max().date())
    except:
        pass
    return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="Run pattern pipeline globally across markets.")
    parser.add_argument("symbol", help="Stock symbol, e.g. NICA, AAPL, BTCUSDT")
    parser.add_argument("--market", default="nepal", help="Market name, e.g. nepal, intl, crypto")
    parser.add_argument("--strategy", default="in_out", help="Strategy variant, e.g. in_out, structural_v2")
    parser.add_argument("--ema-short", type=int, default=9)
    parser.add_argument("--ema-long", type=int, default=18)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--refresh", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-stale-days", type=int, default=0)
    parser.add_argument("--years", type=int, default=None,
                        help="Limit analysis to last N years of data (e.g. 5). Default: all data.")
    parser.add_argument("--show-roles", action="store_true", help="Show pattern role labels (0, 1, 2, 3)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    market = args.market.lower()
    strategy = args.strategy.lower()
    py = sys.executable

    # Define result path and state file
    res_dir = Path(f"results/{market}/{symbol}/{strategy}")
    state_file = res_dir / "analysis_state.json"
    
    current_config = {
        "ema_short": args.ema_short,
        "ema_long": args.ema_long,
        "threshold": args.threshold,
        "years": args.years,
        "strategy": strategy,
        "market": market
    }

    # --- Pre-run Check: Skip if everything is already fresh ---
    if args.refresh != 'always' and state_file.exists():
        current_csv_date = get_last_data_date(symbol, market)
        if current_csv_date:
            try:
                with open(state_file, 'r') as f:
                    last_state = json.load(f)
                
                # If parameters match and we have the same date
                if (last_state.get("last_data_date") == current_csv_date and 
                    last_state.get("config") == current_config):
                    
                    # For nepal market, avoid running if it's currently a non-trading day/time
                    # and we already have the previous session's data.
                    if market == "nepal":
                        from zoneinfo import ZoneInfo
                        from datetime import timedelta
                        now_nepal = datetime.now(ZoneInfo("Asia/Kathmandu"))
                        today = now_nepal.date()
                        
                        # Monday=0 ... Sunday=6. NEPSE is Sun-Thu (6, 0, 1, 2, 3)
                        is_trading_day = now_nepal.weekday() in (6, 0, 1, 2, 3)
                        
                        # Simplified freshness:
                        # If today is NOT a trading day (Fri/Sat), and data is from the last trading day, skip.
                        # If today IS a trading day but it's before market (11 AM), skip.
                        should_check_for_new_data = True
                        
                        if not is_trading_day:
                            should_check_for_new_data = False
                        elif now_nepal.hour < 11:
                            should_check_for_new_data = False
                        
                        if not should_check_for_new_data:
                            print(f"\n--- [Global Skip] All results (+charts) are up-to-date for {symbol} ({current_csv_date}). ---")
                            return
            except Exception:
                pass

    # 1. EMA Visualization & Trend Detection
    ema_cmd = [
        py,
        "code/algorithms/ema_viz.py",
        symbol,
        "--market", market,
        "--strategy", strategy,
        "--ema-short", str(args.ema_short),
        "--ema-long", str(args.ema_long),
        "--threshold", str(args.threshold),
        "--refresh", args.refresh,
        "--max-stale-days", str(args.max_stale_days),
    ]
    if args.years is not None:
        ema_cmd += ["--years", str(args.years)]
    
    run_cmd(ema_cmd)

    # After ema_viz runs, we definitely have the latest data CSV.
    # Check if we can skip the rest of the detection/analysis steps.
    current_max_date = get_last_data_date(symbol, market)
    
    if args.refresh != 'always' and state_file.exists():
        try:
            with open(state_file, 'r') as f:
                last_state = json.load(f)
            
            if (last_state.get("last_data_date") == current_max_date and 
                last_state.get("config") == current_config):
                print(f"\n--- Analysis up-to-date for {symbol} ({current_max_date}). Skipping further processing. ---")
                return
        except:
            pass

    # 2. IN/OUT Pattern Detection
    detector_cmd = [
        py, 
        "code/algorithms/in_out/detector.py", 
        symbol, 
        "--market", market,
        "--strategy", strategy
    ]
    if args.show_roles:
        detector_cmd += ["--show-roles"]
    
    # 3. Transition Pattern Analysis
    analyzer_cmd = [
        py, 
        "code/algorithms/in_out/analyzer.py", 
        symbol, 
        "--market", market,
        "--strategy", strategy
    ]

    if args.years is not None:
        detector_cmd += ["--years", str(args.years)]
        analyzer_cmd += ["--years", str(args.years)]

    run_cmd(detector_cmd)
    run_cmd(analyzer_cmd)

    # Save state after successful completion
    os.makedirs(res_dir, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump({
            "last_data_date": current_max_date,
            "last_run_timestamp": datetime.now().isoformat(),
            "config": current_config
        }, f, indent=2)

    print(f"\n✓ Pattern pipeline complete for {symbol} ({market})")
    print(f"  Results cached: {res_dir}")


if __name__ == "__main__":
    main()
