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


def run_cmd(cmd: list[str]) -> None:
    import os
    print(f"\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "." # Ensure project root is in path
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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


    print(f"\n✓ Pattern pipeline complete for {symbol} ({market})")
    print(f"  Results: results/{market}/{symbol}/{strategy}/")


if __name__ == "__main__":
    main()
