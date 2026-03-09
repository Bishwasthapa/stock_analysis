"""
One-command pipeline runner for a Nepal stock symbol.

Runs, in order:
1) analyze_nepal_stock.py
2) pattern_detector_v2.py
3) transition_pattern_analysis.py
4) nearest_history_match.py
5) nearest_history_walkforward.py

Usage:
  ./venv/bin/python run_symbol_pipeline.py NICA
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full stock pipeline with one command.")
    parser.add_argument("symbol", help="Stock symbol, e.g. NICA, HIDCLP, SRLI")
    parser.add_argument("--ema-short", type=int, default=9)
    parser.add_argument("--ema-long", type=int, default=18)
    parser.add_argument("--refresh", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument("--walkforward-horizons", default="5,10,20")
    parser.add_argument("--walkforward-distance", choices=["auto", "euclidean", "dtw"], default="auto")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    py = sys.executable

    run_cmd(
        [
            py,
            "analyze_nepal_stock.py",
            symbol,
            "--ema-short",
            str(args.ema_short),
            "--ema-long",
            str(args.ema_long),
            "--refresh",
            args.refresh,
            "--max-stale-days",
            str(args.max_stale_days),
        ]
    )
    run_cmd([py, "pattern_detector_v2.py", symbol])
    run_cmd([py, "transition_pattern_analysis.py", symbol])
    run_cmd([py, "nearest_history_match.py", symbol])
    run_cmd(
        [
            py,
            "nearest_history_walkforward.py",
            symbol,
            "--distance",
            args.walkforward_distance,
            "--regime-filter",
            "auto",
            "--eval-step",
            "5",
            "--candidate-stride",
            "2",
            "--horizons",
            args.walkforward_horizons,
        ]
    )

    print(f"\n✓ Full pipeline complete for {symbol}")
    print(f"  Results: stocks/nepal/{symbol}/results/")


if __name__ == "__main__":
    main()
