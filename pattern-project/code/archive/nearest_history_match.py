"""
Nearest-history pattern matcher for NEPSE stock CSV data.

Given a symbol, this script:
1. Takes the latest `window` closes as the current pattern.
2. Finds most similar historical windows (z-normalized distance).
3. Measures what happened after each historical match over `horizon` bars.
4. Exports clean CSV + markdown summary.

Output directory:
  stocks/nepal/<SYMBOL>/generic/nearest_history_match/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dtaidistance import dtw


@dataclass
class MatchRow:
    rank: int
    start_idx: int
    end_idx: int
    start_date: str
    end_date: str
    distance: float
    entry_idx: int
    entry_date: str
    entry_price: float
    exit_idx: int
    exit_date: str
    exit_price: float
    fwd_return_pct: float
    max_up_pct: float
    max_down_pct: float
    trend_pct: float
    vol_pct: float
    trend_diff: float
    vol_diff: float


def _load_ohlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    columns_lower = {c.lower(): c for c in df.columns}

    date_col = None
    for c in ("date", "published_date", "datetime"):
        if c in columns_lower:
            date_col = columns_lower[c]
            break
    if date_col is None:
        raise KeyError(f"No date column found in {csv_path}")

    close_col = None
    for c in ("close",):
        if c in columns_lower:
            close_col = columns_lower[c]
            break
    if close_col is None:
        raise KeyError(f"No close column found in {csv_path}")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col]),
            "Close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - float(np.mean(arr))) / std


def _segment_features(prices: np.ndarray) -> Tuple[float, float]:
    """Return (trend_pct, vol_pct) for a price segment."""
    trend_pct = (prices[-1] / prices[0] - 1.0) * 100.0 if prices[0] != 0 else 0.0
    rets = np.diff(prices) / prices[:-1]
    vol_pct = float(np.std(rets) * 100.0) if len(rets) > 0 else 0.0
    return float(trend_pct), vol_pct


def _distance(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    if mode == "dtw":
        return float(dtw.distance_fast(a.astype(np.double), b.astype(np.double)))
    return float(np.linalg.norm(a - b))


def _find_matches(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    top_k: int,
    distance_mode: str,
    regime_filter: bool,
    trend_tol: float,
    vol_tol: float,
) -> Tuple[List[MatchRow], Dict[str, float]]:
    closes = df["Close"].to_numpy(dtype=float)
    dates = df["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    n = len(closes)

    if n < window + horizon + 20:
        raise ValueError(
            f"Not enough rows ({n}) for window={window}, horizon={horizon}. Need at least {window + horizon + 20}."
        )

    current_start = n - window
    current_segment = _zscore(closes[current_start : current_start + window])
    current_raw = closes[current_start : current_start + window]
    current_trend, current_vol = _segment_features(current_raw)

    candidates = []
    scanned = 0
    filtered_out = 0
    # candidate segment [i, i+window-1], future measured from i+window-1 for `horizon` bars
    max_i = n - window - horizon
    for i in range(0, max_i + 1):
        # avoid overlap with current window
        if i + window - 1 >= current_start:
            continue

        scanned += 1
        raw_seg = closes[i : i + window]
        seg_trend, seg_vol = _segment_features(raw_seg)
        trend_diff = abs(seg_trend - current_trend)
        vol_diff = abs(seg_vol - current_vol)
        if regime_filter and (trend_diff > trend_tol or vol_diff > vol_tol):
            filtered_out += 1
            continue

        seg = _zscore(closes[i : i + window])
        distance = _distance(current_segment, seg, distance_mode)

        entry_idx = i + window - 1
        exit_idx = entry_idx + horizon
        entry_price = closes[entry_idx]
        exit_price = closes[exit_idx]
        fwd_return = (exit_price / entry_price - 1.0) * 100.0

        future = closes[entry_idx + 1 : exit_idx + 1]
        max_up = (np.max(future) / entry_price - 1.0) * 100.0
        max_down = (np.min(future) / entry_price - 1.0) * 100.0

        candidates.append(
            MatchRow(
                rank=0,
                start_idx=i,
                end_idx=i + window - 1,
                start_date=dates[i],
                end_date=dates[i + window - 1],
                distance=distance,
                entry_idx=entry_idx,
                entry_date=dates[entry_idx],
                entry_price=float(entry_price),
                exit_idx=exit_idx,
                exit_date=dates[exit_idx],
                exit_price=float(exit_price),
                fwd_return_pct=float(fwd_return),
                max_up_pct=float(max_up),
                max_down_pct=float(max_down),
                trend_pct=seg_trend,
                vol_pct=seg_vol,
                trend_diff=trend_diff,
                vol_diff=vol_diff,
            )
        )

    # Fallback: if regime filter is too strict and leaves no candidates, retry without it.
    fallback_used = False
    if regime_filter and len(candidates) == 0:
        fallback_used = True
        return _find_matches(
            df=df,
            window=window,
            horizon=horizon,
            top_k=top_k,
            distance_mode=distance_mode,
            regime_filter=False,
            trend_tol=trend_tol,
            vol_tol=vol_tol,
        )

    candidates.sort(key=lambda m: m.distance)
    top = candidates[:top_k]
    for idx, m in enumerate(top, start=1):
        m.rank = idx

    returns = [m.fwd_return_pct for m in top]
    summary = {
        "matches": len(top),
        "avg_fwd_return_pct": float(mean(returns)) if returns else 0.0,
        "median_fwd_return_pct": float(median(returns)) if returns else 0.0,
        "win_rate_pct": float(sum(1 for r in returns if r > 0) / len(returns) * 100.0) if returns else 0.0,
        "best_fwd_return_pct": float(max(returns)) if returns else 0.0,
        "worst_fwd_return_pct": float(min(returns)) if returns else 0.0,
        "current_window_start_date": dates[current_start],
        "current_window_end_date": dates[n - 1],
        "distance_mode": distance_mode,
        "regime_filter": "ON" if regime_filter else "OFF",
        "trend_tol": float(trend_tol),
        "vol_tol": float(vol_tol),
        "current_trend_pct": float(current_trend),
        "current_vol_pct": float(current_vol),
        "candidate_scanned": int(scanned),
        "candidate_kept": int(len(candidates)),
        "candidate_filtered_out": int(filtered_out),
        "fallback_used": "YES" if fallback_used else "NO",
    }
    return top, summary


def _write_matches_csv(path: Path, matches: List[MatchRow]) -> None:
    fieldnames = [
        "rank",
        "start_idx",
        "end_idx",
        "start_date",
        "end_date",
        "distance",
        "entry_idx",
        "entry_date",
        "entry_price",
        "exit_idx",
        "exit_date",
        "exit_price",
        "fwd_return_pct",
        "max_up_pct",
        "max_down_pct",
        "trend_pct",
        "vol_pct",
        "trend_diff",
        "vol_diff",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in matches:
            w.writerow(
                {
                    "rank": m.rank,
                    "start_idx": m.start_idx,
                    "end_idx": m.end_idx,
                    "start_date": m.start_date,
                    "end_date": m.end_date,
                    "distance": f"{m.distance:.6f}",
                    "entry_idx": m.entry_idx,
                    "entry_date": m.entry_date,
                    "entry_price": f"{m.entry_price:.4f}",
                    "exit_idx": m.exit_idx,
                    "exit_date": m.exit_date,
                    "exit_price": f"{m.exit_price:.4f}",
                    "fwd_return_pct": f"{m.fwd_return_pct:.4f}",
                    "max_up_pct": f"{m.max_up_pct:.4f}",
                    "max_down_pct": f"{m.max_down_pct:.4f}",
                    "trend_pct": f"{m.trend_pct:.4f}",
                    "vol_pct": f"{m.vol_pct:.4f}",
                    "trend_diff": f"{m.trend_diff:.4f}",
                    "vol_diff": f"{m.vol_diff:.4f}",
                }
            )


def _write_summary_md(path: Path, symbol: str, window: int, horizon: int, summary: Dict[str, float], matches: List[MatchRow]) -> None:
    lines = []
    lines.append(f"# Nearest History Match Report ({symbol})")
    lines.append("")
    lines.append(f"- Current window: `{summary['current_window_start_date']}` to `{summary['current_window_end_date']}`")
    lines.append(f"- Pattern window: `{window}` bars")
    lines.append(f"- Forward horizon: `{horizon}` bars")
    lines.append(f"- Top matches used: `{int(summary['matches'])}`")
    lines.append(f"- Distance mode: `{summary['distance_mode']}`")
    lines.append(
        f"- Regime filter: `{summary['regime_filter']}` "
        f"(trend tol={summary['trend_tol']:.2f}, vol tol={summary['vol_tol']:.2f})"
    )
    lines.append(
        f"- Current regime: trend `{summary['current_trend_pct']:.2f}%`, "
        f"volatility `{summary['current_vol_pct']:.2f}%`"
    )
    lines.append(
        f"- Candidate windows: scanned `{summary['candidate_scanned']}`, "
        f"kept `{summary['candidate_kept']}`, filtered `{summary['candidate_filtered_out']}`"
    )
    if summary.get("fallback_used") == "YES":
        lines.append("- Fallback used: regime filter was too strict, auto-reran without filter.")
    lines.append("")
    lines.append("## Outcome Summary (Top Matches)")
    lines.append("")
    lines.append(f"- Avg forward return: `{summary['avg_fwd_return_pct']:.2f}%`")
    lines.append(f"- Median forward return: `{summary['median_fwd_return_pct']:.2f}%`")
    lines.append(f"- Win rate: `{summary['win_rate_pct']:.2f}%`")
    lines.append(f"- Best forward return: `{summary['best_fwd_return_pct']:.2f}%`")
    lines.append(f"- Worst forward return: `{summary['worst_fwd_return_pct']:.2f}%`")
    lines.append("")
    lines.append("## Top Matches")
    lines.append("")
    lines.append("| Rank | Segment Dates | Distance | Trend % | Vol % | Forward Return % | Max Up % | Max Down % |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for m in matches:
        lines.append(
            f"| {m.rank} | {m.start_date} -> {m.end_date} | {m.distance:.4f} | "
            f"{m.trend_pct:.2f} | {m.vol_pct:.2f} | {m.fwd_return_pct:.2f} | {m.max_up_pct:.2f} | {m.max_down_pct:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-history pattern matcher")
    parser.add_argument("symbol", help="NEPSE symbol, e.g. NICA")
    parser.add_argument("--window", type=int, default=40, help="Pattern window length (default: 40)")
    parser.add_argument("--horizon", type=int, default=10, help="Forward bars to measure outcome (default: 10)")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top matches (default: 12)")
    parser.add_argument(
        "--distance",
        choices=["dtw", "euclidean"],
        default="dtw",
        help="Distance metric for pattern similarity (default: dtw)",
    )
    parser.add_argument(
        "--no-regime-filter",
        action="store_true",
        help="Disable regime filtering (trend/vol matching) for candidates.",
    )
    parser.add_argument(
        "--trend-tol",
        type=float,
        default=8.0,
        help="Allowed absolute trend%% difference for regime filter (default: 8.0)",
    )
    parser.add_argument(
        "--vol-tol",
        type=float,
        default=2.5,
        help="Allowed absolute volatility%% difference for regime filter (default: 2.5)",
    )
    parser.add_argument("--input-csv", help="Optional explicit input CSV path")
    parser.add_argument("--output-dir", help="Optional explicit output dir")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    input_csv = Path(args.input_csv) if args.input_csv else Path(f"data/nepal/{symbol}.csv")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"stocks/nepal/{symbol}/generic/nearest_history_match")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_ohlc(input_csv)
    matches, summary = _find_matches(
        df,
        window=args.window,
        horizon=args.horizon,
        top_k=args.top_k,
        distance_mode=args.distance,
        regime_filter=not args.no_regime_filter,
        trend_tol=args.trend_tol,
        vol_tol=args.vol_tol,
    )

    matches_csv = output_dir / "nearest_matches.csv"
    summary_md = output_dir / "nearest_match_report.md"
    _write_matches_csv(matches_csv, matches)
    _write_summary_md(summary_md, symbol=symbol, window=args.window, horizon=args.horizon, summary=summary, matches=matches)

    print(f"Nearest-history report generated for {symbol}")
    print(f"  Input: {input_csv}")
    print(f"  Matches CSV: {matches_csv}")
    print(f"  Summary MD: {summary_md}")
    print(f"  Avg return: {summary['avg_fwd_return_pct']:.2f}%")
    print(f"  Win rate: {summary['win_rate_pct']:.2f}%")
    print(f"  Distance mode: {summary['distance_mode']}")
    print(f"  Regime filter: {summary['regime_filter']}")


if __name__ == "__main__":
    main()
