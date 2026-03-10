"""
Robust walk-forward nearest-history forecaster.

Features:
1) Walk-forward evaluation (no lookahead leakage)
2) Weighted nearest-neighbor forecasts for multiple horizons
3) Uncertainty bands (p10/p50/p90 from neighbor outcomes)
4) Confidence score and confidence-gated signals
5) Auto model selection across distance/regime configurations

Outputs:
  stocks/nepal/<SYMBOL>/generic/nearest_history_walkforward/
    - model_selection.csv
    - walkforward_predictions.csv
    - walkforward_summary.csv
    - backtest_signals.csv
    - backtest_summary.csv
    - calibration_table.csv
    - calibration_summary.csv
    - latest_forecast.csv
    - walkforward_report.md
    - human_readable_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from dtaidistance import dtw


@dataclass
class ModelConfig:
    distance: str
    regime_filter: bool
    top_k: int
    trend_tol: float
    vol_tol: float

    @property
    def name(self) -> str:
        rf = "on" if self.regime_filter else "off"
        return f"{self.distance}_regime_{rf}_k{self.top_k}"


def _load_ohlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    date_col = None
    for c in ("date", "published_date", "datetime"):
        if c in cols:
            date_col = cols[c]
            break
    if date_col is None:
        raise KeyError(f"No date column found in {csv_path}")
    if "close" not in cols:
        raise KeyError(f"No close column found in {csv_path}")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col]),
            "Close": pd.to_numeric(df[cols["close"]], errors="coerce"),
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
    trend_pct = (prices[-1] / prices[0] - 1.0) * 100.0 if prices[0] != 0 else 0.0
    rets = np.diff(prices) / prices[:-1]
    vol_pct = float(np.std(rets) * 100.0) if len(rets) > 0 else 0.0
    return float(trend_pct), vol_pct


def _distance(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    if mode == "dtw":
        return float(dtw.distance_fast(a.astype(np.double), b.astype(np.double)))
    return float(np.linalg.norm(a - b))


def _confidence_score(weights: np.ndarray, distances: np.ndarray, top_k: int) -> float:
    # Effective sample size from weights: higher is better
    ess = 1.0 / float(np.sum(weights ** 2))
    ess_norm = min(1.0, ess / max(1.0, float(top_k)))
    # Distance concentration: tighter distance spread => better
    d_mean = float(np.mean(distances))
    d_std = float(np.std(distances))
    spread = d_std / (d_mean + 1e-9)
    spread_score = math.exp(-spread)  # in (0,1]
    # Blend
    return 0.55 * ess_norm + 0.45 * spread_score


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> List[float]:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    out = []
    for q in quantiles:
        idx = np.searchsorted(cdf, q, side="left")
        idx = min(idx, len(v) - 1)
        out.append(float(v[idx]))
    return out


def _predict_for_asof(
    closes: np.ndarray,
    as_of_end: int,
    window: int,
    horizons: Sequence[int],
    cfg: ModelConfig,
    candidate_stride: int,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Tuple[float, float, float]], int, float]:
    """
    Returns:
      pred_ret[h], up_prob[h], bands[h]=(p10,p50,p90), used_matches, confidence
    """
    hmax = max(horizons)
    current_start = as_of_end - window + 1
    cur_raw = closes[current_start : as_of_end + 1]
    cur_norm = _zscore(cur_raw)
    cur_trend, cur_vol = _segment_features(cur_raw)

    max_candidate_start = as_of_end - window - hmax + 1
    if max_candidate_start < 0:
        zero_pred = {h: 0.0 for h in horizons}
        zero_bands = {h: (0.0, 0.0, 0.0) for h in horizons}
        return zero_pred, zero_pred, zero_bands, 0, 0.0

    candidates: List[Tuple[float, int]] = []
    for i in range(0, max_candidate_start + 1, max(1, candidate_stride)):
        cand_raw = closes[i : i + window]
        if cfg.regime_filter:
            cand_trend, cand_vol = _segment_features(cand_raw)
            if abs(cand_trend - cur_trend) > cfg.trend_tol or abs(cand_vol - cur_vol) > cfg.vol_tol:
                continue
        d = _distance(cur_norm, _zscore(cand_raw), cfg.distance)
        candidates.append((d, i))

    if not candidates and cfg.regime_filter:
        # Fallback without regime filter to avoid empty prediction.
        fallback = ModelConfig(
            distance=cfg.distance,
            regime_filter=False,
            top_k=cfg.top_k,
            trend_tol=cfg.trend_tol,
            vol_tol=cfg.vol_tol,
        )
        return _predict_for_asof(closes, as_of_end, window, horizons, fallback, candidate_stride)

    if not candidates:
        zero_pred = {h: 0.0 for h in horizons}
        zero_bands = {h: (0.0, 0.0, 0.0) for h in horizons}
        return zero_pred, zero_pred, zero_bands, 0, 0.0

    candidates.sort(key=lambda x: x[0])
    top = candidates[: cfg.top_k]
    distances = np.array([d for d, _ in top], dtype=float)
    eps = 1e-9
    weights = np.array([1.0 / (d + eps) for d in distances], dtype=float)
    weights /= np.sum(weights)
    confidence = _confidence_score(weights, distances, cfg.top_k)

    pred_ret: Dict[int, float] = {}
    up_prob: Dict[int, float] = {}
    bands: Dict[int, Tuple[float, float, float]] = {}
    for h in horizons:
        rets = []
        ups = []
        for (_, start_idx), w in zip(top, weights):
            entry = start_idx + window - 1
            exit_ = entry + h
            r = (closes[exit_] / closes[entry] - 1.0) * 100.0
            rets.append(r)
            ups.append(1.0 if r > 0 else 0.0)
        ret_arr = np.array(rets, dtype=float)
        pred_ret[h] = float(np.dot(weights, ret_arr))
        up_prob[h] = float(np.dot(weights, np.array(ups, dtype=float)))
        p10, p50, p90 = _weighted_quantiles(ret_arr, weights, [0.1, 0.5, 0.9])
        bands[h] = (p10, p50, p90)

    return pred_ret, up_prob, bands, len(top), confidence


def _actual_returns(closes: np.ndarray, as_of_end: int, horizons: Sequence[int]) -> Dict[int, float]:
    entry = closes[as_of_end]
    return {h: float((closes[as_of_end + h] / entry - 1.0) * 100.0) for h in horizons}


def _metrics(pred: np.ndarray, act: np.ndarray) -> Dict[str, float]:
    if len(pred) == 0:
        return {"samples": 0, "mae": 0.0, "rmse": 0.0, "corr": 0.0, "hit_rate": 0.0}
    err = pred - act
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 1 else 0.0
    hit = float(np.mean((pred > 0) == (act > 0)) * 100.0)
    return {"samples": int(len(pred)), "mae": mae, "rmse": rmse, "corr": corr, "hit_rate": hit}


def _fit_prob_calibration(rows_train: List[dict], horizon: int, bins: int = 10) -> Tuple[List[dict], float]:
    """
    Bin-based probability calibration on train set.
    Returns calibration rows and global positive rate fallback.
    """
    if not rows_train:
        return [], 0.5
    y = np.array([1.0 if float(r[f"actual_ret_h{horizon}_pct"]) > 0 else 0.0 for r in rows_train], dtype=float)
    p = np.array([float(r[f"pred_up_prob_h{horizon}"]) for r in rows_train], dtype=float)
    global_rate = float(np.mean(y)) if len(y) else 0.5

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        raw_avg = float(np.mean(p[mask])) if cnt else float((lo + hi) / 2.0)
        emp_rate = float(np.mean(y[mask])) if cnt else global_rate
        rows.append(
            {
                "horizon": horizon,
                "bin_idx": i,
                "bin_low": f"{lo:.4f}",
                "bin_high": f"{hi:.4f}",
                "count": cnt,
                "raw_avg_prob": f"{raw_avg:.4f}",
                "empirical_up_rate": f"{emp_rate:.4f}",
            }
        )
    return rows, global_rate


def _apply_bin_calibration(prob: float, table_rows: List[dict], fallback: float) -> float:
    if not table_rows:
        return fallback
    p = min(1.0, max(0.0, prob))
    for r in table_rows:
        lo = float(r["bin_low"])
        hi = float(r["bin_high"])
        is_last = abs(hi - 1.0) < 1e-12
        if (p >= lo and p < hi) or (is_last and p <= hi):
            return float(r["empirical_up_rate"])
    return fallback


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.0
    return float(np.mean((probs - outcomes) ** 2))


def _signal_from_forecast(
    pred_ret: float,
    up_prob: float,
    confidence: float,
    min_conf: float,
    ret_threshold: float,
    up_prob_threshold: float,
) -> str:
    if confidence < min_conf:
        return "FLAT"
    if pred_ret >= ret_threshold and up_prob >= up_prob_threshold:
        return "LONG"
    if pred_ret <= -ret_threshold and up_prob <= (1.0 - up_prob_threshold):
        return "SHORT"
    return "FLAT"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _bias_label(pred_ret: float, up_prob: float) -> str:
    if pred_ret >= 0.5 and up_prob >= 0.55:
        return "BULLISH"
    if pred_ret <= -0.5 and up_prob <= 0.45:
        return "BEARISH"
    return "MIXED"


def _run_walkforward(
    closes: np.ndarray,
    dates: np.ndarray,
    window: int,
    horizons: Sequence[int],
    cfg: ModelConfig,
    eval_step: int,
    candidate_stride: int,
    min_history_bars: int,
) -> List[dict]:
    hmax = max(horizons)
    n = len(closes)
    start_asof = max(window + min_history_bars, window + hmax + 5)
    end_asof = n - hmax - 1
    if end_asof <= start_asof:
        return []

    rows: List[dict] = []
    for as_of_end in range(start_asof, end_asof + 1, max(1, eval_step)):
        pred_ret, up_prob, bands, used, conf = _predict_for_asof(
            closes=closes,
            as_of_end=as_of_end,
            window=window,
            horizons=horizons,
            cfg=cfg,
            candidate_stride=candidate_stride,
        )
        act_ret = _actual_returns(closes, as_of_end, horizons)
        row = {
            "as_of_idx": as_of_end,
            "as_of_date": dates[as_of_end],
            "matches_used": used,
            "confidence": f"{conf:.4f}",
            "model": cfg.name,
        }
        for h in horizons:
            p10, p50, p90 = bands[h]
            row[f"pred_ret_h{h}_pct"] = f"{pred_ret[h]:.4f}"
            row[f"pred_up_prob_h{h}"] = f"{up_prob[h]:.4f}"
            row[f"pred_p10_h{h}_pct"] = f"{p10:.4f}"
            row[f"pred_p50_h{h}_pct"] = f"{p50:.4f}"
            row[f"pred_p90_h{h}_pct"] = f"{p90:.4f}"
            row[f"actual_ret_h{h}_pct"] = f"{act_ret[h]:.4f}"
        rows.append(row)
    return rows


def _split_train_val(rows: List[dict], train_ratio: float = 0.7) -> Tuple[List[dict], List[dict]]:
    if not rows:
        return [], []
    cut = max(1, int(len(rows) * train_ratio))
    return rows[:cut], rows[cut:]


def _score_model(rows_val: List[dict], horizons: Sequence[int], score_horizon: int) -> float:
    if not rows_val:
        return -1e9
    h = score_horizon
    pred = np.array([float(r[f"pred_ret_h{h}_pct"]) for r in rows_val], dtype=float)
    act = np.array([float(r[f"actual_ret_h{h}_pct"]) for r in rows_val], dtype=float)
    m = _metrics(pred, act)
    # Higher hit/corr and lower MAE is better
    return m["hit_rate"] + 10.0 * m["corr"] - 2.0 * m["mae"]


def _backtest_signals(
    rows: List[dict],
    signal_horizon: int,
    min_conf: float,
    ret_threshold: float,
    up_prob_threshold: float,
    fee_bps: float,
    slippage_bps: float,
    use_calibrated_prob: bool,
) -> Tuple[List[dict], dict]:
    out = []
    trade_returns = []
    gross_trade_returns = []
    roundtrip_cost_pct = 2.0 * (fee_bps + slippage_bps) / 100.0
    for r in rows:
        pred_ret = float(r[f"pred_ret_h{signal_horizon}_pct"])
        if use_calibrated_prob and f"cal_up_prob_h{signal_horizon}" in r:
            up_prob = float(r[f"cal_up_prob_h{signal_horizon}"])
        else:
            up_prob = float(r[f"pred_up_prob_h{signal_horizon}"])
        conf = float(r["confidence"])
        actual = float(r[f"actual_ret_h{signal_horizon}_pct"])
        signal = _signal_from_forecast(pred_ret, up_prob, conf, min_conf, ret_threshold, up_prob_threshold)
        gross_ret = 0.0
        net_ret = 0.0
        if signal == "LONG":
            gross_ret = actual
            net_ret = gross_ret - roundtrip_cost_pct
            trade_returns.append(net_ret)
            gross_trade_returns.append(gross_ret)
        elif signal == "SHORT":
            gross_ret = -actual
            net_ret = gross_ret - roundtrip_cost_pct
            trade_returns.append(net_ret)
            gross_trade_returns.append(gross_ret)
        out.append(
            {
                "as_of_date": r["as_of_date"],
                "signal": signal,
                "pred_ret_pct": f"{pred_ret:.4f}",
                "pred_up_prob": f"{up_prob:.4f}",
                "confidence": f"{conf:.4f}",
                "actual_ret_pct": f"{actual:.4f}",
                "gross_strategy_ret_pct": f"{gross_ret:.4f}",
                "cost_pct": f"{roundtrip_cost_pct:.4f}" if signal != "FLAT" else "0.0000",
                "net_strategy_ret_pct": f"{net_ret:.4f}",
            }
        )

    total = len(out)
    trades = len(trade_returns)
    wins = sum(1 for x in trade_returns if x > 0)
    summary = {
        "samples": total,
        "trades": trades,
        "trade_rate_pct": (trades / total * 100.0) if total else 0.0,
        "win_rate_pct": (wins / trades * 100.0) if trades else 0.0,
        "avg_gross_trade_ret_pct": float(np.mean(gross_trade_returns)) if trades else 0.0,
        "cum_gross_strategy_ret_pct": float(np.sum(gross_trade_returns)) if trades else 0.0,
        "avg_trade_ret_pct": float(np.mean(trade_returns)) if trades else 0.0,
        "median_trade_ret_pct": float(np.median(trade_returns)) if trades else 0.0,
        "cum_strategy_ret_pct": float(np.sum(trade_returns)) if trades else 0.0,
        "roundtrip_cost_pct": roundtrip_cost_pct,
        "use_calibrated_prob": "YES" if use_calibrated_prob else "NO",
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust nearest-history walk-forward forecaster")
    parser.add_argument("symbol", help="NEPSE symbol, e.g. NICA")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--distance", choices=["auto", "euclidean", "dtw"], default="auto")
    parser.add_argument("--regime-filter", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--trend-tol", type=float, default=8.0)
    parser.add_argument("--vol-tol", type=float, default=2.5)
    parser.add_argument("--eval-step", type=int, default=5)
    parser.add_argument("--candidate-stride", type=int, default=2)
    parser.add_argument("--min-history-bars", type=int, default=250)
    parser.add_argument("--score-horizon", type=int, default=10)
    parser.add_argument("--signal-horizon", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--ret-threshold", type=float, default=0.5)
    parser.add_argument("--up-prob-threshold", type=float, default=0.55)
    parser.add_argument("--fee-bps", type=float, default=20.0, help="One-way fee in bps (default: 20)")
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="One-way slippage in bps (default: 10)")
    parser.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
    parser.add_argument("--input-csv", help="Optional explicit input CSV path")
    parser.add_argument("--output-dir", help="Optional explicit output dir")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    if args.score_horizon not in horizons or args.signal_horizon not in horizons:
        raise ValueError("--score-horizon and --signal-horizon must be included in --horizons")

    input_csv = Path(args.input_csv) if args.input_csv else Path(f"data/nepal/{symbol}.csv")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"stocks/nepal/{symbol}/generic/nearest_history_walkforward")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_ohlc(input_csv)
    closes = df["Close"].to_numpy(dtype=float)
    dates = df["Date"].dt.strftime("%Y-%m-%d").to_numpy()

    # Build candidate model configs
    distances = [args.distance] if args.distance != "auto" else ["euclidean", "dtw"]
    if args.regime_filter == "auto":
        regime_flags = [True, False]
    else:
        regime_flags = [args.regime_filter == "on"]
    configs = [
        ModelConfig(
            distance=d,
            regime_filter=rf,
            top_k=args.top_k,
            trend_tol=args.trend_tol,
            vol_tol=args.vol_tol,
        )
        for d in distances
        for rf in regime_flags
    ]

    model_rows = []
    all_runs: Dict[str, List[dict]] = {}
    best_name = None
    best_score = -1e18
    for cfg in configs:
        run_rows = _run_walkforward(
            closes=closes,
            dates=dates,
            window=args.window,
            horizons=horizons,
            cfg=cfg,
            eval_step=args.eval_step,
            candidate_stride=args.candidate_stride,
            min_history_bars=args.min_history_bars,
        )
        train_rows, val_rows = _split_train_val(run_rows, train_ratio=0.7)
        score = _score_model(val_rows, horizons, args.score_horizon)
        all_runs[cfg.name] = run_rows
        model_rows.append(
            {
                "model": cfg.name,
                "distance": cfg.distance,
                "regime_filter": "ON" if cfg.regime_filter else "OFF",
                "samples_total": len(run_rows),
                "samples_train": len(train_rows),
                "samples_val": len(val_rows),
                "score": f"{score:.4f}",
            }
        )
        if score > best_score:
            best_score = score
            best_name = cfg.name

    if best_name is None:
        raise RuntimeError("Could not evaluate any model configuration.")

    best_rows = all_runs[best_name]

    # Summary metrics per horizon for best model
    summary_rows = []
    for h in horizons:
        pred = np.array([float(r[f"pred_ret_h{h}_pct"]) for r in best_rows], dtype=float)
        act = np.array([float(r[f"actual_ret_h{h}_pct"]) for r in best_rows], dtype=float)
        m = _metrics(pred, act)
        summary_rows.append(
            {
                "horizon": h,
                "samples": m["samples"],
                "mae_pct": f"{m['mae']:.4f}",
                "rmse_pct": f"{m['rmse']:.4f}",
                "corr_pred_actual": f"{m['corr']:.4f}",
                "directional_hit_rate_pct": f"{m['hit_rate']:.2f}",
                "avg_pred_ret_pct": f"{float(np.mean(pred)):.4f}",
                "avg_actual_ret_pct": f"{float(np.mean(act)):.4f}",
            }
        )

    # Probability calibration (fit on train, evaluate on val)
    train_rows, val_rows = _split_train_val(best_rows, train_ratio=0.7)
    calibration_rows = []
    calibration_summary_rows = []
    for h in horizons:
        h_table, fallback = _fit_prob_calibration(train_rows, horizon=h, bins=10)
        calibration_rows.extend(h_table)
        # Apply calibrated probs to all rows for downstream use
        for r in best_rows:
            raw_p = float(r[f"pred_up_prob_h{h}"])
            cal_p = _apply_bin_calibration(raw_p, h_table, fallback) if not args.no_calibration else raw_p
            r[f"cal_up_prob_h{h}"] = f"{cal_p:.4f}"

        if val_rows:
            y_val = np.array([1.0 if float(r[f"actual_ret_h{h}_pct"]) > 0 else 0.0 for r in val_rows], dtype=float)
            p_raw = np.array([float(r[f"pred_up_prob_h{h}"]) for r in val_rows], dtype=float)
            p_cal = np.array([float(r[f"cal_up_prob_h{h}"]) for r in val_rows], dtype=float)
            calibration_summary_rows.append(
                {
                    "horizon": h,
                    "val_samples": len(val_rows),
                    "brier_raw": f"{_brier_score(p_raw, y_val):.6f}",
                    "brier_calibrated": f"{_brier_score(p_cal, y_val):.6f}",
                    "calibration_used": "NO" if args.no_calibration else "YES",
                }
            )

    # Signal backtest on best model
    bt_rows, bt_summary = _backtest_signals(
        rows=best_rows,
        signal_horizon=args.signal_horizon,
        min_conf=args.min_confidence,
        ret_threshold=args.ret_threshold,
        up_prob_threshold=args.up_prob_threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        use_calibrated_prob=not args.no_calibration,
    )

    # Latest forecast row
    latest = best_rows[-1]
    latest_row = {
        "symbol": symbol,
        "model": best_name,
        "as_of_date": latest["as_of_date"],
        "matches_used": latest["matches_used"],
        "confidence": latest["confidence"],
    }
    for h in horizons:
        for key in ("pred_ret", "pred_up_prob", "pred_p10", "pred_p50", "pred_p90"):
            latest_row[f"{key}_h{h}"] = latest[f"{key}_h{h}_pct"] if "pred_up_prob" not in key else latest[f"{key}_h{h}"]

    # Write outputs
    model_sel_path = output_dir / "model_selection.csv"
    pred_path = output_dir / "walkforward_predictions.csv"
    summary_path = output_dir / "walkforward_summary.csv"
    backtest_path = output_dir / "backtest_signals.csv"
    backtest_summary_path = output_dir / "backtest_summary.csv"
    calibration_table_path = output_dir / "calibration_table.csv"
    calibration_summary_path = output_dir / "calibration_summary.csv"
    latest_path = output_dir / "latest_forecast.csv"
    report_path = output_dir / "walkforward_report.md"
    human_path = output_dir / "human_readable_summary.md"

    _write_csv(model_sel_path, list(model_rows[0].keys()), model_rows)
    _write_csv(pred_path, list(best_rows[0].keys()), best_rows)
    _write_csv(
        summary_path,
        [
            "horizon",
            "samples",
            "mae_pct",
            "rmse_pct",
            "corr_pred_actual",
            "directional_hit_rate_pct",
            "avg_pred_ret_pct",
            "avg_actual_ret_pct",
        ],
        summary_rows,
    )
    _write_csv(backtest_path, list(bt_rows[0].keys()) if bt_rows else ["as_of_date"], bt_rows)
    _write_csv(backtest_summary_path, list(bt_summary.keys()), [bt_summary])
    _write_csv(
        calibration_table_path,
        [
            "horizon",
            "bin_idx",
            "bin_low",
            "bin_high",
            "count",
            "raw_avg_prob",
            "empirical_up_rate",
        ],
        calibration_rows,
    )
    _write_csv(
        calibration_summary_path,
        ["horizon", "val_samples", "brier_raw", "brier_calibrated", "calibration_used"],
        calibration_summary_rows,
    )
    _write_csv(latest_path, list(latest_row.keys()), [latest_row])

    # Markdown report
    lines = []
    lines.append(f"# Robust Walk-Forward Report ({symbol})")
    lines.append("")
    lines.append(f"- Data range: `{dates[0]}` to `{dates[-1]}`")
    lines.append(f"- Window: `{args.window}`")
    lines.append(f"- Horizons: `{','.join(map(str, horizons))}`")
    lines.append(f"- Best model: `{best_name}`")
    lines.append(f"- Best-model score ({args.score_horizon}-bar): `{best_score:.4f}`")
    lines.append("")
    lines.append("## Model Selection")
    lines.append("")
    lines.append("| Model | Distance | Regime | Samples Val | Score |")
    lines.append("|---|---|---|---:|---:|")
    for r in sorted(model_rows, key=lambda x: float(x["score"]), reverse=True):
        lines.append(
            f"| {r['model']} | {r['distance']} | {r['regime_filter']} | {r['samples_val']} | {r['score']} |"
        )
    lines.append("")
    lines.append("## Forecast Metrics (Best Model)")
    lines.append("")
    lines.append("| Horizon | Samples | MAE % | RMSE % | Corr | Direction Hit % |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| {r['horizon']} | {r['samples']} | {r['mae_pct']} | {r['rmse_pct']} | "
            f"{r['corr_pred_actual']} | {r['directional_hit_rate_pct']} |"
        )
    lines.append("")
    lines.append("## Signal Backtest (Best Model)")
    lines.append("")
    lines.append(f"- Signal horizon: `{args.signal_horizon}`")
    lines.append(f"- Min confidence: `{args.min_confidence}`")
    lines.append(f"- Return threshold: `{args.ret_threshold}%`")
    lines.append(f"- Up-prob threshold: `{args.up_prob_threshold}`")
    lines.append(f"- Trades: `{bt_summary['trades']}` / `{bt_summary['samples']}` ({bt_summary['trade_rate_pct']:.2f}%)")
    lines.append(f"- Win rate: `{bt_summary['win_rate_pct']:.2f}%`")
    lines.append(f"- Avg gross trade return: `{bt_summary['avg_gross_trade_ret_pct']:.2f}%`")
    lines.append(f"- Avg net trade return: `{bt_summary['avg_trade_ret_pct']:.2f}%`")
    lines.append(f"- Roundtrip cost used: `{bt_summary['roundtrip_cost_pct']:.2f}%`")
    lines.append(f"- Cum gross strategy return: `{bt_summary['cum_gross_strategy_ret_pct']:.2f}%`")
    lines.append(f"- Cum net strategy return (sum of trade returns): `{bt_summary['cum_strategy_ret_pct']:.2f}%`")
    if calibration_summary_rows:
        lines.append("")
        lines.append("## Calibration Check (Validation)")
        lines.append("")
        lines.append("| Horizon | Samples | Brier Raw | Brier Calibrated |")
        lines.append("|---:|---:|---:|---:|")
        for r in calibration_summary_rows:
            lines.append(
                f"| {r['horizon']} | {r['val_samples']} | {r['brier_raw']} | {r['brier_calibrated']} |"
            )
    lines.append("")
    lines.append("## Latest Forecast (Best Model)")
    lines.append("")
    lines.append(f"- As-of date: `{latest_row['as_of_date']}`")
    lines.append(f"- Confidence: `{latest_row['confidence']}`")
    for h in horizons:
        lines.append(
            f"- H{h}: mean `{latest[f'pred_ret_h{h}_pct']}%`, "
            f"up-prob `{float(latest[f'pred_up_prob_h{h}']) * 100:.2f}%`, "
            f"band `[p10={latest[f'pred_p10_h{h}_pct']}%, p50={latest[f'pred_p50_h{h}_pct']}%, p90={latest[f'pred_p90_h{h}_pct']}%]`"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Human-readable quick decision summary
    hlines = []
    hlines.append(f"# Human Readable Summary ({symbol})")
    hlines.append("")
    hlines.append("## Model Health")
    hlines.append(f"- Best model: `{best_name}`")
    hlines.append(f"- Validation score: `{best_score:.2f}`")
    if summary_rows:
        h10 = next((r for r in summary_rows if int(r["horizon"]) == 10), summary_rows[0])
        hlines.append(f"- 10-bar directional hit rate: `{h10['directional_hit_rate_pct']}%`")
        hlines.append(f"- 10-bar correlation (pred vs actual): `{h10['corr_pred_actual']}`")
        hlines.append(f"- 10-bar MAE: `{h10['mae_pct']}%`")
    hlines.append("")
    hlines.append("## Latest Market View")
    hlines.append(f"- As-of date: `{latest_row['as_of_date']}`")
    hlines.append(f"- Forecast confidence score: `{latest_row['confidence']}` (higher is better)")
    hlines.append("")
    hlines.append("| Horizon | Expected Return | Up Probability | P10 | P50 | P90 | Bias |")
    hlines.append("|---:|---:|---:|---:|---:|---:|---|")
    for h in horizons:
        pred = float(latest[f"pred_ret_h{h}_pct"])
        up = float(latest[f"pred_up_prob_h{h}"])
        p10 = float(latest[f"pred_p10_h{h}_pct"])
        p50 = float(latest[f"pred_p50_h{h}_pct"])
        p90 = float(latest[f"pred_p90_h{h}_pct"])
        bias = _bias_label(pred, up)
        hlines.append(
            f"| {h} | {pred:.2f}% | {up*100:.2f}% | {p10:.2f}% | {p50:.2f}% | {p90:.2f}% | {bias} |"
        )
    hlines.append("")
    hlines.append("## How To Use")
    hlines.append("- Treat this as probability guidance, not certainty.")
    hlines.append("- Prefer action only when expected return, up-probability, and median (P50) agree.")
    hlines.append("- If horizons disagree, use smaller position size or stay flat.")
    human_path.write_text("\n".join(hlines) + "\n", encoding="utf-8")

    print(f"Robust walk-forward report generated for {symbol}")
    print(f"  Model selection: {model_sel_path}")
    print(f"  Predictions: {pred_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Backtest signals: {backtest_path}")
    print(f"  Backtest summary: {backtest_summary_path}")
    print(f"  Calibration table: {calibration_table_path}")
    print(f"  Calibration summary: {calibration_summary_path}")
    print(f"  Latest forecast: {latest_path}")
    print(f"  Report: {report_path}")
    print(f"  Human summary: {human_path}")
    print(f"  Best model: {best_name} (score={best_score:.4f})")


if __name__ == "__main__":
    main()
