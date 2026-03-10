"""
Build a concise market-state report from existing pattern outputs.

For each symbol, it reads:
  - stocks/nepal/<SYMBOL>/custom/csv/in_out_pattern_9_18.csv
  - stocks/nepal/<SYMBOL>/custom/csv/transition_clean_prev2_to_next.csv
  - stocks/nepal/<SYMBOL>/custom/csv/transition_train_recent_validation.csv

And outputs:
  - latest clean 2-state combo
  - predicted next state (highest probability)
  - probability and sample size
  - confidence tag
  - drift warning from train-vs-recent validation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


KEEP_TOKENS = {"IN_UP", "IN_DOWN", "OUT_UP", "OUT_DOWN"}


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_clean_prev2(inout_rows: List[dict]) -> Optional[Tuple[str, str]]:
    labels = [r.get("point_label", "") for r in inout_rows if r.get("point_label", "") in KEEP_TOKENS]
    if len(labels) < 2:
        return None
    return labels[-2], labels[-1]


def choose_confidence(prob: float, total_count: int) -> str:
    if total_count >= 8 and prob >= 0.70:
        return "HIGH"
    if total_count >= 5 and prob >= 0.55:
        return "MEDIUM"
    return "LOW"


def predict_next(clean_rows: List[dict], prev2: Tuple[str, str]) -> Tuple[str, float, int]:
    candidates = [
        r for r in clean_rows
        if r.get("prev_2_a") == prev2[0] and r.get("prev_2_b") == prev2[1]
    ]
    if not candidates:
        return "N/A", 0.0, 0

    best = max(candidates, key=lambda r: float(r.get("prob_next_given_prev2", 0.0)))
    return (
        best.get("next", "N/A"),
        float(best.get("prob_next_given_prev2", 0.0)),
        int(float(best.get("total_context_count", 0))),
    )


def drift_status(validation_rows: List[dict], prev2: Tuple[str, str]) -> str:
    key = f"{prev2[0]}|{prev2[1]}"
    for row in validation_rows:
        if row.get("prev_2") == key:
            drift = row.get("top_next_drift", "N/A")
            if drift == "CHANGED":
                return "WARNING_CHANGED"
            if drift == "UNCHANGED":
                return "OK_UNCHANGED"
            return drift or "N/A"
    return "N/A"


def symbol_report(symbol: str, base_dir: Path) -> dict:
    results_dir = base_dir / symbol / "custom"
    inout_rows = read_csv(results_dir / "csv" / "in_out_pattern_9_18.csv")
    clean_rows = read_csv(results_dir / "csv" / "transition_clean_prev2_to_next.csv")
    validation_rows = read_csv(results_dir / "csv" / "transition_train_recent_validation.csv")

    if not inout_rows or not clean_rows:
        return {
            "symbol": symbol,
            "latest_prev2": "N/A",
            "predicted_next": "N/A",
            "probability": "0.0000",
            "context_count": "0",
            "confidence": "N/A",
            "drift_status": "N/A",
            "note": "missing required result files",
        }

    prev2 = latest_clean_prev2(inout_rows)
    if prev2 is None:
        return {
            "symbol": symbol,
            "latest_prev2": "N/A",
            "predicted_next": "N/A",
            "probability": "0.0000",
            "context_count": "0",
            "confidence": "N/A",
            "drift_status": "N/A",
            "note": "not enough clean states",
        }

    predicted, prob, total_count = predict_next(clean_rows, prev2)
    confidence = choose_confidence(prob, total_count)
    drift = drift_status(validation_rows, prev2)
    return {
        "symbol": symbol,
        "latest_prev2": f"{prev2[0]}|{prev2[1]}",
        "predicted_next": predicted,
        "probability": f"{prob:.4f}",
        "context_count": str(total_count),
        "confidence": confidence,
        "drift_status": drift,
        "note": "",
    }


def discover_symbols(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    symbols = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "results" / "csv" / "in_out_pattern_9_18.csv").exists():
            symbols.append(p.name)
    return symbols


def print_table(rows: List[dict]) -> None:
    headers = [
        "symbol", "latest_prev2", "predicted_next", "probability",
        "context_count", "confidence", "drift_status", "note"
    ]
    widths: Dict[str, int] = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(r.get(h, "")))

    def fmt(row: dict) -> str:
        return " | ".join(row.get(h, "").ljust(widths[h]) for h in headers)

    print(fmt({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(fmt(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest state -> next-state market report")
    parser.add_argument("symbols", nargs="*", help="Symbols like NICA HIDCL SHEL")
    parser.add_argument(
        "--base-dir",
        default="stocks/nepal",
        help="Base directory containing <SYMBOL>/results",
    )
    parser.add_argument(
        "--output-csv",
        default="stocks/nepal/market_state_report.csv",
        help="Path to save summary CSV",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    symbols = [s.upper() for s in args.symbols] if args.symbols else discover_symbols(base_dir)
    if not symbols:
        print("No symbols found to report.")
        return

    rows = [symbol_report(sym, base_dir) for sym in symbols]
    print_table(rows)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol", "latest_prev2", "predicted_next", "probability",
                "context_count", "confidence", "drift_status", "note"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
