# Algorithm Notes

This document explains how we draw `in_out_pattern_9_18_visualization.png`, how we label `IN_UP/IN_DOWN/OUT_UP/OUT_DOWN`, and how we build the chain pattern output.

## 1) High/Low Extraction (Crossover-Based)

Source: `code/pattern/analyze_nepal_stock.py`

Rules:
1. Compute EMA crossovers (default `9/18`).
2. Skip early startup crossovers with insufficient history.
3. Create one initial seed point before the first meaningful crossover.
4. For each crossover, pick exactly one point in its forward segment until the next crossover.
5. Enforce alternation: `HIGH -> LOW -> HIGH -> LOW ...`.

Outputs:
- CSV: `stocks/nepal/<SYMBOL>/results/csv/highs_lows_pattern_9_18.csv`
- PNG: `stocks/nepal/<SYMBOL>/results/png/highs_lows_pattern_9_18.png`

## 2) Valid Pattern Definition (0,1,2,3)

Source: `code/pattern/pattern_detector_v2.py`

Valid patterns are *only* complete 4-point windows:

Uptrend:
- `0=LOW, 1=HIGH, 2=LOW, 3=HIGH`
- `LOW(2) > LOW(0)` and `HIGH(3) > HIGH(1)`

Downtrend:
- `0=HIGH, 1=LOW, 2=HIGH, 3=LOW`
- `HIGH(2) < HIGH(0)` and `LOW(3) < LOW(1)`

All other combinations are invalid.

## 3) IN/OUT Labeling

Rule:
- `0` is the *start* of a valid pattern.
- If `0` overlaps any other valid pattern’s `1/2/3`, it becomes **IN**.
- If `0` does **not** overlap, it remains **OUT**.
- Points `1/2/3` are always **IN** (by that pattern’s trend).
- Points not in any valid pattern are **unlabeled**.

Labels:
- `IN_UP` / `IN_DOWN`: points `1/2/3` in valid up/down patterns
- `OUT_UP` / `OUT_DOWN`: point `0` of valid up/down patterns (if not overlapping)

Outputs:
- CSV: `stocks/nepal/<SYMBOL>/results/csv/in_out_pattern_9_18.csv`
- PNG: `stocks/nepal/<SYMBOL>/results/png/in_out_pattern_9_18_visualization.png`

## 4) Chain Pattern Output (Main Algorithm)

Source: `code/pattern/transition_pattern_analysis.py`

This is the main market-footprint algorithm:

1. Scan the labeled stream for immediate valid 4-point patterns.
2. Build a sequence of patterns in time order.
3. Use a sliding window: `(A, B) -> C`, then `(B, C) -> D`, etc.
4. Output dates using each pattern’s **0-point** date.

Main output:
- `stocks/nepal/<SYMBOL>/results/txt/in_out_up_down_9_18_chain.txt`

Support output:
- `stocks/nepal/<SYMBOL>/results/txt/in_out_up_down_9_18.txt`
  (same transitions, aggregated by counts/probabilities)

## 5) Swing Fallback (Optional)

If you want a swing-only view for the same labels:
- `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_to_swing.txt`
