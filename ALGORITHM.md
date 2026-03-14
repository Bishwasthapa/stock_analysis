# Algorithm Notes

This document explains how we draw `in_out_pattern_9_18_visualization.png`, how we label `IN_UP/IN_DOWN/OUT_UP/OUT_DOWN`, and how we build the chain pattern output.

## 1) High/Low Extraction (Crossover-Based)

Source: `code/data_fetchers/nepal.py`

Rules:
1. Compute EMA crossovers (default `9/18`).
2. Skip early startup crossovers with insufficient history.
3. Create one initial seed point before the first meaningful crossover.
4. For each crossover, pick exactly one point in its forward segment until the next crossover.
5. Enforce alternation: `HIGH -> LOW -> HIGH -> LOW ...`.

Outputs:
- CSV: `results/nepal/<SYMBOL>/in_out/csv/highs_lows_pattern_9_18.csv`
- PNG: `results/nepal/<SYMBOL>/in_out/png/highs_lows_pattern_9_18.png`

## 2) Valid Pattern Definition (0,1,2,3)

Source: `code/algorithms/in_out/detector.py`

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
- **Priority Rule:** If a point overlaps multiple patterns (e.g. acts as role 3 in a previous pattern and role 0 in a new pattern), it **prioritizes its role as 0**. It becomes the start of the new trend and adopts the new trend's direction.
- If `0` overlaps any other valid pattern’s `1/2/3`, it becomes **IN** (e.g., a continuous overlapping wave).
- If `0` does **not** overlap, it remains **OUT** (an isolated trend start).
- Points `1/2/3` are always **IN** (following their pattern’s trend).
- Points not in any valid pattern are **unlabeled** (or invalid).

Labels:
- `IN_UP` / `IN_DOWN`: Points that are part of a valid uptrend/downtrend (roles 0, 1, 2, or 3) and overlap with adjacent trend structures. The UP/DOWN strictly indicates the direction of the pattern they belong to (e.g., UP means the segment 0->1->2->3 is an uptrend).
- `OUT_UP` / `OUT_DOWN`: Point `0` of valid up/down patterns that do *not* overlap with a previous pattern.

Outputs:
- CSV: `results/nepal/<SYMBOL>/in_out/csv/in_out_pattern_9_18.csv`
- PNG: `results/nepal/<SYMBOL>/in_out/png/in_out_pattern_9_18_visualization.png`

## 4) Chain Pattern Output (Main Algorithm)

Source: `code/algorithms/in_out/analyzer.py`

This is the main market-footprint algorithm:

1. Scan the labeled stream for immediate valid 4-point patterns.
2. Build a sequence of patterns in time order.
3. Use a sliding window: `(A, B) -> C`, then `(B, C) -> D`, etc.
4. Output dates using each pattern’s **0-point** date.

Main output:
- `results/nepal/<SYMBOL>/in_out/txt/transition_pattern_chain_9_18.txt`

## 5) Final Strategy Output (Iteration Algorithm)

Source: `code/algorithms/in_out/analyzer.py`

This algorithm predicts an outcome C based on an input sequence A + B, tracking chronological occurrences strictly across specific anchor points:

1. **Pattern A:** Begins at any valid pattern.
2. **Pattern B:** The next valid pattern starting at or after **Pattern A's point 3**.
3. **Pattern C (Result):** The next valid pattern starting at or after **Pattern B's point 2**.
4. **Intermediate Swings:** Between Pattern B's point 2 and Pattern C's start, any invalid High/Low points are tracked as `INVALID_UP` and `INVALID_DOWN`, and prefixed to the result C (e.g. `INVALID_UP -> INVALID_DOWN -> OUT_DOWN`).
5. **Next Iteration:** Pattern B becomes the new Pattern A for the next block. Pattern C (the result) is ignored when locating the subsequent inputs.

Main output:
- `results/nepal/<SYMBOL>/in_out/txt/strategy_final_pattern_9_18.txt`
