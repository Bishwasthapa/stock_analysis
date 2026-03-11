# In-Out Pattern Algorithm & Strategy Engine

This directory contains the core logic for the "In-Out" market structure analysis, specifically tuned for NEPSE but market-agnostic.

## 1. The "In-Out" Algorithm (detector.py)

The algorithm identifies market regimes by classifying 4-point structural patterns (Low-High-Low-High for Up, High-Low-High-Low for Down).

### Pattern Classification (Sequential Traversal)
Patterns are identified and labeled through a rigorous chronological scan of the market zigzag:
- **OUT**: A pattern is labeled **OUT** if its start point (0) is structurally independent from the immediately preceding pattern in the scan. This signals a new market structural anchor.
- **IN**: A pattern is labeled **IN** if its start point (0) coincides with points **1, 2, or 3** of the preceding pattern. This identifies a continuous extension or "chaining" of momentum.

## 2. The Strategy Engine (analyzer.py)

The strategy layer builds a predictive model by treating the market as a sliding sequence of these patterns.

### The Sliding Window: (input1 + input2) -> Outcome
The engine uses a 2-pattern context window to predict the next structural event:
- **input1**: First valid 4-point pattern.
- **input2**: Second valid 4-point pattern.
  - **Constraint**: Input patterns cannot **intersect** (no shared internal points). They can only **chain** (`point 3 of A == point 0 of B`).
- **Iteration**: For the next step, `input2` becomes the new `input1`.

### Prediction Priority
The engine prioritizes high-fidelity structural completions over simple directional moves:
1. **Priority 1 (Pattern)**: The engine searches for a valid 4-point pattern that **intersects or chains** with `input2`. If found, the result is recorded as `IN_UP` or `IN_DOWN`.
2. **Priority 2 (Pattern)**: Search for the next separate/sequential valid pattern. If found, result is `OUT_UP` or `OUT_DOWN`.
3. **Priority 3 (Swing)**: If no pattern completion is visible before the end of data or a significant timeline gap, the output is recorded as a **directional swing** (`SWING_UP` or `SWING_DOWN`).

## 3. Usage & Reports

### Running the Analysis
```bash
# Analyze a specific symbol over the last 5 years
./venv/bin/python code/pipelines/run_nepal.py <SYMBOL> --years 5
```

### Key Strategy Outputs
- **Intersecting_path_9_18.txt**: A narrative log of the market's path through its structural transitions (Sequential chain).
- **Final_strategy_9_18.txt**: The statistical breakdown of `(A+B -> Result)` probability matrix and chronological strategy sequences.
- **strategy_recommendations.csv**: Actionable IF/THEN rules derived from the highest-probability historical transitions.

## 4. Visualization
- All charts use a **Dark Theme** (`#0d1117`) for clarity.
- **Green/Blue**: Bullish context/extension.
- **Amber/Red**: Bearish context/extension.
- **Labels (0-3)**: Identify the internal role of each swing point within its respective pattern.
