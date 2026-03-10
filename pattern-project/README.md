# Stock Pattern Prediction Engine

A time-series analysis tool designed to identify recurring price patterns and structural chart formations using Dynamic Time Warping (DTW) and technical indicators.

## 🚀 Technologies Used

- **Python 3.12**: Core programming language.
- **DTAIDistance**: High-performance library for Dynamic Time Warping (DTW) to match time-series shapes regardless of speed or scale.
- **Pandas & NumPy**: Data manipulation and numerical processing.
- **Matplotlib**: Generation of technical charts and pattern visualizations.
- **yfinance**: Integration with Yahoo Finance for international market data.
- **Requests**: For fetching historical NEPSE data from public GitHub repositories.

## 🛠️ Setup Instructions

1. **Clone or download** the project to your local machine.
2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install pandas numpy dtaidistance yfinance requests matplotlib
   ```

## 📈 Usage Guide

The project uses a unified scanner `scan.py`. You can run various analysis modes using flags.

### 0. One-Command Full Pipeline (Recommended)
If you want everything generated in one run (EMA + IN/OUT + transitions + nearest-history + walk-forward):
```bash
./venv/bin/python run_symbol_pipeline.py NICA
```

Just change the symbol:
```bash
./venv/bin/python run_symbol_pipeline.py HIDCLP
./venv/bin/python run_symbol_pipeline.py SRLI
```

Optional refresh controls:
```bash
./venv/bin/python run_symbol_pipeline.py NICA --refresh auto --max-stale-days 7
```

### 1. Match Against a Structural Pattern
Search for specific geometric shapes (like a "Spring") in a stock's history.
```bash
python3 scan.py --symbol NICA --market nepal --pattern spring
```

### 2. Internal Self-Discovery ("Echoes")
Find where the current 40-day price action in a stock happened before in that same stock's history.
```bash
python3 scan.py --symbol NICA --market nepal --self
```

### 3. EMA Crossover Layouts
Find historical precedents for current EMA 9/18 crossover setups.
```bash
python3 scan.py --symbol NICA --market nepal --ema
```

### 4. Visual Pattern Matching (Images & Drawings)
Match a pattern drawing directly against a chart screenshot (no CSV required).
```bash
python3 scan.py --chart_image path/to/chart.png --pattern_image path/to/pattern.png
```

### 5. IN/OUT Directional Pattern Pipeline (NICA-style flow)
Use this when you want repeating combinations like:
`IN_UP + IN_UP -> IN_DOWN` with counts and probabilities.

Run from project root:
```bash
./venv/bin/python analyze_nepal_stock.py NICA
./venv/bin/python pattern_detector_v2.py NICA
./venv/bin/python transition_pattern_analysis.py NICA
```

#Bishwas
### 5A. Core Workflow (Most Important)
This is the key project flow: fetch/update CSV -> draw IN/OUT pattern -> compute immediate complete-pattern transitions.

Copy-paste for any symbol:
```bash
SYMBOL=ACLBSL
./venv/bin/python analyze_nepal_stock.py $SYMBOL --refresh auto
./venv/bin/python pattern_detector_v2.py $SYMBOL
./venv/bin/python transition_pattern_analysis.py $SYMBOL
```

Main outputs to inspect:
- Raw/fetched CSV: `data/nepal/<SYMBOL>.csv`
- IN/OUT labeled stream: `stocks/nepal/<SYMBOL>/results/csv/in_out_pattern_9_18.csv`
- IN/OUT chart: `stocks/nepal/<SYMBOL>/results/png/in_out_pattern_9_18_visualization.png`
- Complete-pattern transition summary: `stocks/nepal/<SYMBOL>/results/txt/in_out_up_down_9_18.txt`
- Transition examples with dates: `stocks/nepal/<SYMBOL>/results/csv/pattern_transition_2to1_examples.csv`

Note:
- `analyze_nepal_stock.py` now auto-downloads NEPSE CSV to
  `data/nepal/<SYMBOL>.csv` if local file is missing.
- Download providers (in order):
  1. GitHub company-wise source (`Aabishkar2/nepse-data`)
  2. Sharesansar full price-history fallback (`/company-price-history`)
  3. Sharesansar chart-history fallback (`/company-chart/history`) as last resort
- So for a new NEPSE symbol (e.g. `HIDCL`), start with:
```bash
./venv/bin/python analyze_nepal_stock.py HIDCL
```
If a symbol is missing in provider #1 (example: `HIDCLP`), provider #2 is used automatically.
Fallback coverage can be shorter than full-history datasets for some symbols.

Local-file freshness behavior:
- Default is `--refresh auto`: if local CSV is older than `--max-stale-days` (default 7), it attempts refresh.
- `--refresh always`: always try refreshing from providers.
- `--refresh never`: use local CSV only.

Full run for `HIDCL`:
```bash
./venv/bin/python analyze_nepal_stock.py HIDCL
./venv/bin/python pattern_detector_v2.py HIDCL
./venv/bin/python transition_pattern_analysis.py HIDCL
```

What is automatic:
- `pattern_detector_v2.py` reads:
  - `stocks/nepal/<SYMBOL>/results/csv/highs_lows_pattern_9_18.csv`
- It writes:
  - `stocks/nepal/<SYMBOL>/results/csv/in_out_pattern_9_18.csv`
- `transition_pattern_analysis.py` then reads that file and writes all transition reports in the same `results/` folder.

Most human-friendly outputs:
- `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_to_next.txt`
  - grouped `prev2 -> next` combinations
- `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_priority.txt`
  - pattern-first next signal with strong-swing fallback
- `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_confirmed.txt`
  - `prev2 -> next` only when the next event confirms full 4-point completion
- `stocks/nepal/<SYMBOL>/results/txt/in_out_up_down_9_18.txt`
  - complete-pattern level transitions using immediate valid 4-point patterns:
    `IN_UP/IN_DOWN + IN_UP/IN_DOWN -> next IN_UP/IN_DOWN`
- `stocks/nepal/<SYMBOL>/results/txt/transition_readable_report_clean.md`
  - easy summary without `INVALID`

Key input/output path flow:
- Input raw CSV (auto-created if missing): `data/nepal/<SYMBOL>.csv`
- Zigzag source for pattern detector: `stocks/nepal/<SYMBOL>/results/csv/highs_lows_pattern_9_18.csv`
- Labeled pattern stream: `stocks/nepal/<SYMBOL>/results/csv/in_out_pattern_9_18.csv`
- Combination report: `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_to_next.txt`
- Priority combo report: `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_priority.txt`
- Confirmed combo report: `stocks/nepal/<SYMBOL>/results/txt/transition_clean_prev2_confirmed.txt`
- Complete-pattern transition report: `stocks/nepal/<SYMBOL>/results/txt/in_out_up_down_9_18.txt`

Pattern-level outputs from `transition_pattern_analysis.py`:
- `pattern_completed_sequence.csv`
  - ordered list of complete valid 4-point patterns
- `pattern_transition_2to1.csv` / `txt/in_out_up_down_9_18.txt`
  - immediate `Pattern1 + Pattern2 -> Pattern3` counts/probabilities
- `pattern_transition_2to1_examples.csv`
  - date ranges for each example transition in human-readable month labels

If you already have `in_out_pattern_9_18.csv` and only want transition analysis:
```bash
./venv/bin/python transition_pattern_analysis.py NICA
```

For another symbol (example `NMB`):
```bash
./venv/bin/python pattern_detector_v2.py NMB
./venv/bin/python transition_pattern_analysis.py NMB
```

Note on x-axis labels:
- `analyze_nepal_stock.py` charts use month names (`Jan 2026`, etc.).
- `pattern_detector_v2.py` visualization (`in_out_pattern_9_18_visualization.png`) uses adaptive date labels for readability (day + month, with automatic spacing).

### 6. Latest State Snapshot Across Symbols
Generate a one-line prediction snapshot per symbol using latest 2 clean states:
```bash
./venv/bin/python market_state_report.py NICA HIDCL SHEL
```

Output:
- `stocks/nepal/market_state_report.csv`

Columns:
- `latest_prev2`: latest two clean labels, e.g. `IN_UP|IN_UP`
- `predicted_next`: most likely next label from historical transitions
- `probability`: historical probability of that next label
- `context_count`: sample size for that context
- `confidence`: simple tag (`HIGH`/`MEDIUM`/`LOW`)
- `drift_status`: if train-vs-recent top next is unchanged

### 7. Nearest History Match (Price Pattern)
Find historical windows most similar to the latest price window and summarize what happened next.
```bash
./venv/bin/python nearest_history_match.py NICA
```

Optional params:
```bash
./venv/bin/python nearest_history_match.py NICA --window 40 --horizon 10 --top-k 12
```

Output folder:
- `stocks/nepal/<SYMBOL>/results/nearest_history_match/`

Files:
- `nearest_matches.csv`: detailed match rows
- `nearest_match_report.md`: human-readable summary

How to read one match row (example):
- `| 1 | 2014-10-08 -> 2014-12-08 | 3.5628 | -12.35 | 7.38 | -12.48 |`
- Meaning:
  - `Rank 1`: most similar historical shape to current window
  - `2014-10-08 -> 2014-12-08`: historical segment dates
  - `3.5628`: shape distance (lower = more similar)
  - `-12.35`: forward return over chosen horizon
  - `7.38`: best upside seen during the horizon
  - `-12.48`: worst downside seen during the horizon

### 8. Robust Walk-Forward Forecast (Recommended)
This is the most complete module in the repo right now. It includes:
- model auto-selection (`euclidean` vs `dtw`, regime filter on/off)
- walk-forward validation (no lookahead leakage)
- uncertainty bands (`P10/P50/P90`)
- probability calibration
- cost-aware backtest (fees + slippage)
- human-readable summary

Run:
```bash
./venv/bin/python nearest_history_walkforward.py NICA --distance auto --regime-filter auto --horizons 5,10,20
```

Output folder:
- `stocks/nepal/<SYMBOL>/results/nearest_history_walkforward/`

Important files:
- `model_selection.csv`: which model variant was best on validation
- `walkforward_summary.csv`: forecast quality by horizon
- `backtest_summary.csv`: gross/net strategy stats with costs
- `calibration_table.csv`: probability bins vs actual up-rate
- `calibration_summary.csv`: Brier score before/after calibration
- `latest_forecast.csv`: latest forecast values
- `walkforward_report.md`: technical consolidated report
- `human_readable_summary.md`: plain-language short report

How to read calibration quickly:
- In `calibration_table.csv`:
  - `raw_avg_prob`: average model probability in a bin
  - `empirical_up_rate`: actual observed up-rate in that bin
  - Closer values = better calibration
- In `calibration_summary.csv`:
  - `brier_raw` vs `brier_calibrated` (lower is better)
  - If calibrated is lower, calibration improved reliability

How to read cost-aware backtest quickly:
- In `backtest_summary.csv`:
  - `avg_gross_trade_ret_pct`: before costs
  - `avg_trade_ret_pct`: after costs (net)
  - `cum_gross_strategy_ret_pct` vs `cum_strategy_ret_pct`: gross vs net total
  - `roundtrip_cost_pct`: per-trade assumed total cost

## 📂 Project Structure

- `core/`: Contains the `PatternEngine` (the analytical brain).
- `patterns/`: Library of reference pattern drawings.
- `stocks/`: Organized data and results per stock.
  - `nepal/NICA/`: Contextual screenshots and saved analysis charts.
- `scan.py`: The main command-line interface.

## 📊 Understanding the Results

When you run a scan, the charts saved in the `results/` folder have specific meanings:

1.  **`structural_matches.png`**:
    - **What it is**: Matches against a "Template" (like your `Pattern.png` drawing).
    - **Use Case**: "Show me every time NICA did the 'Spring' move I drew."
2.  **`self_echoes.png`**:
    - **What it is**: Internal self-similarity (Fractal echoes).
    - **Use Case**: "Take NICA's move from the last 40 days and find where NICA did the EXACT same move in its own past."
3.  **`ema_repeats.png`**:
    - **What it is**: EMA Layout Similarity.
    - **Use Case**: "Find historical spots where the EMA 9 and 18 crossovers were in the same position as they are now."

## 📊 Summary
Analysis results are saved as PNG charts within each stock's directory.
