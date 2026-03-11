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

### 0. One-Command Pattern Pipeline (Recommended)
If you want everything generated in one run (EMA + IN/OUT + chain transitions):
```bash
SYMBOL=NICA
./venv/bin/python code/pipelines/run_nepal.py $SYMBOL
```

Optional refresh controls:
```bash
./venv/bin/python code/pipelines/run_nepal.py $SYMBOL --refresh auto --max-stale-days 7
```

### 1. Core Pattern Workflow (Most Important)
This is the key flow: fetch/update CSV -> draw IN/OUT pattern -> compute immediate complete-pattern transitions.

Copy-paste for any symbol:
```bash
SYMBOL=NICA
./venv/bin/python code/data_fetchers/nepal.py $SYMBOL --refresh auto
./venv/bin/python code/algorithms/custom_in_out/detector.py $SYMBOL
./venv/bin/python code/algorithms/custom_in_out/analyzer.py $SYMBOL
```

### 💹 Strategy & Recommendations (Read these first)
High-level signals generated from the patterns:
- `results/nepal/<SYMBOL>/in_out/csv/strategy_top_setups.csv` - **Best patterns found.**
- `results/nepal/<SYMBOL>/in_out/csv/strategy_recommendations.csv` - "IF/THEN" rules.
- `results/nepal/<SYMBOL>/in_out/txt/Final_strategy_9_18.txt` - **Detailed dual-pattern strategy sequences.**

Detailed documentation of the custom strategy and traversal logic can be found in:
[**code/algorithms/in_out/README.md**](file:///mnt/personal/stock/own_for_analysis/pattern-project/code/algorithms/in_out/README.md)

### 📈 Movement & History (The "How it happened")
Detailed trail of the market patterns:
- `results/nepal/<SYMBOL>/in_out/txt/Intersecting_path_9_18.txt` - Human-readable sequence (Sequential chain).

### 🔮 Forecasts & Predictions
Probability-based guesses for the next move:
- `results/nepal/<SYMBOL>/in_out/csv/forecast_next_signal.csv` - Next likely swing or pattern.
- `results/nepal/<SYMBOL>/in_out/csv/forecast_confirmed_completions.csv` - High-probability 4-point completions.

### 🔢 Statistics & Raw Data
The underlying math:
- `results/nepal/<SYMBOL>/in_out/csv/stats_token_performance.csv` - Pct move summary per label.
- `results/nepal/<SYMBOL>/in_out/csv/stats_raw_transition_matrix.csv` - Full transition database.

Algorithm details:
- See `ALGORITHM.md` for the exact logic used to draw the IN/OUT chart, how valid patterns are defined, the overlapping priority rules for trend starts (`role 0`), and how chain transitions are computed.

Note:
- `analyze_nepal_stock.py` now auto-downloads NEPSE CSV to
  `data/nepal/<SYMBOL>.csv` if local file is missing.
- Download providers (in order):
  1. GitHub company-wise source (`Aabishkar2/nepse-data`)
  2. Sharesansar full price-history fallback (`/company-price-history`)
  3. Sharesansar chart-history fallback (`/company-chart/history`) as last resort
If a symbol is missing in provider #1 (example: `HIDCLP`), provider #2 is used automatically.
Fallback coverage can be shorter than full-history datasets for some symbols.

Local-file freshness behavior:
- Default is `--refresh auto`: if local CSV is older than `--max-stale-days` (default 7), it attempts refresh.
- `--refresh always`: always try refreshing from providers.
- `--refresh never`: use local CSV only.

What is automatic:
- `pattern_detector_v2.py` reads:
  - `results/nepal/<SYMBOL>/in_out/csv/highs_lows_pattern_9_18.csv`
- It writes:
  - `results/nepal/<SYMBOL>/in_out/csv/in_out_pattern_9_18.csv`
- `transition_pattern_analysis.py` then reads that file and writes all transition reports in the same `<SYMBOL>/in_out/` folder.

Key input/output path flow:
- Input raw CSV (auto-created if missing): `data/nepal/<SYMBOL>.csv`
- Zigzag source for pattern detector: `results/nepal/<SYMBOL>/in_out/csv/highs_lows_pattern_9_18.csv`
- Labeled pattern stream: `results/nepal/<SYMBOL>/in_out/csv/in_out_pattern_9_18.csv`
- Complete-pattern transition report: `results/nepal/<SYMBOL>/in_out/txt/in_out_up_down_9_18.txt`

Pattern-level outputs from `analyzer.py`:
- `pattern_completed_sequence.csv`
  - ordered list of complete valid 4-point patterns
- `transition_pattern_path_9_18.csv` / `txt/transition_pattern_path_9_18.txt`
  - immediate `Pattern1 + Pattern2 -> [Invalid Path] -> Pattern3` counts/probabilities
- `pattern_transition_2to1_examples.csv`
  - date ranges for each example transition in human-readable month labels

If you already have `in_out_pattern_9_18.csv` and only want transition analysis:
```bash
./venv/bin/python code/pattern/transition_pattern_analysis.py NICA
```

Note on x-axis labels:
- `analyze_nepal_stock.py` charts use month names (`Jan 2026`, etc.).
- `pattern_detector_v2.py` visualization (`in_out_pattern_9_18_visualization.png`) uses adaptive date labels for readability (day + month, with automatic spacing).

### 6. Latest State Snapshot Across Symbols
Generate a one-line prediction snapshot per symbol using latest 2 clean states:
```bash
./venv/bin/python code/tools/market_state_report.py NICA HIDCL SHEL
```

Output:
- `results/nepal/market_state_report.csv`

Columns:
- `latest_prev2`: latest two clean labels, e.g. `IN_UP|IN_UP`
- `predicted_next`: most likely next label from historical transitions
- `probability`: historical probability of that next label
- `context_count`: sample size for that context
- `confidence`: simple tag (`HIGH`/`MEDIUM`/`LOW`)
- `drift_status`: if train-vs-recent top next is unchanged

## 📦 Archived Tools (Optional)
These are kept for reference and are not part of the main pattern pipeline.

- `code/archive/scan.py`
- `code/archive/nearest_history_match.py`
- `code/archive/nearest_history_walkforward.py`
  - outputs go under `results/nepal/<SYMBOL>/generic/`

## 📂 Project Structure

- `code/pattern/`: Pattern pipeline scripts (EMA -> IN/OUT -> transitions).
- `code/tools/`: One-command runners and reports.
- `code/archive/`: Legacy modules not used in the main flow.
- `stocks/`: Organized data and results per stock (`custom/` for this pipeline).
