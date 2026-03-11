# Nepal Stock Pattern Engine 🚀

A high-performance technical analysis suite for identifying structural price patterns (IN/OUT) and EMA crossovers in the Nepal Stock Market.

## 🛠️ Quick Setup

1. **Venv**: `python3 -m venv venv && source venv/bin/activate`
2. **Install**: `pip install -r requirements.txt` (requires `curl_cffi` for automated bypassing).

## 📈 Main Usage Styles

### 1. Full Analysis Pipeline (Recommended)
Run the complete end-to-end analysis (Fetch -> EMA -> Detector -> Strategy).
```bash
python code/pipelines/nepal_pipeline.py NICA --years 5
```

### 2. Just the Data (Unified Fetcher)
Automatically download split-adjusted OHLCV data from NepseAlpha/Sharesansar.
```bash
python code/data_fetchers/nepal.py NICA
```

### 3. Just the Charts (EMA Viz)
Generate dark-mode EMA crossover and Zigzag charts for local data.
```bash
python code/algorithms/ema_viz.py NICA --years 2
```

### 4. Web Dashboard (Interactive Chart Portal)
Start the local API server, then open the browser at `http://localhost:8000`.
```bash
# From the project root
nohup venv/bin/uvicorn code.api.main:app --host 0.0.0.0 --port 8000 >> /tmp/api.log 2>&1 &
```
> **Note:** The server is **not** persistent across reboots — run the command above each time to start it.  
> To stop it: `pkill -f uvicorn`


## 📂 Key Folders
- `data/nepal/`: Raw split-adjusted CSVs.
- `results/nepal/<SYMBOL>/in_out/`:
  - `png/`: Beautiful dark-mode pattern visualizations.
  - `csv/`: Transition matrices, strategy recommendations, and forecasts.
  - `txt/`: Human-readable sequence chains (`Final_strategy_9_18.txt`).

---
*For technical details on the pattern detection logic, see [ALGORITHM.md](file:///mnt/personal/stock/own_for_analysis/pattern-project/ALGORITHM.md).*
