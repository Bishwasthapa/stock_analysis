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

### 📈 Web Dashboard (Interactive Chart Portal)

The system includes a premium, dark-mode interactive dashboard based on Lightweight Charts:
- **TradingView Experience**: Smooth zooming, panning, and crosshair interactions.
- **OHLCV Hover Legend**: Real-time display of Open, High, Low, Close, and Volume data on hover.
- **Structural Path Visualization**: High-contrast white dotted line connecting algorithmic swing points.
- **Noise Highlighting**: Red circles for "OUT" (transitional noise) and Orange squares for "INVALID" (structural breaks).
- **Strategy Scorecard**: Full transparency into sequence probabilities and technical rules.

### Start the Portal
```bash
./start_portal.sh
```
Open your browser at `http://localhost:8000`.

## 📂 Key Folders
- `data/nepal/`: Raw split-adjusted CSVs.
- `results/nepal/<SYMBOL>/in_out/`:
  - `png/`: Beautiful dark-mode pattern visualizations.
  - `csv/`: Transition matrices, strategy recommendations, and forecasts.
  - `txt/`: Human-readable sequence chains (`Final_strategy_9_18.txt`).

## 📐 Statistical Methods

The **Strategy Scorecard** uses two techniques to surface reliable patterns:
- **Wilson Score (95% CI)**: Ranks combos by their *lower confidence bound* to reward consistency over luck.
- **Laplace Smoothing**: Avoids 100% probabilities from single observations by adding a small prior.

---
- **[ALGORITHM.md](file:///mnt/personal/stock/own_for_analysis/pattern-project/ALGORITHM.md)**: Technical details on pattern detection.
- **[DEPLOYMENT_GUIDE.md](file:///mnt/personal/stock/own_for_analysis/pattern-project/deployment_guide.md)**: Steps for hosting on a private server.
- **[BROKER_INTELLIGENCE.md](file:///mnt/personal/stock/own_for_analysis/pattern-project/code/broker_analysis/BROKER_INTELLIGENCE.md)**: tracing Smart Money signals.
