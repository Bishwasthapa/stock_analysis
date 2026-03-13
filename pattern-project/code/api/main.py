from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path for robust imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from code.data_fetchers.nepal import NepalStockService
from code.algorithms.ema_viz import calculate_ema_cross, extract_zigzag_points
import glob
import subprocess

app = FastAPI(title="Nepal Stock Pattern API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; refine for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

class StockDataPoint(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: Optional[float] = None

class PatternPoint(BaseModel):
    date: str
    price: float
    type: str  # 'high' or 'low'
    cross_type: str

# Result Path Helper
def get_result_dir(market: str, symbol: str, strategy: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "results" / market.lower() / symbol.upper() / strategy.lower()

def _round_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Round OHLC price columns to 1 decimal place for consistent analysis."""
    for col in ('Open', 'High', 'Low', 'Close'):
        if col in df.columns:
            df[col] = df[col].round(1)
    return df


@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/api/stocks/{symbol}", response_model=List[StockDataPoint])
def get_stock_data(symbol: str, market: str = Query("nepal"), years: Optional[int] = Query(None), round_prices: bool = Query(True)):
    symbol = symbol.upper()
    try:
        # TODO: Add Market Factory here later (e.g. YahooFinanceService for 'intl')
        service = NepalStockService()
        df = service.load_data(symbol)
        
        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]

        if round_prices:
            df = _round_ohlc(df)
            
        # Convert to standard format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        # Respecting our standard OHLCV mapping
        result = []
        for _, row in df.iterrows():
            result.append(StockDataPoint(
                Date=row['Date'],
                Open=row['Open'],
                High=row['High'],
                Low=row['Low'],
                Close=row['Close'],
                Volume=row.get('traded_quantity', row.get('Volume'))
            ))
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/stocks/{symbol}/patterns", response_model=List[PatternPoint])
def get_stock_patterns(symbol: str, market: str = Query("nepal"), years: Optional[int] = Query(None), threshold: float = Query(0.0), round_prices: bool = Query(True)):
    symbol = symbol.upper()
    try:
        service = NepalStockService()
        df = service.load_data(symbol)
        
        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]

        if round_prices:
            df = _round_ohlc(df)
            
        df = calculate_ema_cross(df, threshold=threshold)
        points = extract_zigzag_points(df)
        
        return [PatternPoint(
            date=p['date'].strftime('%Y-%m-%d'),
            price=p['price'],
            type=p['type'],
            cross_type=p['cross_type']
        ) for p in points]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{symbol}/ema")
def get_ema_data(
    symbol: str,
    market: str = Query("nepal"),
    years: Optional[int] = Query(None),
    short: int = Query(9),
    long: int = Query(18),
    threshold: float = Query(0.0),
    round_prices: bool = Query(True)
):
    """Returns EMA short and long series for charting."""
    symbol = symbol.upper()
    try:
        service = NepalStockService()
        df = service.load_data(symbol)

        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]

        if round_prices:
            df = _round_ohlc(df)
        df = calculate_ema_cross(df, short_span=short, long_span=long, threshold=threshold)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Handle potential NaNs in EMA calculations (e.g. at start of series)
        df = df.fillna(0)

        return {
            "short_span": short,
            "long_span": long,
            "ema_short": [{"time": r["Date"], "value": round(r["ema_short"], 1)} for _, r in df.iterrows()],
            "ema_long":  [{"time": r["Date"], "value": round(r["ema_long"],  1)} for _, r in df.iterrows()],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{symbol}/run")
def run_pipeline(
    symbol: str, 
    market: str = Query("nepal"), 
    strategy: str = Query("in_out"),
    years: Optional[int] = Query(None), 
    threshold: float = Query(0.0)
):
    """Triggers the full global pipeline background analysis."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    # Use the new global pipeline
    pipeline_script = project_root / "code/pipelines/global_pipeline.py"
    python_exe = sys.executable

    cmd = [
        python_exe, str(pipeline_script), symbol, 
        "--market", market,
        "--strategy", strategy,
        "--threshold", str(threshold)
    ]
    if years:
        cmd += ["--years", str(years)]

    # Run the pipeline script
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return {"status": "success", "message": f"Pipeline complete for {symbol}", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        error_msg = f"Pipeline failed: {e.stderr or e.stdout}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/stocks/{symbol}/report")
def get_stock_report(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out"), years: Optional[int] = Query(None)):
    """Returns the latest strategy report from saved CSVs."""
    symbol = symbol.upper()
    base_dir = get_result_dir(market, symbol, strategy)
    # The detector currently saves everything as in_out_pattern_9_18.csv in the strategy folder
    report_path = base_dir / "csv" / "in_out_pattern_9_18.csv"
    forecast_path = base_dir / "csv" / "forecast_next_signal.csv"

    result = {}

    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        if years:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                df = df[df['date'] >= cutoff]
        tail = df.tail(20).fillna('').astype(object)
        result["pattern_data"] = tail.to_dict(orient="records")

    if os.path.exists(forecast_path):
        fc = pd.read_csv(forecast_path).fillna('').astype(object)
        result["forecast"] = fc.head(5).to_dict(orient="records")

    if not result:
        raise HTTPException(status_code=404, detail=f"No report found for {symbol}. Run the full pipeline first.")

    return result


@app.get("/api/stocks/{symbol}/recommendations")
def get_recommendations(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out")):
    """Provides structured advice and reliability scores for the dashboard."""
    symbol = symbol.upper()
    base_dir = get_result_dir(market, symbol, strategy) / "csv"
    rec_path = base_dir / "strategy_recommendations.csv"
    stats_path = base_dir / "stats_context_summary.csv"

    if not os.path.exists(rec_path):
        raise HTTPException(status_code=404, detail=f"No recommendations found for {symbol}.")

    try:
        # Load best recommendations
        df_rec = pd.read_csv(rec_path).fillna('').astype(object)
        recs = df_rec.to_dict(orient="records")

        # Load context stats for overall health
        health_label = "MODERATE"
        if os.path.exists(stats_path):
            df_stats = pd.read_csv(stats_path)
            # Filter for contexts with some volume (at least 5 samples)
            valid_contexts = df_stats[df_stats['total_context_count'] >= 5]
            if not valid_contexts.empty:
                avg_entropy = valid_contexts['entropy'].mean()
                if avg_entropy < 0.7: health_label = "STRONG"
                elif avg_entropy > 1.2: health_label = "WEAK"
            else:
                # Fallback to general entropy if no high-volume contexts
                avg_entropy = df_stats['entropy'].mean() if not df_stats.empty else 1.0
                if avg_entropy < 0.7: health_label = "STRONG"
                elif avg_entropy > 1.2: health_label = "WEAK"

        # Embed health label in the first rec or return separately
        # For simplicity, we add it to a wrapper object
        return {
            "health": health_label,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/strategy_txt")
def get_strategy_txt(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out")):
    """Returns the human-readable Final_strategy report."""
    symbol = symbol.upper()
    txt_path = get_result_dir(market, symbol, strategy) / "txt" / "Final_strategy_9_18.txt"

    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail=f"Strategic text report not found for {symbol}.")

    try:
        with open(txt_path, "r") as f:
            return {"symbol": symbol, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import math

def _wilson_score(count: int, total: int, z: float = 1.96) -> float:
    """Wilson Score lower confidence bound (95%). Penalises small samples.
    Returns a value in [0, 1] — higher means more trustworthy."""
    if total == 0:
        return 0.0
    p_hat = count / total
    denom = 1 + z * z / total
    centre = p_hat + z * z / (2 * total)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total))
    return (centre - spread) / denom


def parse_strategy_text(content: str) -> list[dict]:
    """Parses the 'A + B -> Path -> Target' rules from Final_strategy.txt.
    Each combo is enriched with:
      - wilson_score  : lower confidence bound (95%) — used for ranking
      - adjusted_prob : Laplace-smoothed probability for honest display
    """
    import re
    combos = []
    
    # Splitting by "INPUT1 + INPUT2:" style headers (at start of line)
    sections = re.split(r'^\s*([A-Z_]+\s*\+\s*[A-Z_]+:)\s*$', content, flags=re.MULTILINE)
    
    for i in range(1, len(sections), 2):
        header = sections[i].strip().strip(':')
        body   = sections[i+1]
        
        # Match outcomes WITH a path: "-> [PATH] -> TARGET | count=N/T (X%)"
        outcomes_with_path = re.findall(
            r'^\s*-\>\s+\[(.*?)\]\s+\-\>\s+([A-Z_]+)\s+\| count=(\d+)/(\d+) \(.*?\)', body, re.MULTILINE
        )
        # Match DIRECT outcomes (no path): "-> TARGET | count=N/T (X%)"
        outcomes_direct = re.findall(
            r'^\s*-\>\s+([A-Z_]+)\s+\| count=(\d+)/(\d+) \(.*?\)', body, re.MULTILINE
        )

        # Match profit buckets: "-> At Least +5% Profit reached | count=7/7 (100.0%)"
        profit_hits = re.findall(
            r'-\>\s+At Least\s+\+(.*?)%\s+Profit reached\s+\| count=(\d+)/(\d+) \((.*?)%\)', body
        )
        profit_map = [f"+{p}%: {prob}%" for p, c, t, prob in profit_hits]

        for path, target, cnt_s, tot_s in outcomes_with_path:
            cnt, tot = int(cnt_s), int(tot_s)
            raw_pct   = (cnt / tot * 100) if tot else 0.0
            adj_pct   = ((cnt + 1) / (tot + 2) * 100)  # Laplace smoothing
            ws        = _wilson_score(cnt, tot)
            combos.append({
                "pairing"           : header,
                "path"              : path,
                "target"            : target,
                "ratio"             : f"{cnt}/{tot}",
                "probability"       : f"{raw_pct:.1f}%",
                "adjusted_probability": f"{adj_pct:.1f}%",
                "wilson_score"      : round(ws * 100, 1),
                "is_bull"           : "UP" in target,
                "profit_stats"      : profit_map
            })

        for target, cnt_s, tot_s in outcomes_direct:
            cnt, tot = int(cnt_s), int(tot_s)
            raw_pct   = (cnt / tot * 100) if tot else 0.0
            adj_pct   = ((cnt + 1) / (tot + 2) * 100)
            ws        = _wilson_score(cnt, tot)
            combos.append({
                "pairing"           : header,
                "path"              : "",
                "target"            : target,
                "ratio"             : f"{cnt}/{tot}",
                "probability"       : f"{raw_pct:.1f}%",
                "adjusted_probability": f"{adj_pct:.1f}%",
                "wilson_score"      : round(ws * 100, 1),
                "is_bull"           : "UP" in target,
                "profit_stats"      : profit_map
            })
            
    return combos


@app.get("/api/stocks/{symbol}/strategy_combos")
def get_strategy_combos(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out")):
    """Returns parsed Double Combination results, sorted by Wilson Score (reliable first)."""
    symbol = symbol.upper()
    txt_path = get_result_dir(market, symbol, strategy) / "txt" / "Final_strategy_9_18.txt"

    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail=f"Strategic text report not found for {symbol}.")

    try:
        with open(txt_path, "r") as f:
            content = f.read()
        combos = parse_strategy_text(content)
        # Sort: highest Wilson Score first (most statistically reliable)
        combos.sort(key=lambda x: x["wilson_score"], reverse=True)
        return combos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/structure")
def get_structure_data(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out"), years: Optional[int] = Query(None)):
    """Returns the categorized in/out pattern points for zigzag mapping."""
    symbol = symbol.upper()
    csv_path = get_result_dir(market, symbol, strategy) / "csv" / "in_out_pattern_9_18.csv"

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Structure data not found for {symbol}.")

    try:
        df = pd.read_csv(csv_path)
        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= cutoff]
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Handle NaNs for JSON serialization
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/viz")
def get_visualization(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out")):
    """Serves the static pattern visualization image."""
    symbol = symbol.upper()
    viz_path = get_result_dir(market, symbol, strategy) / "png" / f"{strategy}_pattern_9_18_visualization.png"

    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail=f"Visualization not found for {symbol}.")

    return FileResponse(viz_path)


@app.get("/api/stocks/{symbol}/forecast")
def get_stock_forecast(symbol: str, market: str = Query("nepal"), strategy: str = Query("in_out")):
    """Returns the live pattern forecast JSON generated by analyzer.py."""
    symbol = symbol.upper()
    forecast_path = get_result_dir(market, symbol, strategy) / "csv" / "current_forecast.json"

    if not os.path.exists(forecast_path):
        return {
            "symbol": symbol,
            "current_setup": "UNKNOWN",
            "last_pattern_date": "N/A",
            "total_historical_matches": 0,
            "outcomes": [],
            "error": "No forecast data generated yet. Run Full Analysis first."
        }

    try:
        with open(forecast_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Intelligence Section (Brokerage Analysis) ---

@app.post("/api/intelligence/broker-analysis")
def trigger_broker_analysis(
    side: str = Query("buyer"), 
    num_brokers: int = Query(5), 
    num_stocks: int = Query(5),
    turnover: int = Query(30),
    date: str = Query(None)
):
    """Triggers the Brokerage Intelligence analysis script with dynamic settings."""
    project_root = Path(__file__).resolve().parent.parent.parent
    runner_script = project_root / "run_broker_analysis.sh"
    
    # Pass all dynamic parameters to analyze_floorsheet.py
    cmd = [
        str(runner_script), "analyze_floorsheet.py", 
        f"--num-brokers={num_brokers}",
        f"--limit={num_stocks}",
        f"--turnover={turnover}"
    ]
    if side == "seller":
        cmd.append("--seller")
    else:
        cmd.append("--buyer")
        
    if date:
        cmd.append(f"--discovery-date={date}")

    try:
        # Run and wait for completion (it takes ~10-20s for scraping)
        env = os.environ.copy()
        # Use /bin/bash explicitly to ensure POSIX compliance with '.' command in the runner
        result = subprocess.run(["/bin/bash"] + cmd, env=env, capture_output=True, text=True, check=True, cwd=str(project_root))
        
        return {
            "status": "success", 
            "message": "Brokerage Analysis Complete",
            "output_preview": result.stdout[-500:] # Last 500 chars
        }
    except subprocess.CalledProcessError as e:
        error_output = e.stderr or e.stdout
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_output}")


@app.get("/api/intelligence/latest-report")
def get_latest_broker_report(date: str = Query(None)):
    """Retrieves the content of the most recent brokerage analysis report, optionally filtered by date."""
    project_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = project_root / "results" / "broker_analysis"
    
    if not reports_dir.exists():
         raise HTTPException(status_code=404, detail="No reports directory found.")
         
    # Find files. If date is provided, filter by report-{date}_*.txt
    pattern = f"report-{date}_*.txt" if date else "report-*.txt"
    report_files = glob.glob(str(reports_dir / pattern))
    
    if not report_files:
        raise HTTPException(status_code=404, detail=f"No broker reports found {'for date ' + date if date else ''}.")
        
    latest_report = max(report_files, key=os.path.getmtime)
    
    try:
        with open(latest_report, "r") as f:
            return {
                "filename": os.path.basename(latest_report),
                "content": f.read()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read report: {str(e)}")


@app.get("/api/intelligence/reports")
def list_broker_reports():
    """Returns a list of all available brokerage reports."""
    project_root = Path(__file__).resolve().parent.parent.parent
    reports_dir = project_root / "results" / "broker_analysis"
    
    if not reports_dir.exists():
        return []
        
    report_files = glob.glob(str(reports_dir / "report-*.txt"))
    reports = []
    for f in report_files:
        reports.append({
            "name": os.path.basename(f),
            "created_at": os.path.getmtime(f)
        })
    
    # Sort by creation time descending
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return reports




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
