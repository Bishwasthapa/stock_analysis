from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
from pathlib import Path

# Add project root to sys.path for robust imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from code.data_fetchers.nepal import NepalStockService
from code.algorithms.ema_viz import calculate_ema_cross, extract_zigzag_points

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

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/api/stocks/{symbol}", response_model=List[StockDataPoint])
def get_stock_data(symbol: str, years: Optional[int] = Query(None)):
    symbol = symbol.upper()
    try:
        service = NepalStockService()
        df = service.load_data(symbol)
        
        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]
            
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
def get_stock_patterns(symbol: str, years: Optional[int] = Query(None)):
    symbol = symbol.upper()
    try:
        service = NepalStockService()
        df = service.load_data(symbol)
        
        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]
            
        df = calculate_ema_cross(df)
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
    years: Optional[int] = Query(None),
    short: int = Query(9),
    long: int = Query(18),
):
    """Returns EMA short and long series for charting."""
    symbol = symbol.upper()
    try:
        service = NepalStockService()
        df = service.load_data(symbol)

        if years:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df['Date'] >= cutoff]

        df = calculate_ema_cross(df, short_span=short, long_span=long)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Handle potential NaNs in EMA calculations (e.g. at start of series)
        df = df.fillna(0)

        return {
            "short_span": short,
            "long_span": long,
            "ema_short": [{"time": r["Date"], "value": round(r["ema_short"], 2)} for _, r in df.iterrows()],
            "ema_long":  [{"time": r["Date"], "value": round(r["ema_long"],  2)} for _, r in df.iterrows()],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/stocks/{symbol}/run")
def run_pipeline(symbol: str, years: Optional[int] = Query(None)):
    """Triggers the full nepal_pipeline.py background analysis."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    pipeline_script = project_root / "code/pipelines/nepal_pipeline.py"
    python_exe = sys.executable

    cmd = [python_exe, str(pipeline_script), symbol]
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
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e.stderr}")


@app.get("/api/stocks/{symbol}/report")
def get_strategy_report(symbol: str, years: Optional[int] = Query(None)):
    """Returns the latest IN/OUT strategy report from saved CSVs."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    report_path  = project_root / f"results/nepal/{symbol}/in_out/csv/in_out_pattern_9_18.csv"
    forecast_path = project_root / f"results/nepal/{symbol}/in_out/csv/forecast_next_signal.csv"

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
def get_recommendations(symbol: str):
    """Provides structured advice and reliability scores for the dashboard."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    rec_path = project_root / f"results/nepal/{symbol}/in_out/csv/strategy_recommendations.csv"

    if not os.path.exists(rec_path):
        raise HTTPException(status_code=404, detail=f"No recommendations found for {symbol}.")

    try:
        df = pd.read_csv(rec_path).fillna('').astype(object)
        if 'count' in df.columns:
            df = df.sort_values(by='count', ascending=False)
        
        # Handle NaNs
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/strategy_txt")
def get_strategy_txt(symbol: str):
    """Returns the human-readable Final_strategy_9_18.txt report."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    txt_path = project_root / f"results/nepal/{symbol}/in_out/txt/Final_strategy_9_18.txt"

    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail=f"Strategic text report not found for {symbol}.")

    try:
        with open(txt_path, "r") as f:
            return {"symbol": symbol, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_strategy_text(content: str) -> list[dict]:
    """Parses the 'A + B -> Path -> Target' rules from Final_strategy.txt."""
    import re
    combos = []
    
    # regex to find "IN_DOWN + IN_UP:" style headers
    sections = re.split(r'\n([A-Z_]+ \+ [A-Z_]+:)\n', content)
    
    # Sections[0] is preamble, [1] is header, [2] is content, [3] is header...
    for i in range(1, len(sections), 2):
        header = sections[i].strip(':')
        body   = sections[i+1]
        
        # Regex to find individual outcomes: "-> [CHAINED] -> IN_UP | count=1/3 (33.33%)"
        outcomes = re.findall(r'-> \[(.*?)\] -> ([A-Z_]+) \| count=(\d+/\d+) \((.*?)\)', body)
        
        for path, target, ratio, prob in outcomes:
            combos.append({
                "pairing": header,
                "path": path,
                "target": target,
                "ratio": ratio,
                "probability": prob,
                "is_bull": "UP" in target
            })
            
    return combos


@app.get("/api/stocks/{symbol}/strategy_combos")
def get_strategy_combos(symbol: str):
    """Returns the parsed Double Combination results from Final_strategy.txt."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    txt_path = project_root / f"results/nepal/{symbol}/in_out/txt/Final_strategy_9_18.txt"

    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail=f"Strategic text report not found for {symbol}.")

    try:
        with open(txt_path, "r") as f:
            content = f.read()
            combos = parse_strategy_text(content)
            return combos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/structure")
def get_structure_data(symbol: str, years: Optional[int] = Query(None)):
    """Returns the categorized in/out pattern points for zigzag mapping."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    csv_path = project_root / f"results/nepal/{symbol}/in_out/csv/in_out_pattern_9_18.csv"

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
def get_visualization(symbol: str):
    """Serves the static in_out_pattern_9_18_visualization.png image."""
    symbol = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent
    viz_path = project_root / f"results/nepal/{symbol}/in_out/png/in_out_pattern_9_18_visualization.png"

    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail=f"Visualization not found for {symbol}.")

    return FileResponse(viz_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
