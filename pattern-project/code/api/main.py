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

        return {
            "short_span": short,
            "long_span": long,
            "ema_short": [{"time": r["Date"], "value": round(r["ema_short"], 2)} for _, r in df.iterrows()],
            "ema_long":  [{"time": r["Date"], "value": round(r["ema_long"],  2)} for _, r in df.iterrows()],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
