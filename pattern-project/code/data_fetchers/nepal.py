import os
import re
import json
import pandas as pd
import requests
from datetime import datetime, UTC
import numpy as np
import sys
from pathlib import Path

class NepalStockService:
    """
    Unified service for fetching and managing Nepal stock market data.
    Priority: Local JSON -> Local CSV (if fresh) -> NepseAlpha (Adjusted) -> Fallbacks.
    """
    
    def __init__(self, data_dir: str = 'data/nepal', max_stale_days: int = 0):
        self.data_dir = data_dir
        self.max_stale_days = max_stale_days
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self, symbol: str, refresh_mode: str = 'auto') -> pd.DataFrame:
        """
        Public interface to load or fetch stock data for a given symbol.
        """
        symbol = symbol.upper()
        json_path = os.path.join(self.data_dir, f'{symbol}.json')
        csv_path = os.path.join(self.data_dir, f'{symbol}.csv')

        # 1. Process Raw JSON if present
        if os.path.exists(json_path):
            self._convert_json_to_csv(symbol, json_path, csv_path)

        # 2. Check local CSV and staleness
        local_exists = os.path.exists(csv_path)
        is_manual = self._is_manual_export(csv_path) if local_exists else False

        should_fetch = False
        if not local_exists:
            should_fetch = True
        elif is_manual:
            print(f"  Detected manual CSV export for {symbol}. Preserving adjusted data.")
        elif refresh_mode == 'always':
            should_fetch = True
        elif refresh_mode == 'auto':
            latest_date = self._get_latest_date(csv_path)
            if latest_date is None:
                should_fetch = True
            else:
                age = (datetime.now(UTC).date() - latest_date).days
                if age > self.max_stale_days:
                    print(f"  Local data stale ({age} days old). Attempting refresh...")
                    should_fetch = True

        if should_fetch:
            success = self._fetch_all_providers(symbol, csv_path)
            if not success and not local_exists:
                raise FileNotFoundError(f"Failed to fetch data for {symbol} and no local cache exists.")

        # 3. Load and Normalize
        df = pd.read_csv(csv_path)
        return self._normalize_dataframe(df, symbol)

    def _convert_json_to_csv(self, symbol: str, json_path: str, csv_path: str):
        """Convert a raw NepseAlpha JSON network response to our standard CSV."""
        print(f"  Converting raw NepseAlpha JSON for {symbol}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get("s") != "ok" or not data.get("t"):
                return

            t, o, h, l, c, v = data["t"], data["o"], data["h"], data["l"], data["c"], data["v"]
            n = min(len(t), len(o), len(h), len(l), len(c), len(v))
            
            dates = [datetime.fromtimestamp(ts, UTC).date().isoformat() for ts in t[:n]]
            close_series = pd.Series(c[:n], dtype=float)
            
            df = pd.DataFrame({
                "published_date": dates,
                "open": pd.Series(o[:n], dtype=float),
                "high": pd.Series(h[:n], dtype=float),
                "low": pd.Series(l[:n], dtype=float),
                "close": close_series,
                "per_change": (close_series.pct_change() * 100.0).round(4),
                "traded_quantity": pd.Series(v[:n], dtype=float),
                "traded_amount": (pd.Series(v[:n], dtype=float) * close_series).round(2),
                "status": 0
            })
            df.to_csv(csv_path, index=False)
            print(f"  Successfully converted JSON to CSV: {symbol} ({n} rows).")
        except Exception as e:
            print(f"  Error converting JSON: {e}")

    def _is_manual_export(self, path: str) -> bool:
        """Check if a CSV is a manual export that should not be overwritten."""
        try:
            df = pd.read_csv(path, nrows=5)
            # Differentiate by presence of 'published_date' (our standard) vs 'time/date' (TV/Alpha exports)
            cols = [c.lower().strip() for c in df.columns]
            if any(c in ['time', 'date'] for c in cols) and 'published_date' not in cols:
                return True
        except: pass
        return False

    def _get_latest_date(self, path: str):
        """Extract the latest date from a local CSV."""
        try:
            df = pd.read_csv(path)
            for col in ['published_date', 'date', 'Date', 'time']:
                if col in df.columns:
                    d = pd.to_datetime(df[col], errors='coerce').dropna()
                    if not d.empty: return d.max().date()
        except: pass
        return None

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize column names and types for the analysis engine."""
        df = df.copy()
        cols = {c.lower().strip(): c for c in df.columns}
        
        # Mapping variants to standard keys
        mapping = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open'}
        for low, high in mapping.items():
            if low in cols:
                df.rename(columns={cols[low]: high}, inplace=True)
            elif high not in df.columns:
                # Fallback if specific column missing
                if high == 'Close': raise KeyError(f"Missing Close column for {symbol}")
                df[high] = df['Close'] # Link H/L to Close if missing

        # Standardize Date
        date_col = next((cols[k] for k in ['published_date', 'date', 'time'] if k in cols), None)
        if not date_col: raise KeyError(f"No date column found for {symbol}")
        
        df['Date'] = pd.to_datetime(df[date_col])
        return df.sort_values('Date').reset_index(drop=True)

    def _fetch_all_providers(self, symbol: str, path: str) -> bool:
        """Try providers in sequence until one succeeds."""
        providers = [
            self._fetch_nepsealpha,
            self._fetch_sharesansar_history,
            self._fetch_sharesansar_udf,
            self._fetch_github
        ]
        for p in providers:
            if p(symbol, path): return True
        return False

    def _fetch_nepsealpha(self, symbol: str, path: str) -> bool:
        """NepseAlpha Adjusted API via curl_cffi bypass."""
        try:
            from curl_cffi import requests as cffi_requests
            url = f"https://nepsealpha.com/trading/1/history?symbol={symbol}&resolution=1D&frame=1"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://nepsealpha.com/nepse-chart",
                "X-Requested-With": "XMLHttpRequest"
            }
            resp = cffi_requests.get(url, headers=headers, impersonate="chrome120", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("s") == "ok":
                    t, o, h, l, c, v = data["t"], data["o"], data["h"], data["l"], data["c"], data["v"]
                    n = min(len(t), len(o), len(h), len(l), len(c), len(v))
                    df = pd.DataFrame({
                        "published_date": [datetime.fromtimestamp(ts, UTC).date().isoformat() for ts in t[:n]],
                        "open": o[:n], "high": h[:n], "low": l[:n], "close": c[:n],
                        "per_change": (pd.Series(c[:n]).pct_change() * 100).round(4),
                        "traded_quantity": v[:n],
                        "traded_amount": (pd.Series(v[:n]) * pd.Series(c[:n])).round(2),
                        "status": 0
                    })
                    df.to_csv(path, index=False)
                    print(f"  Fetched {symbol} from NepseAlpha (Adjusted)")
                    return True
        except: pass
        return False

    def _fetch_sharesansar_history(self, symbol: str, path: str) -> bool:
        """Secondary provider: Sharesansar company-price-history DataTable endpoint."""
        company_url = f"https://www.sharesansar.com/company/{symbol}"
        history_url = "https://www.sharesansar.com/company-price-history"
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            page = session.get(company_url, timeout=20, headers=headers)
            if page.status_code != 200: return False

            html = page.text
            company_match = re.search(r"id=['\"]companyid['\"][^>]*>([^<]+)<", html)
            token_match = re.search(r"name=['\"]_token['\"][^>]*content=['\"]([^'\"]+)", html)
            if not company_match or not token_match: return False
            
            company_id = company_match.group(1).strip()
            csrf_token = token_match.group(1)

            post_headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": company_url,
                "Origin": "https://www.sharesansar.com",
                "X-CSRF-Token": csrf_token,
                "X-Requested-With": "XMLHttpRequest",
            }

            page_size = 50
            first_payload = {"company": company_id, "draw": 1, "start": 0, "length": page_size, "adjusted": 1}
            first = session.post(history_url, data=first_payload, headers=post_headers, timeout=20)
            if first.status_code != 200: return False
            
            first_json = first.json()
            total = int(first_json.get("recordsTotal", 0))
            data_rows = list(first_json.get("data", []))
            if total == 0: return False

            start = page_size
            draw = 2
            while start < total:
                payload = {"company": company_id, "draw": draw, "start": start, "length": page_size, "adjusted": 1}
                resp = session.post(history_url, data=payload, headers=post_headers, timeout=20)
                if resp.status_code != 200: break
                chunk = resp.json().get("data", [])
                if not chunk: break
                data_rows.extend(chunk)
                start += page_size
                draw += 1

            cleaned = []
            for r in data_rows:
                cleaned.append({
                    "published_date": str(r.get("published_date", "")).strip(),
                    "open": _to_float(r.get("open")),
                    "high": _to_float(r.get("high")),
                    "low": _to_float(r.get("low")),
                    "close": _to_float(r.get("close")),
                    "per_change": _to_float(r.get("per_change")),
                    "traded_quantity": _to_float(r.get("traded_quantity")),
                    "traded_amount": _to_float(r.get("traded_amount")),
                    "status": 0
                })
            
            df = pd.DataFrame(cleaned)
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
            df = df.dropna(subset=["published_date", "close"]).sort_values("published_date")
            df["published_date"] = df["published_date"].dt.strftime("%Y-%m-%d")
            df.to_csv(path, index=False)
            print(f"  Fetched {symbol} from Sharesansar (Adjusted Table)")
            return True
        except: return False

    def _fetch_sharesansar_udf(self, symbol: str, path: str) -> bool:
        """Secondary provider: Sharesansar TradingView-UDF history endpoint."""
        url = "https://www.sharesansar.com/company-chart/history"
        params = {"symbol": symbol, "resolution": "1D", "from": 0, "to": 9999999999, "countback": 5000, "adjusted": "true"}
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://www.sharesansar.com/company/{symbol}"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code != 200: return False
            p = resp.json()
            if p.get("s") != "ok" or not p.get("t"): return False

            t, o, h, l, c, v = p["t"], p["o"], p["h"], p["l"], p["c"], p["v"]
            n = min(len(t), len(o), len(h), len(l), len(c), len(v))
            
            close_s = pd.Series(c[:n], dtype=float)
            df = pd.DataFrame({
                "published_date": [datetime.fromtimestamp(ts, UTC).date().isoformat() for ts in t[:n]],
                "open": o[:n], "high": h[:n], "low": l[:n], "close": c[:n],
                "per_change": (close_s.pct_change() * 100).round(4),
                "traded_quantity": v[:n],
                "traded_amount": (pd.Series(v[:n]) * close_s).round(2),
                "status": 0
            })
            df.to_csv(path, index=False)
            print(f"  Fetched {symbol} from Sharesansar (Adjusted Chart)")
            return True
        except: return False

    def _fetch_github(self, symbol: str, path: str) -> bool:
        url = f"https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise/{symbol}.csv"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, 'w') as f: f.write(r.text)
                print(f"  Fetched {symbol} from GitHub (Unadjusted)")
                return True
        except: pass
        return False

def _to_float(v):
    if v is None: return np.nan
    try: return float(str(v).replace(",", "").strip())
    except: return np.nan

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Nepal Stock data using the unified service.")
    parser.add_argument("symbol", help="Stock symbol (e.g., FMDBL)")
    parser.add_argument("--refresh", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--stale", type=int, default=0, help="Max stale days (default 0)")
    args = parser.parse_args()
    
    service = NepalStockService(max_stale_days=args.stale)
    try:
        df = service.load_data(args.symbol, refresh_mode=args.refresh)
        print(f"✓ Success: {len(df)} rows for {args.symbol} are ready in data/nepal/{args.symbol.upper()}.csv")
    except Exception as e:
        print(f"✗ Error: {e}")
