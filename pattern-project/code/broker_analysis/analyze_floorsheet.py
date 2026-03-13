import argparse
import json
import requests
import urllib3
import time
import sys
import os
from datetime import datetime
from playwright.sync_api import sync_playwright

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DualLogger:
    """
    Redirects stdout to both the terminal and a file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        self.log.close()

def get_top_brokers_from_sharesansar(num_brokers=5, side='buyer', target_date=None):
    """
    Scrapes ShareSansar Top Brokers page, and returns (top_broker_ids, detected_market_date)
    """
    print(f"Scraping ShareSansar for Top {num_brokers} {side.capitalize()} Brokers{' for ' + target_date if target_date else ''}...")
    top_brokers = []
    detected_market_date = target_date # Fallback
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-http2", "--no-sandbox"]
        )
        page = browser.new_page()
        # Set a real user agent to avoid blocks
        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"})
        page.goto("https://www.sharesansar.com/top-brokers", wait_until="networkidle")
        
        try:
            # First, check the current date on the page
            page.wait_for_selector("#date", timeout=10000)
            page_date = page.input_value("#date")
            if not target_date:
                detected_market_date = page_date
                print(f"Detected Market Date from ShareSansar: {detected_market_date}")

            page.wait_for_selector("table#myTable", timeout=15000)
            
            # If a specific date is requested, set it and search
            if target_date:
                print(f"Applying date filter: {target_date}")
                
                # First, ensure we are in 'Date Wise' mode if possible. 
                try:
                    mode_selector = page.locator("select:visible, .select2-container:visible, button:has-text('Today'), button:has-text('Daily'), .dropdown-toggle").first
                    if mode_selector.count() > 0:
                        mode_selector.click()
                        page.wait_for_timeout(500)
                        date_wise = page.locator("a:has-text('Datewise'), a:has-text('Date Wise'), option:has-text('Datewise'), li:has-text('Datewise')").first
                        if date_wise.count() > 0:
                            date_wise.click()
                            page.wait_for_timeout(1000)
                except:
                    pass

                # Now fill the date
                page.click("#date")
                page.keyboard.press("Control+A")
                page.keyboard.press("Backspace")
                page.type("#date", target_date, delay=100)
                
                # Click search button
                print("Clicking Search button...")
                page.click("#btn_topbrokers_submit")
                
                # Wait for the table row 1 to potentially change or page to reload
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(3000) 

            # Click "Difference (Rs.)" header to sort
            print("Sorting by Difference...")
            header = page.locator("th:has-text('Difference (Rs.)')")
            header.click()
            time.sleep(1)
            
            if side == 'buyer':
                header.click()
                time.sleep(1)
            
            # Extract top N broker IDs and details
            rows = page.locator("table#myTable tbody tr").all()
            
            if not rows or len(rows) == 0:
                print(f"[WRN] No data rows found on ShareSansar for {target_date if target_date else 'today'}.")
                print("      This usually means the market was closed or data is not yet available.")
            
            broker_details = []
            for row in rows:
                if len(broker_details) >= num_brokers:
                    break
                    
                try:
                    cells = row.locator("td").all()
                    if len(cells) < 7: continue
                    
                    broker_no = cells[1].inner_text(timeout=3000).strip()
                    broker_name = cells[2].inner_text(timeout=3000).strip()
                    buy_amt = cells[3].inner_text(timeout=3000).strip()
                    sell_amt = cells[4].inner_text(timeout=3000).strip()
                    diff_amt = cells[6].inner_text(timeout=3000).strip()

                    if broker_no.isdigit():
                        broker_details.append({
                            "id": int(broker_no),
                            "name": broker_name,
                            "buy": buy_amt,
                            "sell": sell_amt,
                            "diff": diff_amt
                        })
                except Exception:
                    continue
                    
            browser.close()
            return broker_details, detected_market_date
            
        except Exception as e:
            if "Timeout" in str(e):
                print(f"[ERR] ShareSansar Timeout: Data not found for {target_date if target_date else 'today'}.")
                print("      Suggestion: verify if the market was open on this date.")
            else:
                print(f"Error scraping ShareSansar: {e}")
            
        browser.close()
        
    return top_brokers, detected_market_date

def get_stock_mappings(auth_token):
    """
    Fetches the list of all securities and creates a symbol-to-id mapping.
    """
    print("Fetching stock symbol-to-id mappings...")
    url = "https://www.nepalstock.com/api/nots/security?nonDelisted=true"
    headers = {
        "Authorization": auth_token,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, verify=False)
        resp.raise_for_status()
        data = resp.json()
        return {s["symbol"]: s["id"] for s in data if "symbol" in s and "id" in s}
    except Exception as e:
        print(f"Warning: Could not fetch stock mappings: {e}")
        return {}

def detect_dominance(broker_positions):
    """
    Detects market imbalances where a single broker's net position 
    significantly outweighs the average of the opposing side.
    """
    net_positions = []
    for broker_id, pos in broker_positions.items():
        net_positions.append({
            "broker_id": broker_id,
            "net_qty": pos["buy_qty"] - pos["sell_qty"]
        })
    
    # Filter out zero net positions
    net_positions = [p for p in net_positions if p["net_qty"] != 0]
    if not net_positions:
        return None

    # Sort by net quantity
    net_positions.sort(key=lambda x: x["net_qty"], reverse=True)
    
    top_buyer = net_positions[0]
    top_seller = net_positions[-1]
    
    net_buyers = [p for p in net_positions if p["net_qty"] > 0]
    net_sellers = [p for p in net_positions if p["net_qty"] < 0]
    
    avg_net_sell = abs(sum(p["net_qty"] for p in net_sellers) / len(net_sellers)) if net_sellers else 0
    avg_net_buy = sum(p["net_qty"] for p in net_buyers) / len(net_buyers) if net_buyers else 0
    
    dominance_type = None
    dominance_strength = 0
    
    # Condition A: Buyer Dominance (Absorption)
    if top_buyer["net_qty"] > 0 and len(net_buyers) < len(net_sellers) and top_buyer["net_qty"] > (avg_net_sell * 1.5):
        dominance_type = "Buyer Absorption Candidate"
        dominance_strength = top_buyer["net_qty"] / (avg_net_sell if avg_net_sell > 0 else 1)

    # Condition B: Seller Dominance (Distribution)
    elif abs(top_seller["net_qty"]) > 0 and len(net_sellers) < len(net_buyers) and abs(top_seller["net_qty"]) > (avg_net_buy * 1.5):
        dominance_type = "Seller Absorption Candidate"
        dominance_strength = abs(top_seller["net_qty"]) / (avg_net_buy if avg_net_buy > 0 else 1)

    if dominance_type:
        return {
            "type": dominance_type,
            "strength": dominance_strength,
            "top_buyer": top_buyer,
            "top_seller": top_seller,
            "net_buyers_count": len(net_buyers),
            "net_sellers_count": len(net_sellers)
        }
    return None

def find_existing_report(reports_dir, market_date, mode, side, brokers, limit, turnover, aggregated, ignore_self):
    """
    Scans the reports directory for an existing report with identical parameters.
    'brokers' can be a list of IDs or an integer (count) for discovery-mode matching.
    Returns (filepath, content) if found, else (None, None).
    """
    if not os.path.exists(reports_dir):
        return None, None
        
    mode_label = mode.upper()
    settings_line = f"Settings: Side={side.capitalize()}, Limit={limit}, Aggregated={aggregated}, IgnoreSelfTrades={ignore_self}, TurnoverMatch={turnover}"
    
    # Matching pattern: report-MODE-DATE_*.txt
    prefix = f"report-{mode_label}-"
    if market_date:
        prefix += f"{market_date}_"
        
    suffix = f"_{side}.txt"
    
    import re
    
    # Get files sorted by modified time (newest first)
    try:
        files = [f for f in os.listdir(reports_dir) if f.startswith(f"report-{mode_label}-") and f.endswith(suffix)]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)
    except:
        return None, None
    
    for filename in files:
        if market_date and not filename.startswith(f"report-{mode_label}-{market_date}_"):
            continue
            
        filepath = os.path.join(reports_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                header = f.read(2000) # Only read first 2KB for speed
                
                # Check for settings parity first
                if settings_line not in header:
                    continue
                
                # Check for broker parity
                broker_match = re.search(r"Target Brokers: \[(.*?)\]", header)
                if not broker_match:
                    continue
                
                found_ids_raw = [x.strip() for x in broker_match.group(1).split(",") if x.strip()]
                found_ids = [int(x) for x in found_ids_raw if x.isdigit()]
                
                match_found = False
                if isinstance(brokers, list):
                    if sorted(found_ids) == sorted(brokers):
                        match_found = True
                elif isinstance(brokers, int):
                    if len(found_ids) == brokers:
                        match_found = True
                
                if match_found:
                    full_content = header if len(header) < 2000 else None
                    if full_content is None:
                        f.seek(0)
                        full_content = f.read()
                        
                    try:
                        os.utime(filepath, None) # Update modified time
                    except:
                        pass
                    return filepath, full_content
        except:
            continue
    return None, None

def get_latest_market_date_from_reports(reports_dir):
    """
    Tries to infer the latest market date by looking at existing filenames.
    """
    if not os.path.exists(reports_dir):
        return None
    files = [f for f in os.listdir(reports_dir) if f.startswith("report-")]
    if not files:
        return None
    # Filenames are like report-MODE-YYYY-MM-DD_...
    dates = []
    import re
    for f in files:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', f)
        if match:
            dates.append(match.group(1))
    if not dates:
        return None
    return sorted(dates, reverse=True)[0]

def fetch_with_retries(url, headers, payload=None, params=None, method="POST", max_retries=3):
    """
    Helper to fetch data with retries for connection issues.
    """
    for i in range(max_retries):
        try:
            if method == "POST":
                resp = requests.post(url, headers=headers, json=payload, params=params, verify=False, timeout=15)
            else:
                resp = requests.get(url, headers=headers, params=params, verify=False, timeout=15)
            
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            if i == max_retries - 1:
                raise e
            wait_time = (i + 1) * 2
            print(f"      Connection error: {e}. Retrying in {wait_time}s... ({i+1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            raise e

def fetch_stock_floorsheet(auth_token, payload, symbol, stock_id):
    """
    Fetches floorsheet data for a specific stock using its numerical stockId.
    """
    url = "https://www.nepalstock.com/api/nots/nepse-data/floorsheet"
    params = {
        "size": 5000, 
        "sort": "contractId,desc",
        "stockId": stock_id
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_token,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Fetching floorsheet data for {symbol} (ID: {stock_id})...")
    data = fetch_with_retries(url, headers, payload, params)
    
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("floorsheets", {}).get("content", []) or data.get("content", []) or data.get("floorsheet", {}).get("content", [])
    
    return items

def get_auth_data_from_network(target_date=None):
    """
    Launch headless Chromium, go to NEPSE floor-sheet page.
    Intercept the network request to get the valid Authorization token
    AND the payload (which contains the dynamic session/contract ID).
    If target_date is provided (YYYY-MM-DD), it filters the floorsheet accordingly.
    """
    auth_data = {"token": None, "payload": {}}
    
    print(f"Launching browser to capture session data{' for ' + target_date if target_date else ''}...")
    with sync_playwright() as p:
        # ... [browser launch args] ...
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-http2",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-features=IsolateOrigins,site-per-process"
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        def handle_request(request):
            # Capture Authorization from ANY NEPSE API hit for maximum resilience
            if "/api/nots/" in request.url:
                token = request.header_value("Authorization")
                if token:
                    auth_data["token"] = token
                    # CRITICAL: Capture the exact floorsheet session payload (e.g., {"id": 12345})
                    if "floorsheet" in request.url and request.method == "POST":
                        try:
                            pd = request.post_data_json
                            if pd and "id" in pd:
                                auth_data["payload"] = pd
                        except:
                            pass

        page.on("request", handle_request)
        print("Navigating to NEPSE Floorsheet to capture session data...")
        try:
            # Using faster 'domcontentloaded' and hyphenated URL
            page.goto("https://www.nepalstock.com/floor-sheet", wait_until="domcontentloaded", timeout=45000)
            
            if target_date:
                try:
                    # Attempt to set date, but with a snappier timeout
                    page.wait_for_selector("input[formcontrolname='businessDate']", timeout=5000)
                    print(f"Applying NEPSE date filter: {target_date}")
                    input_field = page.locator("input[formcontrolname='businessDate']")
                    input_field.click()
                    input_field.fill(target_date)
                    input_field.press("Tab")
                    filter_btn = page.locator("button:has-text('Filter'), button:has-text('Search'), .glyphicon-search").first
                    filter_btn.click()
                except:
                    print(f"Note: NEPSE date-picker not available or already set. Capturing latest session.")
            
            # DYNAMIC POLLING: Exit as soon as we have the token AND payload
            print("Capturing background API session (Early-Exit Mode)...")
            start_time = time.time()
            max_wait = 15  # 15s max wait for the background hits
            while time.time() - start_time < max_wait:
                if auth_data["token"] and auth_data["payload"]:
                    print(f"Session data captured in {time.time() - start_time:.1f}s!")
                    break
                page.wait_for_timeout(500)
            
            if not auth_data["token"]:
                print("Warning: Max wait reached without capturing valid session.")
                
        except Exception as e:
            print(f"Warning during session capture: {e}")
        finally:
            browser.close()
    
    return auth_data

def fetch_floorsheet(auth_token, payload, broker_id=None, sort="contractId,desc"):
    url = "https://www.nepalstock.com/api/nots/nepse-data/floorsheet"
    params = {"size": 500, "sort": sort} 
    
    if broker_id:
        params["buyerBroker"] = broker_id
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_token,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Fetching floorsheet data (Broker: {broker_id})...")
    return fetch_with_retries(url, headers, payload, params)

def format_currency(value):
    if value is None or value == "-": return "-"
    try:
        return f"{float(value):,.2f}"
    except:
        return value

def analyze_data(items, top_n, aggregate=True, include_self_trades=False, turnover_info_map=None):
    """
    Process the floorsheet data and return results. UNFILTERED by turnover by default.
    """
    self_trade_count = 0
    
    # 1. First Pass: Filter only for self-trades
    filtered_items = []
    for item in items:
        buyer = item.get("buyerMemberId")
        seller = item.get("sellerMemberId")
        
        if not include_self_trades and buyer == seller:
            self_trade_count += 1
            continue
        filtered_items.append(item)

    print(f"Analysis: Found {len(items)} items. Removed {self_trade_count} self-trades.")

    results = []

    # 2. Process: Aggregate or List
    if aggregate:
        print("Mode: Aggregating quantities by Stock.")
        agg_map = {}
        for item in filtered_items:
            stock = item.get("stockSymbol")
            qty = item.get("contractQuantity", 0)
            if stock in agg_map:
                agg_map[stock] += qty
            else:
                agg_map[stock] = qty
        
        sorted_results = sorted(agg_map.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_n} Stocks by Volume (Aggregated):")
        print("-" * 115)
        print(f"{'S.N.':<5} | {'Stock':<10} | {'Total Qty':<15} | {'Mkt Vol Rank':<12} | {'LTP':<10} | {'Mkt Turnover':<20}")
        print("-" * 115)
        for i, (stock, qty) in enumerate(sorted_results[:top_n], 1):
            info = turnover_info_map.get(stock, {}) if turnover_info_map else {}
            m_rank = info.get("rank", "-")
            ltp = info.get("ltp", "-")
            turnover = format_currency(info.get("turnover", "-"))
            
            print(f"{i:<5} | {stock:<10} | {qty:<15} | {m_rank:<12} | {ltp:<10} | {turnover:<20}")
            results.append({"stock": stock, "qty": qty, "rank": m_rank, "ltp": ltp, "turnover": turnover})
            
    else:
        print("Mode: Listing top individual transactions.")
        sorted_items = sorted(filtered_items, key=lambda x: x.get("contractQuantity", 0), reverse=True)
        
        print(f"\nTop {top_n} Transactions by Volume:")
        print("-" * 140)
        print(f"{'S.N.':<5} | {'Stock':<10} | {'Qty':<10} | {'Rate':<10} | {'Buyer':<6} | {'Seller':<6} | {'Mkt Vol Rank':<12} | {'LTP':<10} | {'Mkt Turnover':<20}")
        print("-" * 140)
        for i, item in enumerate(sorted_items[:top_n], 1):
             stock = item.get('stockSymbol')
             info = turnover_info_map.get(stock, {}) if turnover_info_map else {}
             m_rank = info.get("rank", "-")
             ltp = info.get("ltp", "-")
             turnover = format_currency(info.get("turnover", "-"))
             
             print(f"{i:<5} | {stock:<10} | {item.get('contractQuantity'):<10} | {item.get('contractRate'):<10} | {item.get('buyerMemberId'):<6} | {item.get('sellerMemberId'):<6} | {m_rank:<12} | {ltp:<10} | {turnover:<20}")
             results.append({"stock": stock, "qty": item.get('contractQuantity'), "rank": m_rank, "ltp": ltp, "turnover": turnover})

    return results

def get_top_turnover_stocks(auth_token, limit=30):
    """
    Fetches top turnover stocks from NEPSE API.
    Returns:
      If limit > 0: list of objects with symbol, ltp, turnover.
      If mapping needed: dict { symbol: {rank, ltp, turnover} }.
    """
    print(f"Fetching Top {limit} Turnover Stocks from NEPSE...")
    url = "https://www.nepalstock.com/api/nots/top-ten/turnover?all=true"
    headers = {
        "Authorization": auth_token,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, verify=False)
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data, key=lambda x: x.get('turnover', 0), reverse=True)
        return sorted_data[:limit]
    except Exception as e:
        print(f"Warning: Could not fetch top turnover stocks: {e}")
        return []

def get_turnover_info_map(top_stocks):
    """Converts a top stocks list into a lookup map for analyze_data."""
    info_map = {}
    for rank, item in enumerate(top_stocks, 1):
        info_map[item.get('symbol')] = {
            "rank": rank,
            "ltp": item.get("closingPrice", "-"),
            "turnover": item.get("turnover", 0)
        }
    return info_map

def load_config():
    try:
        with open("floorsheet_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="NEPSE Floorsheet Analyzer")
    
    def_broker_count = config.get("default_broker_count", 5)
    def_stocks_limit = config.get("stocks_per_broker", 5)
    def_show_all = config.get("show_all_results", False)
    def_aggregate = config.get("aggregate_stocks", True) 
    def_ignore_self = config.get("ignore_self_trades", True) 
    def_side = config.get("market_side", "buyer")
    def_brokers = config.get("specific_brokers", [])
    def_disc_date = config.get("discovery_date")
    def_turnover_limit = config.get("top_turnover_limit", 30)

    parser.add_argument("--broker", type=str, nargs='*', help="Specific Broker IDs (e.g. 44,3). Overrides discovery.")
    parser.add_argument("--num-brokers", type=int, default=def_broker_count, help=f"Number of brokers to discover (default: {def_broker_count})")
    parser.add_argument("--discovery-date", type=str, default=def_disc_date, help="Date for broker discovery (YYYY-MM-DD). Default: latest.")
    parser.add_argument("--buyer", action="store_true", help="Discover top Buyers (Positive Difference)")
    parser.add_argument("--seller", action="store_true", help="Discover top Sellers (Negative Difference)")
    parser.add_argument("--limit", type=int, default=def_stocks_limit, help=f"Stocks per broker (default: {def_stocks_limit})")
    parser.add_argument("--all", action="store_true", default=def_show_all, help="Show all transactions/stocks")
    # Using BooleanOptionalAction to allow both --add and --no-add for proper config override
    parser.add_argument("--add", action=argparse.BooleanOptionalAction, default=def_aggregate, help="Enable stock aggregation (Add/Compare)")
    parser.add_argument("--self-trades", action=argparse.BooleanOptionalAction, default=not def_ignore_self, help="Include self-trades in results")
    parser.add_argument("--turnover", type=int, default=def_turnover_limit, help=f"Match results against Top N turnover stocks (default: {def_turnover_limit})")
    parser.add_argument("--mode", type=str, default="micro", choices=["micro", "dominance"], help="Analysis Mode: micro (per-broker) or dominance (absorption scan)")
    args = parser.parse_args()
    
    side = 'seller' if args.seller else ('buyer' if args.buyer else def_side)
    discovery_date = args.discovery_date if args.discovery_date else def_disc_date
    
    # Discovery Phase
    broker_list = []
    discovered_details = []
    market_date = discovery_date # Fallback
    broker_source = "Discovered"

    if args.broker:
        broker_source = "CLI Override"
        for item in args.broker:
            if ',' in item:
                broker_list.extend([int(x.strip()) for x in item.split(',') if x.strip().isdigit()])
            elif item.isdigit():
                broker_list.append(int(item))
    elif def_brokers:
        broker_source = "Config Specific"
        broker_list = def_brokers if isinstance(def_brokers, list) else [def_brokers]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reports_dir = os.path.join(project_root, "results", "broker_analysis")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    # --- Smart Scan Skipping (Optimistic) ---
    # We can check for a report immediately if we have enough info
    search_date = discovery_date or get_latest_market_date_from_reports(reports_dir)
    if search_date:
        # If broker_list is empty, we search by count (args.num_brokers)
        search_target = broker_list if broker_list else args.num_brokers
        existing_file, existing_content = find_existing_report(
            reports_dir, search_date, args.mode, side, search_target, 
            args.limit, args.turnover, args.add, not args.self_trades
        )
        if existing_file:
            print(f"\n[INFO] Identical report found: {os.path.basename(existing_file)}")
            print(f"       Configuration matches exactly. Displaying existing data...\n")
            print("-" * 50)
            print(existing_content)
            print("-" * 50)
            return

    # If the list is still empty, we must scrape.
    if not broker_list:
        discovered_details, market_date = get_top_brokers_from_sharesansar(args.num_brokers, side, discovery_date)
        broker_list = [b["id"] for b in discovered_details]
        
        # Check cache again after scraping ONLY market_date (if we still need to fetch details)
        # But wait, if get_top_brokers_from_sharesansar ran, it already scraped.
        # We want to check cache BEFORE that if possible.
        # However, Top N brokers change every day, so we need the date.
    else:
        # even if we have specific brokers, let's try to detect the date if not provided
        if not discovery_date:
            discovered_details, market_date = get_top_brokers_from_sharesansar(1, side, None)
        else:
            market_date = discovery_date

    if not broker_list or not market_date:
        print(f"\n[ERR] Broker/Date Discovery Failed")
        print(f"      Please verify if the target date was a trading day.")
        return

    # Setup Reporting with preserved filenames
    report_date = market_date
    timestamp = datetime.now().strftime("%H%M%S")
    mode_label = args.mode.upper()
    
    log_file_name = f"report-{mode_label}-{report_date}_{timestamp}_{side}.txt"
    log_file = os.path.join(reports_dir, log_file_name)
    
    # --- Smart Scan Skipping (Final Check) ---
    existing_file, existing_content = find_existing_report(
        reports_dir, market_date, args.mode, side, broker_list, 
        args.limit, args.turnover, args.add, not args.self_trades
    )
    
    if existing_file:
        print(f"\n[INFO] Identical report found: {os.path.basename(existing_file)}")
        print(f"       Configuration matches exactly. Displaying existing data...\n")
        print("-" * 50)
        print(existing_content)
        print("-" * 50)
        return

    with DualLogger(log_file):
        print(f"--- Market Date: {market_date} | Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Archiving report to: {log_file}")
        
        include_self = args.self_trades
        settings_str = f"Settings: Side={side.capitalize()}, Limit={args.limit}, Aggregated={args.add}, IgnoreSelfTrades={not include_self}, TurnoverMatch={args.turnover}"

        auth_data = get_auth_data_from_network(market_date)
        if auth_data["token"] is None:
            print(f"\n[ERR] NEPSE Data Capture Failed")
            print(f"      The tool couldn't retrieve valid session tokens for {market_date}.")
            print(f"      This can happen if the market was closed or the site is unresponsive.")
            return

        if discovered_details:
            print(f"\n--- [DISCOVERY SUMMARY] Top {len(discovered_details)} {side.capitalize()} Brokers ---")
            print(f"{'Rank':<5} | {'Broker':<8} | {'Name':<25} | {'Buy':<15} | {'Sell':<15} | {'Diff':<15}")
            print("-" * 100)
            for idx, b in enumerate(discovered_details, 1):
                print(f"{idx:<5} | {b['id']:<8} | {b['name']:<25} | {b['buy']:<15} | {b['sell']:<15} | {b['diff']:<15}")
            print("-" * 100)

        top_stocks_list = get_top_turnover_stocks(auth_data["token"], args.turnover)
        turnover_info_map = get_turnover_info_map(top_stocks_list)

        if args.mode == "dominance":
            print(f"Target Brokers: {sorted(broker_list)} ({broker_source})")
            print(settings_str)
            print(f"\n--- [DOMINANCE SCAN] Target: Top {args.turnover} Turnover Stocks ---")
            stock_mappings = get_stock_mappings(auth_data["token"])
            dom_results = []
            
            for stock in top_stocks_list:
                symbol = stock.get("symbol")
                stock_id = stock_mappings.get(symbol)
                if not stock_id: continue
                
                try:
                    # Delay to avoid rate-limiting/connection drops
                    time.sleep(1.2)
                    items = fetch_stock_floorsheet(auth_data["token"], auth_data["payload"], symbol, stock_id)
                    broker_positions = {}
                    for item in items:
                        buyer = item.get("buyerMemberId")
                        seller = item.get("sellerMemberId")
                        qty = item.get("contractQuantity", 0)
                        if buyer not in broker_positions: broker_positions[buyer] = {"buy_qty": 0, "sell_qty": 0}
                        if seller not in broker_positions: broker_positions[seller] = {"buy_qty": 0, "sell_qty": 0}
                        broker_positions[buyer]["buy_qty"] += qty
                        broker_positions[seller]["sell_qty"] += qty
                    
                    dominance = detect_dominance(broker_positions)
                    if dominance:
                        dom_results.append({"symbol": symbol, "turnover": stock.get("turnover", 0), "dominance": dominance})
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
            
            dom_results.sort(key=lambda x: (x["dominance"]["strength"], x["turnover"]), reverse=True)
            
            print("\n" + "="*90)
            print(f"{'STOCK':<10} | {'TURNOVER (Rs.)':<15} | {'DOMINANCE TYPE':<30} | {'STRENGTH':<10}")
            print("-" * 90)
            for res in dom_results:
                print(f"{res['symbol']:<10} | {format_currency(res['turnover']):<15} | {res['dominance']['type']:<30} | {res['dominance']['strength']:.2f}")
                print(f"  > Top Net Buyer: Broker {res['dominance']['top_buyer']['broker_id']} (+{res['dominance']['top_buyer']['net_qty']}) | Net Sellers: {res['dominance']['net_sellers_count']}")
                print(f"  > Top Net Seller: Broker {res['dominance']['top_seller']['broker_id']} ({res['dominance']['top_seller']['net_qty']}) | Net Buyers: {res['dominance']['net_buyers_count']}")
                print("-" * 90)
            
            if not dom_results:
                print("No major broker absorption/dominance detected in top turnover stocks.")

        else: # Standard Microstructure Mode
            print(f"Target Brokers: {sorted(broker_list)} ({broker_source})")
            print(settings_str)
            
            per_broker_full_results = []

            for b_id in broker_list:
                print(f"\n{'='*20} Broker {b_id} {'='*20}")
                try:
                    data = fetch_floorsheet(auth_token=auth_data["token"], payload=auth_data["payload"], broker_id=b_id, sort="quantity,desc")
                    items = []
                    if isinstance(data, list): items = data
                    elif isinstance(data, dict):
                        items = data.get("floorsheets", {}).get("content", []) or data.get("content", [])
                    
                    limit = len(items) if args.all else args.limit
                    broker_results = analyze_data(items, limit, aggregate=args.add, include_self_trades=include_self, turnover_info_map=turnover_info_map)
                    if broker_results:
                        per_broker_full_results.append({"broker": b_id, "results": broker_results})
                except Exception as e:
                    print(f"Error for Broker {b_id}: {e}")

            # Final High-Signal Summary
            if per_broker_full_results:
                print(f"\n\n{'#'*30} HIGH-SIGNAL MARKET IMPACT SUMMARY {'#'*30}")
                print(f"Stocks listed below are Top Broker favorites that are ALSO in the Market's Top {args.turnover} Turnover.")
                print("-" * 115)
                print(f"{'S.N.':<5} | {'Broker':<8} | {'Stock':<10} | {'Quantity':<15} | {'Mkt Vol Rank':<12} | {'LTP':<10} | {'Mkt Turnover':<20}")
                print("-" * 115)
                sn = 1
                for entry in per_broker_full_results:
                    b_id = entry["broker"]
                    for res in entry["results"]:
                        if res['rank'] != "-":
                            print(f"{sn:<5} | {b_id:<8} | {res['stock']:<10} | {res['qty']:<15} | {res['rank']:<12} | {res['ltp']:<10} | {res['turnover']:<20}")
                            sn += 1
                if sn == 1:
                    print("No high-signal matches found.")
                print("-" * 115)
        
        print(f"\n--- Analysis Session Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

if __name__ == "__main__":
    main()
