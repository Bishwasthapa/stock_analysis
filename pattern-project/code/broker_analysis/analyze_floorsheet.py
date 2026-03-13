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
    Scrapes ShareSansar Top Brokers page, and returns the top N broker IDs
    based on the specified side (buyer = high positive difference, seller = high negative difference).
    """
    print(f"Scraping ShareSansar for Top {num_brokers} {side.capitalize()} Brokers{' for ' + target_date if target_date else ''}...")
    top_brokers = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-http2", "--no-sandbox"]
        )
        page = browser.new_page()
        page.goto("https://www.sharesansar.com/top-brokers")
        
        try:
            page.wait_for_selector("table#myTable", timeout=15000)
            
            # If a specific date is requested, set it and search
            if target_date:
                print(f"Applying date filter: {target_date}")
                page.click("#date")
                page.keyboard.press("Control+A")
                page.keyboard.press("Backspace")
                page.keyboard.type(target_date)
                page.keyboard.press("Enter")
                
                # Double check the value
                val = page.input_value("#date")
                print(f"Date input value after typing: {val}")
                
                page.click("#btn_topbrokers_submit")
                # Wait for the table row 1 to potentially change or page to reload
                page.wait_for_load_state("networkidle")
                time.sleep(3) 

            # Click "Difference (Rs.)" header to sort
            print("Sorting by Difference...")
            header = page.locator("th:has-text('Difference (Rs.)')")
            header.click()
            time.sleep(1)
            
            if side == 'buyer':
                header.click()
                time.sleep(1)
            
            # Extract data
            rows = page.locator("table#myTable tbody tr").all()
            
            if not rows or len(rows) == 0:
                print(f"[WRN] No data rows found on ShareSansar for {target_date if target_date else 'today'}.")
                print("      This usually means the market was closed or data is not yet available.")
                return []

            for row in rows:
                if len(top_brokers) >= num_brokers:
                    break
                    
                try:
                    # cell index 1 is Broker No
                    broker_cell = row.locator("td").nth(1)
                    # Check if cell exists and has content
                    if broker_cell.count() > 0:
                        broker_no = broker_cell.inner_text(timeout=3000).strip()
                        if broker_no.isdigit():
                            top_brokers.append(int(broker_no))
                except Exception:
                    # Skip rows that don't match our expected format (e.g. "No data" rows)
                    continue
                    
        except Exception as e:
            if "Timeout" in str(e):
                print(f"[ERR] ShareSansar Timeout: Data not found for {target_date if target_date else 'today'}.")
                print("      Suggestion: verify if the market was open on this date.")
            else:
                print(f"Error scraping ShareSansar: {e}")
            
        browser.close()
        
    return top_brokers

def get_auth_data_from_network(target_date=None):
    """
    Launch headless Chromium, go to NEPSE floor-sheet page.
    Intercept the network request to get the valid Authorization token
    AND the payload (which contains the dynamic session/contract ID).
    If target_date is provided (YYYY-MM-DD), it filters the floorsheet accordingly.
    """
    auth_data = {"token": None, "payload": None}
    
    print(f"Launching browser to capture session data{' for ' + target_date if target_date else ''}...")
    with sync_playwright() as p:
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
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )

        page = context.new_page()

        def handle_request(request):
            if "nepse-data/floorsheet" in request.url and request.method == "POST":
                headers = request.header_value("Authorization")
                if headers:
                    auth_data["token"] = headers
                    try:
                        auth_data["payload"] = request.post_data_json
                    except:
                        auth_data["payload"] = None

        page.on("request", handle_request)
        page.goto("https://www.nepalstock.com/floor-sheet")
        
        try:
            if target_date:
                # Wait for the date inputs to be available
                # On NEPSE, there is a date picker
                page.wait_for_selector("input[formcontrolname='businessDate']")
                print(f"Applying NEPSE date filter: {target_date}")
                
                # We need to fill the businessDate. Sometimes simple fill() doesn't trigger the reactive form.
                # We use focus/type/enter
                input_field = page.locator("input[formcontrolname='businessDate']")
                input_field.click()
                input_field.fill("") # Clear
                input_field.type(target_date)
                input_field.press("Enter")
                
                # Click search button - on NEPSE it's usually a button with "Filter"
                page.locator("button:has-text('Filter')").click()
                page.wait_for_timeout(5000) # Wait for request to fire
            else:
                page.wait_for_timeout(10000)
        except Exception as e:
            print(f"Warning during session capture: {e}")

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
    resp = requests.post(url, headers=headers, json=payload, params=params, verify=False)
    resp.raise_for_status()
    return resp.json()

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
    Fetches top turnover stocks from NEPSE API and returns a { symbol: {rank, ltp, turnover} } map.
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
        
        info_map = {}
        for rank, item in enumerate(sorted_data[:limit], 1):
            info_map[item.get('symbol')] = {
                "rank": rank,
                "ltp": item.get("closingPrice", "-"),
                "turnover": item.get("turnover", 0)
            }
        return info_map
    except Exception as e:
        print(f"Warning: Could not fetch top turnover stocks: {e}")
        return {}

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
    parser.add_argument("--add", action="store_true", default=def_aggregate, help="Enable stock aggregation (Add/Compare)")
    parser.add_argument("--self-trades", action="store_true", help="Include self-trades in results (Default: Ignored)")
    parser.add_argument("--turnover", type=int, default=def_turnover_limit, help=f"Match results against Top N turnover stocks in summary (default: {def_turnover_limit})")
    args = parser.parse_args()
    
    side = 'seller' if args.seller else ('buyer' if args.buyer else def_side)
    
    # Resolve Discovery Date (Non-interactive)
    discovery_date = args.discovery_date if args.discovery_date else def_disc_date
    
    # If we have a discovery date, use it as the primary prefix so the UI can find the correct report
    report_date_prefix = discovery_date if discovery_date else datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Point to the centralized results directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reports_dir = os.path.join(project_root, "results", "broker_analysis")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    log_file = os.path.join(reports_dir, f"report-{report_date_prefix}_{timestamp}_{side}.txt")
    
    market_date_label = discovery_date if discovery_date else "LATEST (Today)"
    
    with DualLogger(log_file):
        print(f"--- Market Date: {market_date_label} | Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Archiving report to: {log_file}")
        
        broker_list = []
        if args.broker:
            for item in args.broker:
                if ',' in item:
                    broker_list.extend([int(x.strip()) for x in item.split(',') if x.strip().isdigit()])
                elif item.isdigit():
                    broker_list.append(int(item))
        elif def_brokers:
            broker_list = def_brokers if isinstance(def_brokers, list) else [def_brokers]
        
        if not broker_list:
            # Only if no IDs provided, perform discovery
            broker_list = get_top_brokers_from_sharesansar(args.num_brokers, side, discovery_date)

        if not broker_list:
            # Only if no IDs provided, perform discovery
            broker_list = get_top_brokers_from_sharesansar(args.num_brokers, side, discovery_date)

        if not broker_list:
            print(f"\n[ERR] Broker Discovery Failed for {discovery_date}")
            print(f"      No top brokers were found on ShareSansar for this date.")
            print(f"      Please verify if {discovery_date} was a trading day.")
            return

        include_self = args.self_trades or (not def_ignore_self)

        auth_data = get_auth_data_from_network(discovery_date)
        if not auth_data["token"] or not auth_data["payload"]:
            print(f"\n[ERR] NEPSE Data Capture Failed")
            print(f"      The tool couldn't retrieve valid session tokens for {discovery_date}.")
            print(f"      This can happen if the market was closed or the site is unresponsive.")
            return

        turnover_info_map = get_top_turnover_stocks(auth_data["token"], args.turnover)

        print(f"Target Brokers: {broker_list}")
        print(f"Settings: Side={side.capitalize()}, Limit={args.limit}, Aggregated={args.add}, IgnoreSelfTrades={not include_self}, TurnoverMatch={args.turnover}")
        
        per_broker_full_results = []

        for b_id in broker_list:
            print(f"\n{'='*20} Broker {b_id} {'='*20}")
            try:
                data = fetch_floorsheet(auth_token=auth_data["token"], payload=auth_data["payload"], broker_id=b_id, sort="quantity,desc")
                
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("floorsheets", {}).get("content", []) or data.get("content", [])
                
                # Analyze ALL items for the per-broker report
                limit = len(items) if args.all else args.limit
                broker_results = analyze_data(items, limit, aggregate=args.add, include_self_trades=include_self, turnover_info_map=turnover_info_map)
                
                if broker_results:
                    per_broker_full_results.append({"broker": b_id, "results": broker_results})
                
            except Exception as e:
                print(f"Error for Broker {b_id}: {e}")

        # Final High-Signal Summary (Strictly filtered)
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
                    # ONLY show if it has a turnover rank (meaning it was in the Top N list)
                    if res['rank'] != "-":
                        print(f"{sn:<5} | {b_id:<8} | {res['stock']:<10} | {res['qty']:<15} | {res['rank']:<12} | {res['ltp']:<10} | {res['turnover']:<20}")
                        sn += 1
            
            if sn == 1:
                print("No high-signal matches found for the analyzed brokers.")
            print("-" * 115)
        
        print(f"\n--- Analysis Session Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

if __name__ == "__main__":
    main()
