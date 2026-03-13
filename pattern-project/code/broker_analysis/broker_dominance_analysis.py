import argparse
import json
import requests
import urllib3
import time
import math
from playwright.sync_api import sync_playwright

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_auth_data_from_network():
    """
    Launch headless Chromium, go to NEPSE floor-sheet page.
    Intercept the network request to get the valid Authorization token
    AND the payload (which contains the dynamic session/contract ID).
    """
    auth_data = {"token": None, "payload": None}
    
    print("Launching browser to capture session data...")
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
            page.wait_for_timeout(10000)
        except Exception:
            pass

        browser.close()
        
        return auth_data

def get_top_turnover_stocks(auth_token, limit=30):
    """
    Fetches top turnover stocks from NEPSE API.
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
    resp = requests.post(url, headers=headers, json=payload, params=params, verify=False)
    resp.raise_for_status()
    data = resp.json()
    
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("floorsheets", {}).get("content", []) or data.get("content", []) or data.get("floorsheet", {}).get("content", [])
    
    return items

def detect_dominance(broker_positions):
    """
    Detect Condition A (Buyer Dominance) and Condition B (Seller Dominance).
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
    
    # Condition A: Buyer Dominance
    # One or a few brokers have large positive net buy positions
    # Many other brokers have net sell positions
    # Top buying broker's net buy volume is significantly larger than the average selling broker’s net sell volume
    
    print(f"DEBUG: Top Buyer Qty: {top_buyer['net_qty']}, Top Seller Qty: {top_seller['net_qty']}, Net Buyers: {len(net_buyers)}, Net Sellers: {len(net_sellers)}, Avg Net Sell: {avg_net_sell}, Avg Net Buy: {avg_net_buy}")
    
    if top_buyer["net_qty"] > 0 and len(net_buyers) < len(net_sellers) and top_buyer["net_qty"] > (avg_net_sell * 1.5):
        dominance_type = "Buyer Absorption Candidate"
        dominance_strength = top_buyer["net_qty"] / (avg_net_sell if avg_net_sell > 0 else 1)

    # Condition B: Seller Dominance
    # One or a few brokers have large net sell positions
    # Many other brokers have net buy positions
    # Top selling broker's net sell volume is significantly larger than the average buying broker’s net buy volume
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

def main():
    parser = argparse.ArgumentParser(description="NEPSE Broker Dominance Analysis")
    parser.add_argument("--turnover-limit", type=int, default=20, help="Number of top turnover stocks to analyze")
    args = parser.parse_args()

    auth_data = get_auth_data_from_network()
    if not auth_data["token"] or not auth_data["payload"]:
        print("Error: Could not retrieve NEPSE session data.")
        return

    top_stocks = get_top_turnover_stocks(auth_data["token"], args.turnover_limit)
    stock_mappings = get_stock_mappings(auth_data["token"])
    
    results = []
    
    for stock in top_stocks:
        symbol = stock.get("symbol")
        turnover = stock.get("turnover", 0)
        stock_id = stock_mappings.get(symbol)
        
        if not stock_id:
            print(f"Warning: Stock ID not found for {symbol}, skipping...")
            continue
            
        try:
            items = fetch_stock_floorsheet(auth_data["token"], auth_data["payload"], symbol, stock_id)
            print(f"DEBUG: Found {len(items)} items for {symbol}")
            
            broker_positions = {}
            for item in items:
                buyer = item.get("buyerMemberId")
                seller = item.get("sellerMemberId")
                qty = item.get("contractQuantity", 0)
                
                if buyer not in broker_positions:
                    broker_positions[buyer] = {"buy_qty": 0, "sell_qty": 0}
                if seller not in broker_positions:
                    broker_positions[seller] = {"buy_qty": 0, "sell_qty": 0}
                    
                broker_positions[buyer]["buy_qty"] += qty
                broker_positions[seller]["sell_qty"] += qty
            
            dominance = detect_dominance(broker_positions)
            if dominance:
                results.append({
                    "symbol": symbol,
                    "turnover": turnover,
                    "dominance": dominance
                })
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            
    # Rank results
    # Rank by dominance strength and then by turnover
    results.sort(key=lambda x: (x["dominance"]["strength"], x["turnover"]), reverse=True)
    
    print("\n" + "="*80)
    print(f"{'STOCK':<10} | {'TURNOVER':<15} | {'DOMINANCE TYPE':<30} | {'STRENGTH':<10}")
    print("-" * 80)
    
    for res in results:
        sym = res["symbol"]
        turnover_str = f"{res['turnover']:,.2f}"
        dom_type = res["dominance"]["type"]
        strength = f"{res['dominance']['strength']:.2f}"
        
        print(f"{sym:<10} | {turnover_str:<15} | {dom_type:<30} | {strength:<10}")
        
        top_buyer = res["dominance"]["top_buyer"]
        top_seller = res["dominance"]["top_seller"]
        
        print(f"  > Top Net Buyer: Broker {top_buyer['broker_id']} (+{top_buyer['net_qty']}) | Net Sellers: {res['dominance']['net_sellers_count']}")
        print(f"  > Top Net Seller: Broker {top_seller['broker_id']} ({top_seller['net_qty']}) | Net Buyers: {res['dominance']['net_buyers_count']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
