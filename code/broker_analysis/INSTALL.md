# NEPSE Floorsheet Analysis Tool - Installation Guide

Follow these steps to set up the environment and run the analysis scripts.

## 1. Create a Virtual Environment
Create a clean Python environment called `nepal_env`.
```bash
python3 -m venv nepal_env
```

## 2. Activate the Environment
Activating the environment ensures you are using the isolated Python and libraries.
```bash
source nepal_env/bin/activate
```

## 3. Install Dependencies
Install the required Python packages: `requests`, `playwright`, and `urllib3`.
```bash
pip install requests playwright urllib3
```

## 4. Install Playwright Browsers
Playwright requires its own browser binaries (Chromium).
```bash
python -m playwright install chromium
```
*(If prompted or if you are on a fresh Linux server, running `python -m playwright install-deps` might be needed to install OS-level dependencies).*

### Configuration
Defaults are centrally managed in `floorsheet_config.json`. 

**Final Config Structure:**
```json
{
    "default_broker_count": 5,
    "market_side": "buyer",
    "discovery_date": null,
    "top_turnover_limit": 30,
    "stocks_per_broker": 5,
    "aggregate_stocks": true,
    "show_all_results": false,
    "ignore_self_trades": true,
    "specific_brokers": []
}
```

**Key Settings:**
-   `discovery_date`: Must be in `YYYY-MM-DD` format (e.g. `"2026-01-20"`). Set to `null` for latest.
-   `top_turnover_limit`: Only show stocks that are also in the market's Top N turnover list.

### Usage
Run the analysis using the following command:
```bash
./nepal_env/bin/python nepalstock_floorsheet.py
```

**Override Flags:**
-   `--buyer` / `--seller`: Discover top buyers or sellers.
-   `--discovery-date 2026-01-20`: Historical broker discovery.
-   `--turnover 30`: Match against different Top Turnover limit.
-   `--broker 44,58`: Analyze specific IDs.
-   `--add`: Enable stock aggregation.
-   `--limit N`: Custom stocks per broker.
