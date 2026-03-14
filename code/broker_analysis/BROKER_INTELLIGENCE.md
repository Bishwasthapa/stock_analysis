# 🕵️ Institutional Intelligence: User Guide

The **Broker Intelligence** module is designed to track "Smart Money" movements in the Nepal Stock Market (NEPSE) by analyzing real-time floorsheet data.

---

## 🚀 Scan Modes Explained

In the web portal, you can toggle between two distinct analysis modes depending on your goal.

### 1. 🔍 Microstructure Mode (The "Stalker")
**Purpose:** Track the specific activity of the market's biggest players.

*   **Logic:**
    1.  Identifies the **Top N Brokers** (by Net Buy/Sell volume) from ShareSansar.
    2.  Downloads every single trade those specific brokers made today.
    3.  Aggregates their positions to show exactly which stocks they are accumulating or dumping.
*   **When to use:** Use this to see general "Smart Money" sentiment and popular institutional picks.

---

### 2. ⚡ Absorption Mode (The "Signal Finder")
**Purpose:** Detect extreme supply/demand imbalances in high-liquidity stocks.

*   **Logic:**
    1.  Scans the **Top 30 Turnover Stocks** (the most liquid market leaders).
    2.  For each stock, it calculates the net position of every active broker.
    3.  Looks for **Imbalance Patterns**:
        *   **Condition A (Buyer Absorption):** A single broker is buying so aggressively that they are absorbing the sales of almost the entire rest of the market. This often precedes a price breakout.
        *   **Condition B (Seller Distribution):** A single broker is selling heavily while many smaller buyers are trying to catch the fall. This often precedes a price drop.
*   **When to use:** Use this to find the most "Aggressive" institucional plays. These are high-conviction signals where one entity is making a massive stand.

---

## 📊 Understanding the Terminal Output

| Term | What it means |
| :--- | :--- |
| **Net Qty** | Total Buy Quantity minus Total Sell Quantity for that broker. |
| **Mkt Vol Rank** | How high this stock ranks in total market volume today. |
| **Absorption Strength** | (In Absorption Mode) A ratio of how much the lead broker is buying compared to the average seller. Higher = More aggressive. |
| **High-Signal Summary** | A filtered list of stocks where a "Top Broker" pick overlaps with a "Top Turnover" market leader. |

---

## 🛠️ Performance Tip
Analysis sessions take **15-20 seconds** because the engine performs real-time security bypasses and session captures to ensure data accuracy.

**Happy Hunting!** 🏹
