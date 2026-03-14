# TBOS & T-Shift Structural Algorithm

This document defines the logic for identifying **Trend Shift (T-Shift)** and **Trend Break of Structure (TBOS)** based on structural price action.

## 1. Core Principles
*   **Trend Identification**: Primary direction (Uptrend/Downtrend) is derived from the **9/18 EMA Cross**.
    *   **Uptrend**: Price moving from a structural Low to a structural High.
    *   **Downtrend**: Price moving from a structural High to a structural Low.
*   **Structural Points**: The algorithm tracks two primary pivots at any given time:
    1.  **Trend Shift (T-Shift)**
    2.  **Trend Break of Structure (TBOS)**

---

## 2. Logic: Uptrend Analysis
*In an active Uptrend (traversing from a structural Low toward a structural High):*

### Scenario A: Structural Failure (The "Shift")
**Condition**: The current Low becomes **lower** than the previous structural Low.
*   **Action**: 
    *   **T-Shift**: Moves to the **last High**.
    *   **T-BOS**: Moves to the **current Low** (the newly formed lower low).

### Scenario B: Structural Continuation (The "BOS")
**Condition**: The current Low is held (Higher than previous Low) **AND** the next High is **higher** than the previous structural High.
*   **Action**:
    *   **T-Shift**: Moves to the **current High**.
    *   **T-BOS**: Moves to (or remains at) the **previous structural Low**.

---

## 3. Logic: Downtrend Analysis
*In an active Downtrend (traversing from a structural High toward a structural Low):*

### Scenario A: Structural Failure (The "Shift")
**Condition**: The current High becomes **higher** than the previous structural High.
*   **Action**:
    *   **T-Shift**: Moves to the **last Low**.
    *   **T-BOS**: Moves to the **current High** (the newly formed higher high).

### Scenario B: Structural Continuation (The "BOS")
**Condition**: The current High is held (Lower than previous High) **AND** the next Low is **lower** than the previous structural Low.
*   **Action**:
    *   **T-Shift**: Moves to the **current Low**.
    *   **T-BOS**: Moves to (or remains at) the **previous structural High**.

---

## 4. Implementation Workflow
1.  **Traverse Points**: Iterate through price points (swings identified by EMA).
2.  **State Tracking**: Maintain a sliding window of the `Last High` and `Last Low`.
3.  **Update Triggers**: 
    *   On a **Lower Low**: Trigger Scenario A (T-Shift to High, T-BOS to current Low).
    *   On a **Higher High** (while Low is held): Trigger Scenario B (T-Shift to High, T-BOS to Low).

---

## 4. Analytical Impact
By identifying these points, the Pattern Forecast Engine can filter historical data based on whether the **Second Trend Input (B)** is:
*   Currently in a **T-Shift** state (Structural Failure/Reversal).
*   Currently in a **T-BOS** state (Structural Strength/Continuation).

This adds a "Regime Filter" that ensures probabilities are calculated only against historical matches that share the same structural health.
