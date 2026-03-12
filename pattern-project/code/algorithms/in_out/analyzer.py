"""
Analyze repeating 2-token -> 1-token transition patterns from point-label output.

Input:
  results/nepal/<SYMBOL>/in_out/csv/in_out_pattern_9_18.csv
  (expects a `point_label` column from pattern_detector_v2.py)

Outputs:
  csv/:
    - stats_token_performance.csv
    - stats_raw_transition_matrix.csv
    - stats_context_stability.csv
    - strategy_pattern_reliability.csv
    - strategy_top_setups.csv
    - strategy_recommendations.csv
    - movement_clean_transitions.csv
    - movement_detailed_paths.csv
    - forecast_next_signal.csv
    - forecast_confirmed_completions.csv
    - forecast_completion_examples.csv
    - movement_history_log.csv
    - movement_pattern_transitions.csv
    - movement_transition_examples.csv
  txt/ (kept):
    - transition_pattern_chain_9_18.txt
    - transition_pattern_path_9_18.txt
  txt/ (kept):
    - transition_pattern_chain_9_18.txt
    - transition_pattern_path_9_18.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Context = Tuple[str, str]


def read_rows(input_csv: Path, date_cutoff: Optional[str] = None) -> List[dict]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    if "point_label" not in rows[0]:
        raise ValueError(
            f"`point_label` column not found in {input_csv}. "
            "Run pattern_detector_v2.py first."
        )
    if date_cutoff and "date" in rows[0]:
        from datetime import datetime as _dt
        cutoff = _dt.strptime(date_cutoff, "%Y-%m-%d")
        rows = [r for r in rows if r.get("date", "") and _dt.strptime(r["date"][:10], "%Y-%m-%d") >= cutoff]
        if not rows:
            raise ValueError(f"No rows remaining after filtering to {date_cutoff}+")
    return rows


def build_context_counter(tokens: Sequence[str]) -> Dict[Context, Counter]:
    counter: Dict[Context, Counter] = defaultdict(Counter)
    for i in range(2, len(tokens)):
        prev2 = (tokens[i - 2], tokens[i - 1])
        nxt = tokens[i]
        counter[prev2][nxt] += 1
    return counter


def entropy_of(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_role(value: str) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return -1


def _wilson_score(ups: int, total: int, z: float = 1.28) -> float:
    """
    Wilson score interval lower bound.
    z=1.28 for 80% confidence, z=1.96 for 95%.
    """
    if total == 0:
        return 0.0
    p = ups / total
    return (p + z*z/(2*total) - z * math.sqrt((p*(1-p) + z*z/(4*total))/total)) / (1 + z*z/total)


def _date_label(value: str) -> str:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return str(value)


def find_all_valid_patterns(rows: List[dict]) -> List[dict]:
    all_patterns = []
    for i in range(len(rows) - 3):
        r0, r1, r2, r3 = rows[i], rows[i+1], rows[i+2], rows[i+3]
        if not (r0["type"] and r1["type"] and r2["type"] and r3["type"]):
            continue
            
        p0_price = float(r0["price"])
        p1_price = float(r1["price"])
        p2_price = float(r2["price"])
        p3_price = float(r3["price"])
        
        is_up = (r0["type"].upper() == 'LOW' and r1["type"].upper() == 'HIGH' and 
                 r2["type"].upper() == 'LOW' and r3["type"].upper() == 'HIGH' and 
                 p2_price > p0_price and p3_price > p1_price)
                 
        is_down = (r0["type"].upper() == 'HIGH' and r1["type"].upper() == 'LOW' and 
                   r2["type"].upper() == 'HIGH' and r3["type"].upper() == 'LOW' and 
                   p2_price < p0_price and p3_price < p1_price)
                   
        if is_up or is_down:
            token = "IN_UP" if is_up else "IN_DOWN"
            all_patterns.append({
                "idx_0": int(r0["index"]),
                "idx_1": int(r1["index"]),
                "idx_2": int(r2["index"]),
                "idx_3": int(r3["index"]),
                "date_0": r0["date"],
                "date_1": r1["date"],
                "date_2": r2["date"],
                "date_3": r3["date"],
                "date_0_label": _date_label(r0["date"]),
                "date_1_label": _date_label(r1["date"]),
                "date_2_label": _date_label(r2["date"]),
                "date_3_label": _date_label(r3["date"]),
                "pattern_token": token,
                "trend_type": "UPTREND" if is_up else "DOWNTREND"
            })
    return all_patterns


def analyze(
    rows: List[dict],
    out_dir: Path,
    csv_dir: Path,
    txt_dir: Path,
    split_ratio: float,
    min_context_count: int,
    stable_threshold: float,
    strong_swing_min_move_pct: float,
) -> Dict[str, int]:
    tokens = [r["point_label"] for r in rows]
    dates = [r.get("date", "") for r in rows]
    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    # 1) Token summary
    token_rows = []
    for label, count in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
        token_rows.append(
            {
                "label": label,
                "count": count,
                "percent": f"{(count / total_tokens * 100):.2f}",
            }
        )

    # 2) Full 2->1 table and context summary
    full_counter = build_context_counter(tokens)
    full_rows = []
    context_rows = []
    for ctx, next_counter in sorted(full_counter.items()):
        total_ctx = sum(next_counter.values())
        top_next, top_count = max(next_counter.items(), key=lambda x: x[1])
        top_prob = top_count / total_ctx if total_ctx else 0.0
        ent = entropy_of(next_counter)
        stability = (
            "STABLE"
            if total_ctx >= min_context_count and top_prob >= stable_threshold
            else "UNSTABLE"
        )
        score = total_ctx * top_prob

        context_rows.append(
            {
                "prev_2": f"{ctx[0]}|{ctx[1]}",
                "total_context_count": total_ctx,
                "top_next": top_next,
                "top_next_count": top_count,
                "top_next_prob": f"{top_prob:.4f}",
                "entropy": f"{ent:.4f}",
                "stability": stability,
                "score_count_x_prob": f"{score:.4f}",
            }
        )

        for nxt, cnt in sorted(next_counter.items(), key=lambda x: (-x[1], x[0])):
            full_rows.append(
                {
                    "prev_2": f"{ctx[0]}|{ctx[1]}",
                    "next": nxt,
                    "count": cnt,
                    "total_context_count": total_ctx,
                    "prob_next_given_prev2": f"{(cnt / total_ctx):.4f}",
                }
            )

    write_csv(
        csv_dir / "stats_raw_transition_matrix.csv",
        ["prev_2", "next", "count", "total_context_count", "prob_next_given_prev2"],
        full_rows,
    )
    context_rows_sorted = sorted(
        context_rows,
        key=lambda r: (-float(r["score_count_x_prob"]), -int(r["total_context_count"])),
    )
    write_csv(
        csv_dir / "stats_context_summary.csv",
        [
            "prev_2",
            "total_context_count",
            "top_next",
            "top_next_count",
            "top_next_prob",
            "entropy",
            "stability",
            "score_count_x_prob",
        ],
        context_rows_sorted,
    )

    # 3) Train vs recent validation
    split_idx = max(3, int(total_tokens * split_ratio))
    train_tokens = tokens[:split_idx]
    recent_tokens = tokens[split_idx:]
    train_counter = build_context_counter(train_tokens)
    recent_counter = build_context_counter(recent_tokens)

    all_contexts = sorted(set(train_counter.keys()) | set(recent_counter.keys()))
    validation_rows = []
    for ctx in all_contexts:
        tc = train_counter.get(ctx, Counter())
        rc = recent_counter.get(ctx, Counter())

        train_total = sum(tc.values())
        recent_total = sum(rc.values())

        train_top_next, train_top_count = ("", 0)
        train_top_prob = 0.0
        if train_total > 0:
            train_top_next, train_top_count = max(tc.items(), key=lambda x: x[1])
            train_top_prob = train_top_count / train_total

        recent_top_next, recent_top_count = ("", 0)
        recent_top_prob = 0.0
        if recent_total > 0:
            recent_top_next, recent_top_count = max(rc.items(), key=lambda x: x[1])
            recent_top_prob = recent_top_count / recent_total

        hit_count = rc.get(train_top_next, 0) if train_top_next else 0
        hit_rate = (hit_count / recent_total) if recent_total else 0.0
        drift = (
            "UNCHANGED"
            if train_top_next and recent_top_next and train_top_next == recent_top_next
            else "CHANGED"
        )
        if not train_top_next or not recent_top_next:
            drift = "N/A"

        validation_rows.append(
            {
                "prev_2": f"{ctx[0]}|{ctx[1]}",
                "train_count": train_total,
                "train_top_next": train_top_next,
                "train_top_prob": f"{train_top_prob:.4f}",
                "recent_count": recent_total,
                "recent_top_next": recent_top_next,
                "recent_top_prob": f"{recent_top_prob:.4f}",
                "recent_hit_rate_of_train_top": f"{hit_rate:.4f}",
                "top_next_drift": drift,
            }
        )

    validation_rows_sorted = sorted(
        validation_rows, key=lambda r: (-int(r["train_count"]), -int(r["recent_count"]))
    )
    write_csv(
        csv_dir / "strategy_pattern_reliability.csv",
        [
            "prev_2",
            "train_count",
            "train_top_next",
            "train_top_prob",
            "recent_count",
            "recent_top_next",
            "recent_top_prob",
            "recent_hit_rate_of_train_top",
            "top_next_drift",
        ],
        validation_rows_sorted,
    )

    # 4) Top actionable contexts
    actionable_rows = [
        r for r in context_rows_sorted if int(r["total_context_count"]) >= min_context_count
    ]
    write_csv(
        csv_dir / "strategy_top_setups.csv",
        [
            "prev_2",
            "total_context_count",
            "top_next",
            "top_next_count",
            "top_next_prob",
            "entropy",
            "stability",
            "score_count_x_prob",
        ],
        actionable_rows,
    )

    # 5) Easy-read dominant pattern rules
    easy_rows = []
    for r in context_rows_sorted:
        total_count = int(r["total_context_count"])
        top_prob = float(r["top_next_prob"])
        if total_count >= min_context_count and top_prob >= 0.75:
            strength = "STRONG"
        elif total_count >= min_context_count and top_prob >= 0.60:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        easy_rows.append(
            {
                "rule": f"IF {r['prev_2']} THEN {r['top_next']}",
                "prev_2": r["prev_2"],
                "predicted_next": r["top_next"],
                "confidence": r["top_next_prob"],
                "count": r["total_context_count"],
                "strength": strength,
            }
        )

    write_csv(
        csv_dir / "strategy_recommendations.csv",
        ["rule", "prev_2", "predicted_next", "confidence", "count", "strength"],
        easy_rows,
    )

    # text output trimmed (no transition_easy_patterns.txt)

    # 6) Clean combination tables:
    #    Inputs must be valid pattern labels only.
    #    If immediate next label is INVALID, skip forward to the next valid label.
    #    We output both:
    #      A) prev2 -> next valid pattern
    #      B) prev2 -> swing result of that resolved next event (SWING_HIGH/SWING_LOW)
    keep_tokens = {"IN_UP", "IN_DOWN", "OUT_UP", "OUT_DOWN"}
    labels = [r.get("point_label", "") for r in rows]
    swing_types = [str(r.get("type", "")).upper() for r in rows]
    trend_types = [str(r.get("trend_type", "")).upper() for r in rows]
    roles = [_parse_role(str(r.get("pattern_role", ""))) for r in rows]
    row_dates = [str(r.get("date", "")) for r in rows]

    clean_next_counter: Dict[Context, Counter] = defaultdict(Counter)
    clean_swing_counter: Dict[Context, Counter] = defaultdict(Counter)

    for i in range(2, len(labels)):
        a = labels[i - 2]
        b = labels[i - 1]
        if a not in keep_tokens or b not in keep_tokens:
            continue

        # Find the path of intermediate swings until the next valid pattern
        j = i
        intermediate_path = []
        while j < len(labels) and labels[j] not in keep_tokens:
            st = swing_types[j]
            if st == "HIGH":
                intermediate_path.append("INVALID_UP")
            elif st == "LOW":
                intermediate_path.append("INVALID_DOWN")
            j += 1
            
        if j >= len(labels):
            continue
            
        nxt = labels[j]
        clean_next_counter[(a, b)][nxt] += 1
        
        # Build the full path string: e.g. "INVALID_UP -> INVALID_DOWN -> IN_UP"
        # Or just "IN_UP" if there were no intermediate invalid swings
        if intermediate_path:
            full_path = " -> ".join(intermediate_path) + " -> " + nxt
        else:
            full_path = nxt
            
        clean_swing_counter[(a, b)][full_path] += 1

    clean_full_rows = []
    grouped = defaultdict(list)
    for (a, b), cnts in sorted(clean_next_counter.items()):
        total_ctx = sum(cnts.values())
        for nxt, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            clean_row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "next": nxt,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_next_given_prev2": f"{(cnt / total_ctx):.4f}",
            }
            clean_full_rows.append(clean_row)
            grouped[(a, b)].append(clean_row)

    clean_full_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["next"])
    )
    write_csv(
        csv_dir / "movement_clean_transitions.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next",
            "count",
            "total_context_count",
            "prob_next_given_prev2",
        ],
        clean_full_rows,
    )

    # text output trimmed (no transition_clean_prev2_to_next.txt)

    clean_swing_rows = []
    grouped_swing = defaultdict(list)
    for (a, b), cnts in sorted(clean_swing_counter.items()):
        total_ctx = sum(cnts.values())
        for swing_label, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "path_result": swing_label,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_path_given_prev2": f"{(cnt / total_ctx):.4f}",
            }
            clean_swing_rows.append(row)
            grouped_swing[(a, b)].append(row)

    clean_swing_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["path_result"])
    )

    # The `transition_pattern_path_9_18.txt` text format was removed to focus on strategy and chain files

    # 6B) Confirmed 4-point completion table:
    # Count prev2 -> next only when the "next" event can be traced to role 3
    # in the same trend context (0/1/2/3 completion requirement).
    def completion_index_for(start_idx: int) -> int:
        role0 = roles[start_idx]
        if role0 < 0:
            return -1
        if role0 == 3:
            return start_idx
        trend0 = trend_types[start_idx]
        needed = role0 + 1
        for k in range(start_idx + 1, len(rows)):
            if trend_types[k] != trend0:
                continue
            rk = roles[k]
            if rk == needed:
                if rk == 3:
                    return k
                needed += 1
                continue
            if rk > needed:
                return -1
        return -1

    confirmed_counter: Dict[Context, Counter] = defaultdict(Counter)
    confirmed_examples: List[dict] = []
    for i in range(2, len(labels)):
        a = labels[i - 2]
        b = labels[i - 1]
        if a not in keep_tokens or b not in keep_tokens:
            continue

        # same "resolved next valid pattern" logic used by clean table
        j = i
        while j < len(labels) and labels[j] not in keep_tokens:
            j += 1
        if j >= len(labels):
            continue
        nxt = labels[j]

        comp_j = completion_index_for(j)
        if comp_j < 0:
            continue

        confirmed_counter[(a, b)][nxt] += 1
        confirmed_examples.append(
            {
                "prev_2_a": a,
                "prev_2_b": b,
                "next": nxt,
                "date_prev_2_a": row_dates[i - 2],
                "date_prev_2_b": row_dates[i - 1],
                "date_next": row_dates[j],
                "date_completion": row_dates[comp_j],
                "date_prev_2_a_label": _date_label(row_dates[i - 2]),
                "date_prev_2_b_label": _date_label(row_dates[i - 1]),
                "date_next_label": _date_label(row_dates[j]),
                "date_completion_label": _date_label(row_dates[comp_j]),
                "next_role": roles[j],
                "completion_role": roles[comp_j],
            }
        )

    confirmed_rows = []
    confirmed_grouped = defaultdict(list)
    for (a, b), cnts in sorted(confirmed_counter.items()):
        total_ctx = sum(cnts.values())
        for nxt, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "next": nxt,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_next_given_prev2_confirmed": f"{(cnt / total_ctx):.4f}",
            }
            confirmed_rows.append(row)
            confirmed_grouped[(a, b)].append(row)

    confirmed_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["next"])
    )
    write_csv(
        csv_dir / "forecast_confirmed_completions.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next",
            "count",
            "total_context_count",
            "prob_next_given_prev2_confirmed",
        ],
        confirmed_rows,
    )

    # text output trimmed (no transition_clean_prev2_confirmed.txt)

    # 7) Priority output:
    #    prev2 inputs are valid patterns only.
    #    Priority:
    #      - if immediate next is valid pattern -> use that
    #      - else if immediate next is strong swing -> use SWING_HIGH/LOW
    #      - else -> no actionable signal
    prices = []
    for r in rows:
        try:
            prices.append(float(r.get("price", "")))
        except Exception:
            prices.append(float("nan"))

    priority_counter: Dict[Context, Counter] = defaultdict(Counter)
    for i in range(2, len(labels)):
        a = labels[i - 2]
        b = labels[i - 1]
        if a not in keep_tokens or b not in keep_tokens:
            continue

        nxt_label = labels[i]
        if nxt_label in keep_tokens:
            priority_counter[(a, b)][nxt_label] += 1
            continue

        # Fallback: immediate strong swing only.
        if i - 1 < 0:
            continue
        p_prev = prices[i - 1]
        p_now = prices[i]
        if math.isnan(p_prev) or math.isnan(p_now) or p_prev == 0:
            continue
        move_pct = abs((p_now - p_prev) / p_prev) * 100.0
        if move_pct < strong_swing_min_move_pct:
            continue

        st = swing_types[i]
        if st == "HIGH":
            priority_counter[(a, b)]["SWING_HIGH"] += 1
        elif st == "LOW":
            priority_counter[(a, b)]["SWING_LOW"] += 1

    priority_rows = []
    priority_grouped = defaultdict(list)
    for (a, b), cnts in sorted(priority_counter.items()):
        total_ctx = sum(cnts.values())
        for target, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            signal_type = "PATTERN" if target in keep_tokens else "SWING"
            row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "next_signal": target,
                "signal_type": signal_type,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_signal_given_prev2": f"{(cnt / total_ctx):.4f}",
            }
            priority_rows.append(row)
            priority_grouped[(a, b)].append(row)

    priority_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["next_signal"])
    )
    write_csv(
        csv_dir / "forecast_next_signal.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next_signal",
            "signal_type",
            "count",
            "total_context_count",
            "prob_signal_given_prev2",
        ],
        priority_rows,
    )

    # text output trimmed (no transition_clean_prev2_priority.txt)

    # 8) Pattern-level transitions (immediate complete patterns only: 0,1,2,3)
    #    Build valid patterns by scanning every immediate 4-point window.
    #    Then transition uses adjacent valid patterns in scan order:
    #    Pattern[i-2] + Pattern[i-1] -> Pattern[i]
    completed_patterns: List[dict] = []
    n = len(rows)
    prices_num = []
    types_norm = []
    for r in rows:
        try:
            prices_num.append(float(r.get("price", "")))
        except Exception:
            prices_num.append(float("nan"))
        types_norm.append(str(r.get("type", "")).upper())

    def _is_valid_up(i0: int) -> bool:
        i1, i2, i3 = i0 + 1, i0 + 2, i0 + 3
        if any(math.isnan(prices_num[k]) for k in (i0, i1, i2, i3)):
            return False
        return (
            types_norm[i0] == "LOW"
            and types_norm[i1] == "HIGH"
            and types_norm[i2] == "LOW"
            and types_norm[i3] == "HIGH"
            and prices_num[i2] > prices_num[i0]
            and prices_num[i3] > prices_num[i1]
        )

    def _is_valid_down(i0: int) -> bool:
        i1, i2, i3 = i0 + 1, i0 + 2, i0 + 3
        if any(math.isnan(prices_num[k]) for k in (i0, i1, i2, i3)):
            return False
        return (
            types_norm[i0] == "HIGH"
            and types_norm[i1] == "LOW"
            and types_norm[i2] == "HIGH"
            and types_norm[i3] == "LOW"
            and prices_num[i2] < prices_num[i0]
            and prices_num[i3] < prices_num[i1]
        )

    for i in range(n - 3):
        token = ""
        trend = ""
        if _is_valid_up(i):
            token = labels[i]
            trend = "UPTREND"
        elif _is_valid_down(i):
            token = labels[i]
            trend = "DOWNTREND"
        else:
            continue

        if token not in keep_tokens:
            continue

        completed_patterns.append(
            {
                "pattern_token": token,
                "trend_type": trend,
                "idx_0": i,
                "idx_1": i + 1,
                "idx_2": i + 2,
                "idx_3": i + 3,
                "date_0": row_dates[i],
                "date_1": row_dates[i + 1],
                "date_2": row_dates[i + 2],
                "date_3": row_dates[i + 3],
                "date_0_label": _date_label(row_dates[i]),
                "date_1_label": _date_label(row_dates[i + 1]),
                "date_2_label": _date_label(row_dates[i + 2]),
                "date_3_label": _date_label(row_dates[i + 3]),
            }
        )

    completed_patterns.sort(key=lambda r: r["idx_0"])

    # --- DOUBLE COMBINATION STRATEGY ENGINE: (A + B -> Path -> Target) ---
    all_patterns = find_all_valid_patterns(rows)
    # input1, input2: valid patterns, non-intersecting
    # path: list of all directional swings between input2's end and target's start
    # target: next valid pattern or END_OF_DATA
    # iteration: input1 = input2 (old), input2 = target (new)
    
    iterative_transitions = []
    if len(completed_patterns) >= 2:
        # Initial context
        p_a_idx = 0
        while p_a_idx < len(completed_patterns) - 1:
            p_a = completed_patterns[p_a_idx]
            
            # Find the first p_b after p_a that doesn't intersect
            p_b = None
            p_b_idx = -1
            for j in range(p_a_idx + 1, len(completed_patterns)):
                cand_b = completed_patterns[j]
                if cand_b["idx_0"] >= p_a["idx_3"]:
                    p_b = cand_b
                    p_b_idx = j
                    break
            
            if not p_b:
                break
                
            # IDENTIFY THE IMMEDIATE TARGET
            # Rule: If p_c is 'IN' (chained/intersected), Target = Pattern C.
            # Else, Target = The very first swing (OUT_UP or OUT_DOWN).
            
            p_c = None
            for p_cand in all_patterns:
                if p_cand["idx_0"] > p_b["idx_0"]:
                    p_c = p_cand
                    break

            is_linked = p_c and (p_c["idx_0"] <= p_b["idx_3"])
            
            if is_linked:
                # IMMEDIATE PATTERN TARGET
                target_start_idx = p_c["idx_0"]
                scan_end = p_c["idx_3"]
                dir_str = "UP" if p_c["trend_type"] == "UPTREND" else "DOWN"
                target_label = f"IN_{dir_str}"
                path_swings = [] # Chained/Intersected patterns have no gap path
            else:
                # IMMEDIATE SWING TARGET (There is a gap)
                # We stop at the very first swing after B ends
                target_start_idx = p_b["idx_3"] + 1
                scan_end = target_start_idx
                
                # Use swing_types to label this single outside move
                st = swing_types[target_start_idx] if target_start_idx < len(rows) else "END"
                # Renamed for better readability as per user request
                dir_label = "UP" if st.upper() == "HIGH" else "DOWN"
                target_label = f"SWING_{dir_label}"
                path_swings = [f"SWING_{st.upper()}"]
                
                # Capture the date of this specific swing for the report
                t_date_val = rows[target_start_idx]["date_label"] if target_start_idx < len(rows) else "N/A"
                p_c = {"date_0_label": t_date_val} # Fake p_c object just for the date
            
            # Entry and Direction
            entry_price = prices_num[p_b["idx_3"]]
            is_long = types_norm[p_b["idx_3"]] == "LOW"
            
            # Peak Excursion tracking (from B end to Target end)
            peak_profit = 0.0
            max_drawdown = 0.0
            
            for k in range(p_b["idx_3"] + 1, min(scan_end + 1, len(rows))):
                if math.isnan(prices_num[k]): continue
                
                if is_long:
                    ret = (prices_num[k] - entry_price) / entry_price
                else:
                    ret = (entry_price - prices_num[k]) / entry_price
                
                peak_profit = max(peak_profit, ret)
                max_drawdown = min(max_drawdown, ret)

            path_label = " -> ".join(path_swings) if path_swings else ""
            efficiency = len(path_swings)
            
            iterative_transitions.append({
                "a": p_a,
                "b": p_b,
                "path": path_label,
                "target": target_label,
                "peak_profit": peak_profit * 100.0,
                "drawdown": max_drawdown * 100.0,
                "efficiency": efficiency,
                "p_c_obj": p_c
            })
            
            # SLIDING WINDOW: next input1 = current input2
            p_a_idx = p_b_idx

    iter_counter: Dict[Context, Counter] = defaultdict(Counter)
    for trans in iterative_transitions:
        ctx = (trans["a"]["pattern_token"], trans["b"]["pattern_token"])
        if trans['path']:
            combined_result = f"[{trans['path']}] -> {trans['target']}"
        else:
            combined_result = f"{trans['target']}"
        iter_counter[ctx][combined_result] += 1

    iter_txt = txt_dir / "Final_strategy_9_18.txt"
    with iter_txt.open("w", encoding="utf-8") as f:
        f.write("Double Combination Strategy: (A + B -> Path -> Target)\n")
        f.write("Rules:\n")
        f.write("  - Inputs (A, B): Valid patterns, non-intersecting.\n")
        f.write("  - Path: Every swing between B's end and Target's start (DIRECT = no gap).\n")
        f.write("  - Target: The next structural anchor pattern.\n")
        f.write("  - Iteration: Next A = current B.\n\n")
        
        # ── Chronological sequences first ──
        f.write("Chronological Strategy Sequences:\n")
        for i, trans in enumerate(iterative_transitions):
            a, b, path, target = trans["a"], trans["b"], trans["path"], trans["target"]
            pk, eff = trans["peak_profit"], trans["efficiency"]
            t_date = trans["p_c_obj"]["date_0_label"] if trans["p_c_obj"] else "N/A"
            if path:
                f.write(f"{i+1}. {a['pattern_token']} ({a['date_0_label']}) + {b['pattern_token']} ({b['date_0_label']}) -> [Peak +{pk:.1f}% | Eff: {eff}] -> {target} ({t_date})\n")
            else:
                f.write(f"{i+1}. {a['pattern_token']} ({a['date_0_label']}) + {b['pattern_token']} ({b['date_0_label']}) -> [Peak +{pk:.1f}%] -> {target} ({t_date})\n")

        f.write("\n")

        # ── Probability tables: Structural AND Profit ──
        # Group by setup
        setup_groups = defaultdict(list)
        for trans in iterative_transitions:
            ctx = (trans["a"]["pattern_token"], trans["b"]["pattern_token"])
            if "END_OF_DATA" not in trans["target"]:
                setup_groups[ctx].append(trans)

        for ctx, transitions in sorted(setup_groups.items()):
            a_token, b_token = ctx
            total = len(transitions)
            if total == 0: continue

            f.write(f"{a_token} + {b_token}:\n")
            
            # 1. Structural Targets (Count unique sequences)
            seq_counts = Counter()
            for t in transitions:
                if t['path']:
                    seq_counts[f"[{t['path']}] -> {t['target']}"] += 1
                else:
                    seq_counts[t['target']] += 1
            
            f.write("  (Structural Targets / Next Patterns)\n")
            for seq, count in sorted(seq_counts.items(), key=lambda x: (-x[1], x[0])):
                pct = (count / total) * 100
                f.write(f"    -> {seq:40} | count={count}/{total} ({pct:.2f}%)\n")
            
            # 2. Hybrid Profit Buckets (Cumulative)
            f.write("  (Profit Predictability Machine)\n")
            for threshold in [5, 10, 15, 20]:
                hit_count = sum(1 for t in transitions if t["peak_profit"] >= threshold)
                prob = (hit_count / total) * 100
                f.write(f"    -> At Least +{threshold}% Profit reached     | count={hit_count}/{total} ({prob:.1f}%)\n")
            
            # 3. Efficiency Stats
            avg_eff = sum(t["efficiency"] for t in transitions) / total
            f.write(f"    -> Average Messiness (Swing Count)    | {avg_eff:.1f} swings\n")
            f.write("\n")

    # --- 10) EXPORT LIVE FORECAST ---
    # Current State = Last 2 non-intersecting completed patterns
    print(f"DEBUG: Completed patterns count = {len(completed_patterns)}")
    if len(completed_patterns) >= 2:
        p_b = completed_patterns[-1]
        # Find the latest p_a that doesn't intersect p_b
        p_a = None
        for i in range(len(completed_patterns) - 2, -1, -1):
            cand_a = completed_patterns[i]
            if cand_a["idx_3"] <= p_b["idx_0"]:
                p_a = cand_a
                break
        
        print(f"DEBUG: p_a found = {p_a is not None}")
        if p_a:
            ctx = (p_a["pattern_token"], p_b["pattern_token"])
            print(f"DEBUG: Current Context = {ctx}")
            matches = setup_groups.get(ctx, [])
            
            forecast = {
                "symbol": out_dir.parent.name, # Guess from path
                "current_setup": f"{ctx[0]} + {ctx[1]}",
                "last_pattern_date": p_b["date_3_label"],
                "total_historical_matches": len(matches),
                "outcomes": []
            }
            
            if matches:
                # Group by structural target for cleaner UI display
                target_map = {}
                for m in matches:
                    key = (m["path"], m["target"])
                    if key not in target_map:
                        target_map[key] = []
                    target_map[key].append(m)
                
                for key, trans_list in target_map.items():
                    path, target = key
                    count = len(trans_list)
                    raw_prob = (count / len(matches)) * 100
                    
                    # Profit machine for THIS specific path->target combo
                    pm = {}
                    for thresh in [5, 10, 15, 20]:
                        hits = sum(1 for t in trans_list if t["peak_profit"] >= thresh)
                        pm[f"at_least_{thresh}"] = round((hits / count) * 100, 1)
                    
                    forecast["outcomes"].append({
                        "path": path,
                        "target": target,
                        "count": count,
                        "probability": round(raw_prob, 1),
                        "wilson_score": round(_wilson_score(count, len(matches)) * 100, 1),
                        "profit_machine": pm,
                        "is_bullish": "UP" in target
                    })
                
                # Sort outcomes by Wilson Score
                forecast["outcomes"].sort(key=lambda x: x["wilson_score"], reverse=True)
            
            import json
            with open(csv_dir / "current_forecast.json", "w") as f_json:
                json.dump(forecast, f_json, indent=2)



    stable_count = sum(1 for r in context_rows if r["stability"] == "STABLE")
    unstable_count = sum(1 for r in context_rows if r["stability"] == "UNSTABLE")

    return {
        "total_tokens": total_tokens,
        "total_contexts": len(context_rows),
        "stable_contexts": stable_count,
        "unstable_contexts": unstable_count,
        "train_end_index": split_idx - 1,
        "recent_start_index": split_idx,
        "train_end_date": dates[split_idx - 1] if split_idx - 1 < len(dates) else "",
        "recent_start_date": dates[split_idx] if split_idx < len(dates) else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeating 2-token pattern transitions from in/out labels."
    )
    parser.add_argument("symbol", help="Stock symbol, e.g. NICA")
    parser.add_argument("--market", default="nepal", help="Market name for path organization")
    parser.add_argument("--strategy", default="in_out", help="Strategy variant (e.g. in_out, structural_v2)")
    parser.add_argument(
        "--input-csv",
        help="Path to labeled pattern CSV (default: constructed from market/symbol/strategy)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: same in_out folder as input)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.7,
        help="Train split ratio for chronological validation (default: 0.7)",
    )
    parser.add_argument(
        "--min-context-count",
        type=int,
        default=5,
        help="Minimum context count to include in actionable list (default: 5)",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.60,
        help="Top-next probability threshold for stable context (default: 0.60)",
    )
    parser.add_argument(
        "--strong-swing-min-move-pct",
        type=float,
        default=4.0,
        help=(
            "Immediate fallback swing threshold in percent move from previous point "
            "(default: 4.0)"
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Limit analysis to last N years of data (e.g. 5). Default: all data.",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    market = args.market.lower()
    strategy = args.strategy.lower()
    default_input = Path(f"results/{market}/{symbol}/{strategy}/csv/in_out_pattern_9_18.csv")
    input_csv = Path(args.input_csv) if args.input_csv else default_input
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_csv.parent.parent
    )
    csv_dir = output_dir / "csv"
    txt_dir = output_dir / "txt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime as _dt, timedelta as _td
    _cutoff = None
    if args.years:
        _cutoff = (_dt.today() - _td(days=args.years * 365)).strftime("%Y-%m-%d")
        print(f"  Filtering analysis to last {args.years} years (from {_cutoff})")

    rows = read_rows(input_csv, date_cutoff=_cutoff)
    summary = analyze(
        rows=rows,
        out_dir=output_dir,
        csv_dir=csv_dir,
        txt_dir=txt_dir,
        split_ratio=args.split_ratio,
        min_context_count=args.min_context_count,
        stable_threshold=args.stable_threshold,
        strong_swing_min_move_pct=args.strong_swing_min_move_pct,
    )

    print("\nTransition Pattern Analysis Complete")
    print(f"  Input: {input_csv}")
    print(f"  Output dir: {output_dir}")
    print(f"  Tokens: {summary['total_tokens']}")
    print(f"  Unique 2-token contexts: {summary['total_contexts']}")
    print(f"  Stable contexts: {summary['stable_contexts']}")
    print(f"  Unstable contexts: {summary['unstable_contexts']}")
    print(
        "  Chronological split: "
        f"train end idx {summary['train_end_index']} ({summary['train_end_date']}), "
        f"recent start idx {summary['recent_start_index']} ({summary['recent_start_date']})"
    )
    print("\nGenerated files:")
    print(f"  - {csv_dir / 'stats_raw_transition_matrix.csv'}")
    print(f"  - {csv_dir / 'stats_context_summary.csv'}")
    print(f"  - {csv_dir / 'strategy_pattern_reliability.csv'}")
    print(f"  - {csv_dir / 'strategy_top_setups.csv'}")
    print(f"  - {csv_dir / 'strategy_recommendations.csv'}")
    print(f"  - {csv_dir / 'movement_clean_transitions.csv'}")
    print(f"  - {csv_dir / 'forecast_next_signal.csv'}")
    print(f"  - {csv_dir / 'forecast_confirmed_completions.csv'}")

    print(f"  - {txt_dir / 'Final_strategy_9_18.txt'}")


if __name__ == "__main__":
    main()
