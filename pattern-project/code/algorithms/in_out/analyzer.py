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


def read_rows(input_csv: Path) -> List[dict]:
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


def _date_label(value: str) -> str:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return str(value)


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
    write_csv(
        csv_dir / "stats_token_performance.csv",
        ["label", "count", "percent"],
        token_rows,
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
    write_csv(
        csv_dir / "movement_detailed_paths.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "path_result",
            "count",
            "total_context_count",
            "prob_path_given_prev2",
        ],
        clean_swing_rows,
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
    write_csv(
        csv_dir / "forecast_completion_examples.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next",
            "date_prev_2_a",
            "date_prev_2_b",
            "date_next",
            "date_completion",
            "date_prev_2_a_label",
            "date_prev_2_b_label",
            "date_next_label",
            "date_completion_label",
            "next_role",
            "completion_role",
        ],
        confirmed_examples,
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
    write_csv(
        csv_dir / "movement_history_log.csv",
        [
            "pattern_token",
            "trend_type",
            "idx_0",
            "idx_1",
            "idx_2",
            "idx_3",
            "date_0",
            "date_1",
            "date_2",
            "date_3",
            "date_0_label",
            "date_1_label",
            "date_2_label",
            "date_3_label",
        ],
        completed_patterns,
    )

    # Build the new hybrid chain based on user's specified logic.
    # It prefers continuous chains (linking via roles 1,2,3) and falls back to the next available pattern.
    completed_patterns.sort(key=lambda r: r["idx_0"])
    
    # Create a lookup map for efficient searching of pattern starts
    starts_at = {p["idx_0"]: p for p in completed_patterns}
    
    hybrid_chain = []
    visited_indices = set()
    
    master_list_idx = 0
    while master_list_idx < len(completed_patterns):
        p_current = completed_patterns[master_list_idx]
        
        if p_current["idx_0"] in visited_indices:
            master_list_idx += 1
            continue

        # Start a new chain segment
        hybrid_chain.append(p_current)
        visited_indices.add(p_current["idx_0"])
        
        # Inner loop to follow the continuous chain
        while True:
            p_next = None
            
            # Prioritized search for the next link
            for role_to_check in [1, 2, 3]:
                next_start_index = p_current[f"idx_{role_to_check}"]
                if next_start_index in starts_at:
                    potential_next = starts_at[next_start_index]
                    if potential_next["idx_0"] not in visited_indices:
                        p_next = potential_next
                        break # Found a link
            
            # If a link was found, add it and continue the inner loop
            if p_next:
                hybrid_chain.append(p_next)
                visited_indices.add(p_next["idx_0"])
                p_current = p_next
            else:
                # No continuous link found, break inner loop to start new segment
                break
        
        master_list_idx += 1


    # Re-calculate transition counters and examples based on the new hybrid_chain
    pat_counter: Dict[Context, Counter] = defaultdict(Counter)
    pat_examples: List[dict] = []
    for i in range(2, len(hybrid_chain)):
        a = hybrid_chain[i - 2]
        b = hybrid_chain[i - 1]
        c = hybrid_chain[i]
        ctx = (a["pattern_token"], b["pattern_token"])
        pat_counter[ctx][c["pattern_token"]] += 1
        pat_examples.append(
            {
                "prev_2_a": a["pattern_token"],
                "prev_2_b": b["pattern_token"],
                "next": c["pattern_token"],
                "a_date_0": a["date_0"],
                "a_date_3": a["date_3"],
                "b_date_0": b["date_0"],
                "b_date_3": b["date_3"],
                "c_date_0": c["date_0"],
                "c_date_3": c["date_3"],
                "a_date_0_label": a["date_0_label"],
                "a_date_3_label": a["date_3_label"],
                "b_date_0_label": b["date_0_label"],
                "b_date_3_label": b["date_3_label"],
                "c_date_0_label": c["date_0_label"],
                "c_date_3_label": c["date_3_label"],
            }
        )

    pat_rows = []
    pat_grouped = defaultdict(list)
    for (a, b), cnts in sorted(pat_counter.items()):
        total_ctx = sum(cnts.values())
        for nxt, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "next": nxt,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_next_pattern_given_prev2": f"{(cnt / total_ctx):.4f}",
            }
            pat_rows.append(row)
            pat_grouped[(a, b)].append(row)

    pat_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["next"])
    )
    write_csv(
        csv_dir / "movement_pattern_transitions.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next",
            "count",
            "total_context_count",
            "prob_next_pattern_given_prev2",
        ],
        pat_rows,
    )
    write_csv(
        csv_dir / "movement_transition_examples.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "next",
            "a_date_0",
            "a_date_3",
            "b_date_0",
            "b_date_3",
            "c_date_0",
            "c_date_3",
            "a_date_0_label",
            "a_date_3_label",
            "b_date_0_label",
            "b_date_3_label",
            "c_date_0_label",
            "c_date_3_label",
        ],
        pat_examples,
    )

    # The `in_out_up_down_9_18.txt` file was previously generated here,
    # but it was removed in favor of `transition_pattern_path_9_18.txt` which provides
    # the exact same transition data but also tracks intermediate invalid swings.

    # Chain view: explicit sliding sequence (A+B->C then B+C->D)
    chain_txt = txt_dir / "transition_pattern_chain_9_18.txt"
    with chain_txt.open("w", encoding="utf-8") as f:
        f.write("Pattern chain (immediate sequence, sliding window)\n")
        f.write("Each line uses consecutive complete patterns.\n\n")
        
        for i, p in enumerate(hybrid_chain):
            f.write(f"{i+1}. {p['pattern_token']} @ {p['date_0_label']}\n")
                    
        f.write("\nTransitions (Strict Patterns Only):\n")
        for i in range(2, len(hybrid_chain)):
            a = hybrid_chain[i-2]
            b = hybrid_chain[i-1]
            c = hybrid_chain[i]
            
            # Base probability for just reaching pattern C
            total_ctx = sum(pat_counter.get((a['pattern_token'], b['pattern_token']), {}).values())
            hit_count = pat_counter.get((a['pattern_token'], b['pattern_token']), {}).get(c['pattern_token'], 0)
            pct = (hit_count / total_ctx * 100.0) if total_ctx else 0.0
                
            f.write(
                f"{a['pattern_token']} + {b['pattern_token']} -> {c['pattern_token']} | "
                f"count={hit_count}/{total_ctx} ({pct:.2f}%) | "
                f"dates: {a['date_0_label']} + {b['date_0_label']} -> {c['date_0_label']}\n"
            )

    # Build iterative algorithm for transition_pattern_9_18.txt
    iterative_transitions = []
    visited_iter_inputs = set()
    
    for p_start in completed_patterns:
        if p_start["idx_0"] in visited_iter_inputs:
            continue
            
        current_a = p_start
        visited_iter_inputs.add(current_a["idx_0"])
        used_as_result = set()
        
        while True:
            # Find B: >= current_a["idx_3"], skipping and not using any C from previous iterations
            p_b = None
            for p in completed_patterns:
                if p["idx_0"] >= current_a["idx_3"] and p["idx_0"] not in used_as_result:
                    p_b = p
                    break
                    
            if not p_b:
                break
                
            # Find C: >= p_b["idx_2"]
            p_c = None
            for p in completed_patterns:
                if p["idx_0"] >= p_b["idx_2"]:
                    p_c = p
                    break
                    
            if p_c:
                intermediate_path = []
                for j in range(p_b["idx_2"], p_c["idx_0"]):
                    if labels[j] not in keep_tokens:
                        st = swing_types[j]
                        if st == "HIGH":
                            intermediate_path.append("INVALID_UP")
                        elif st == "LOW":
                            intermediate_path.append("INVALID_DOWN")
                
                if intermediate_path:
                    c_label = " -> ".join(intermediate_path) + " -> " + p_c["pattern_token"]
                else:
                    c_label = p_c["pattern_token"]
                    
                iterative_transitions.append((current_a, p_b, p_c, c_label))
                used_as_result.add(p_c["idx_0"])
                
            current_a = p_b
            visited_iter_inputs.add(current_a["idx_0"])

    iter_counter: Dict[Context, Counter] = defaultdict(Counter)
    for a, b, c, c_label in iterative_transitions:
        ctx = (a["pattern_token"], b["pattern_token"])
        iter_counter[ctx][c_label] += 1

    iter_txt = txt_dir / "strategy_final_pattern_9_18.txt"
    with iter_txt.open("w", encoding="utf-8") as f:
        f.write("Pattern iteration (A + B -> C)\n")
        f.write("B starts >= A's point 3. C starts >= B's point 2.\n")
        f.write("Next iteration input1 = B.\n\n")
        
        grouped_iter = defaultdict(list)
        for (a_token, b_token), cnts in sorted(iter_counter.items()):
            total_ctx = sum(cnts.values())
            for c_token, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
                pct = (cnt / total_ctx) * 100.0 if total_ctx else 0.0
                grouped_iter[(a_token, b_token)].append({
                    "c": c_token,
                    "count": cnt,
                    "total": total_ctx,
                    "pct": pct
                })
                
        for key in sorted(grouped_iter.keys()):
            a_token, b_token = key
            f.write(f"{a_token} + {b_token}:\n")
            for row in grouped_iter[key]:
                f.write(f"  -> {row['c']} | count={row['count']}/{row['total']} ({row['pct']:.2f}%)\n")
            f.write("\n")
            
        f.write("Chronological Iteration Sequences:\n")
        for i, (a, b, c, c_label) in enumerate(iterative_transitions):
            f.write(f"{i+1}. {a['pattern_token']} (@{a['date_0_label']}) + {b['pattern_token']} (@{b['date_0_label']}) -> {c_label} (@{c['date_0_label']})\n")

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
    parser.add_argument(
        "--input-csv",
        help="Path to labeled pattern CSV (default: results/nepal/<symbol>/in_out/csv/in_out_pattern_9_18.csv)",
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
    args = parser.parse_args()

    symbol = args.symbol.upper()
    default_input = Path(f"results/nepal/{symbol}/in_out/csv/in_out_pattern_9_18.csv")
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

    rows = read_rows(input_csv)
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
    print(f"  - {csv_dir / 'stats_token_performance.csv'}")
    print(f"  - {csv_dir / 'stats_raw_transition_matrix.csv'}")
    print(f"  - {csv_dir / 'stats_context_summary.csv'}")
    print(f"  - {csv_dir / 'strategy_pattern_reliability.csv'}")
    print(f"  - {csv_dir / 'strategy_top_setups.csv'}")
    print(f"  - {csv_dir / 'strategy_recommendations.csv'}")
    print(f"  - {csv_dir / 'movement_clean_transitions.csv'}")
    print(f"  - {csv_dir / 'movement_detailed_paths.csv'}")
    print(f"  - {csv_dir / 'forecast_next_signal.csv'}")
    print(f"  - {csv_dir / 'forecast_confirmed_completions.csv'}")
    print(f"  - {csv_dir / 'forecast_completion_examples.csv'}")
    print(f"  - {csv_dir / 'movement_history_log.csv'}")
    print(f"  - {csv_dir / 'movement_pattern_transitions.csv'}")
    print(f"  - {txt_dir / 'transition_pattern_chain_9_18.txt'}")
    print(f"  - {csv_dir / 'movement_transition_examples.csv'}")
    print(f"  - {txt_dir / 'strategy_final_pattern_9_18.txt'}")


if __name__ == "__main__":
    main()
