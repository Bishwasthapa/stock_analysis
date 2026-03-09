"""
Analyze repeating 2-token -> 1-token transition patterns from point-label output.

Input:
  stocks/nepal/<SYMBOL>/results/in_out_pattern_9_18.csv
  (expects a `point_label` column from pattern_detector_v2.py)

Outputs (CSV):
  - transition_token_summary.csv
  - transition_2to1_full.csv
  - transition_context_summary.csv
  - transition_train_recent_validation.csv
  - transition_top_actionable.csv
  - transition_easy_patterns.csv
  - transition_easy_patterns.txt
  - transition_clean_prev2_to_next.csv
  - transition_clean_prev2_to_swing.csv
  - transition_clean_prev2_to_next.txt
  - transition_clean_prev2_to_swing.txt
  - transition_clean_prev2_priority.csv
  - transition_clean_prev2_confirmed.csv
  - transition_confirmed_examples.csv
  - transition_clean_prev2_priority.txt
  - transition_clean_prev2_confirmed.txt
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
        out_dir / "transition_token_summary.csv",
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
        out_dir / "transition_2to1_full.csv",
        ["prev_2", "next", "count", "total_context_count", "prob_next_given_prev2"],
        full_rows,
    )
    context_rows_sorted = sorted(
        context_rows,
        key=lambda r: (-float(r["score_count_x_prob"]), -int(r["total_context_count"])),
    )
    write_csv(
        out_dir / "transition_context_summary.csv",
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
        out_dir / "transition_train_recent_validation.csv",
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
        out_dir / "transition_top_actionable.csv",
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
        out_dir / "transition_easy_patterns.csv",
        ["rule", "prev_2", "predicted_next", "confidence", "count", "strength"],
        easy_rows,
    )

    txt_path = out_dir / "transition_easy_patterns.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Easy Transition Rules (2-token -> next)\n")
        f.write("Sorted by count * confidence\n\n")
        for i, r in enumerate(easy_rows, start=1):
            f.write(
                f"{i}. {r['rule']} | confidence={float(r['confidence']):.2%} "
                f"| count={r['count']} | {r['strength']}\n"
            )

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

        j = i
        while j < len(labels) and labels[j] not in keep_tokens:
            j += 1
        if j >= len(labels):
            continue

        nxt = labels[j]
        clean_next_counter[(a, b)][nxt] += 1

        st = swing_types[j]
        if st == "HIGH":
            swing_label = "SWING_HIGH"
        elif st == "LOW":
            swing_label = "SWING_LOW"
        else:
            swing_label = "SWING_UNKNOWN"
        clean_swing_counter[(a, b)][swing_label] += 1

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
        out_dir / "transition_clean_prev2_to_next.csv",
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

    clean_txt = out_dir / "transition_clean_prev2_to_next.txt"
    with clean_txt.open("w", encoding="utf-8") as f:
        f.write("Clean Prev2 -> Next combinations (valid inputs only, INVALID skipped)\n\n")
        for key in sorted(grouped.keys()):
            left, right = key
            f.write(f"{left} + {right}:\n")
            for row in sorted(grouped[key], key=lambda x: (-int(x["count"]), x["next"])):
                p = float(row["prob_next_given_prev2"]) * 100
                f.write(
                    f"  -> {row['next']} | count={row['count']}/{row['total_context_count']} "
                    f"({p:.2f}%)\n"
                )
            f.write("\n")

    clean_swing_rows = []
    grouped_swing = defaultdict(list)
    for (a, b), cnts in sorted(clean_swing_counter.items()):
        total_ctx = sum(cnts.values())
        for swing_label, cnt in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
            row = {
                "prev_2_a": a,
                "prev_2_b": b,
                "swing_result": swing_label,
                "count": cnt,
                "total_context_count": total_ctx,
                "prob_swing_given_prev2": f"{(cnt / total_ctx):.4f}",
            }
            clean_swing_rows.append(row)
            grouped_swing[(a, b)].append(row)

    clean_swing_rows.sort(
        key=lambda r: (r["prev_2_a"], r["prev_2_b"], -int(r["count"]), r["swing_result"])
    )
    write_csv(
        out_dir / "transition_clean_prev2_to_swing.csv",
        [
            "prev_2_a",
            "prev_2_b",
            "swing_result",
            "count",
            "total_context_count",
            "prob_swing_given_prev2",
        ],
        clean_swing_rows,
    )

    clean_swing_txt = out_dir / "transition_clean_prev2_to_swing.txt"
    with clean_swing_txt.open("w", encoding="utf-8") as f:
        f.write("Clean Prev2 -> Swing combinations (valid inputs only, INVALID skipped)\n\n")
        for key in sorted(grouped_swing.keys()):
            left, right = key
            f.write(f"{left} + {right}:\n")
            for row in sorted(grouped_swing[key], key=lambda x: (-int(x["count"]), x["swing_result"])):
                p = float(row["prob_swing_given_prev2"]) * 100
                f.write(
                    f"  -> {row['swing_result']} | count={row['count']}/{row['total_context_count']} "
                    f"({p:.2f}%)\n"
                )
            f.write("\n")

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
        out_dir / "transition_clean_prev2_confirmed.csv",
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
        out_dir / "transition_confirmed_examples.csv",
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

    confirmed_txt = out_dir / "transition_clean_prev2_confirmed.txt"
    with confirmed_txt.open("w", encoding="utf-8") as f:
        f.write("Confirmed Prev2 -> Next combinations (requires 4-point completion)\n\n")
        for key in sorted(confirmed_grouped.keys()):
            left, right = key
            f.write(f"{left} + {right}:\n")
            for row in sorted(confirmed_grouped[key], key=lambda x: (-int(x["count"]), x["next"])):
                p = float(row["prob_next_given_prev2_confirmed"]) * 100
                f.write(
                    f"  -> {row['next']} | count={row['count']}/{row['total_context_count']} "
                    f"({p:.2f}%)\n"
                )
            f.write("\n")

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
        out_dir / "transition_clean_prev2_priority.csv",
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

    priority_txt = out_dir / "transition_clean_prev2_priority.txt"
    with priority_txt.open("w", encoding="utf-8") as f:
        f.write(
            "Priority Prev2 -> Next signal "
            "(pattern first; else immediate strong swing fallback)\n"
        )
        f.write(f"Strong swing threshold: abs move >= {strong_swing_min_move_pct:.2f}%\n\n")
        for key in sorted(priority_grouped.keys()):
            left, right = key
            f.write(f"{left} + {right}:\n")
            for row in sorted(
                priority_grouped[key],
                key=lambda x: (-int(x["count"]), x["next_signal"]),
            ):
                p = float(row["prob_signal_given_prev2"]) * 100
                f.write(
                    f"  -> {row['next_signal']} [{row['signal_type']}] | "
                    f"count={row['count']}/{row['total_context_count']} ({p:.2f}%)\n"
                )
            f.write("\n")

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
        help="Path to labeled pattern CSV (default: stocks/nepal/<symbol>/results/in_out_pattern_9_18.csv)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: same results folder as input)",
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
    default_input = Path(f"stocks/nepal/{symbol}/results/in_out_pattern_9_18.csv")
    input_csv = Path(args.input_csv) if args.input_csv else default_input
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_csv.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    summary = analyze(
        rows=rows,
        out_dir=output_dir,
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
    print(f"  - {output_dir / 'transition_token_summary.csv'}")
    print(f"  - {output_dir / 'transition_2to1_full.csv'}")
    print(f"  - {output_dir / 'transition_context_summary.csv'}")
    print(f"  - {output_dir / 'transition_train_recent_validation.csv'}")
    print(f"  - {output_dir / 'transition_top_actionable.csv'}")
    print(f"  - {output_dir / 'transition_easy_patterns.csv'}")
    print(f"  - {output_dir / 'transition_easy_patterns.txt'}")
    print(f"  - {output_dir / 'transition_clean_prev2_to_next.csv'}")
    print(f"  - {output_dir / 'transition_clean_prev2_to_next.txt'}")
    print(f"  - {output_dir / 'transition_clean_prev2_to_swing.csv'}")
    print(f"  - {output_dir / 'transition_clean_prev2_to_swing.txt'}")
    print(f"  - {output_dir / 'transition_clean_prev2_priority.csv'}")
    print(f"  - {output_dir / 'transition_clean_prev2_priority.txt'}")
    print(f"  - {output_dir / 'transition_clean_prev2_confirmed.csv'}")
    print(f"  - {output_dir / 'transition_clean_prev2_confirmed.txt'}")
    print(f"  - {output_dir / 'transition_confirmed_examples.csv'}")


if __name__ == "__main__":
    main()
