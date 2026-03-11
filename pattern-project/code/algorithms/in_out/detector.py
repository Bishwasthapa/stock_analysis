"""
Trend-Based Pattern Detection System: Detects IN and OUT patterns based on valid trend continuation.

The system analyzes swing points and marks them as part of valid trends:
- 0: Start of a valid trend (higher low + higher high for uptrend, or lower high + lower low for downtrend)
- 1, 2, 3, ...: Continuation of the trend
- OUT: Isolated points or trend breaks

IN Pattern: Points marked as 1, 2, 3... (part of continuous trends)
OUT Pattern: Points not in a trend, or 0 points without valid continuation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """Represents a single swing point in the sequence."""
    index: int
    price: float
    type: str  # 'HIGH' or 'LOW'
    date: Optional[str] = None
    date_label: Optional[str] = None
    
    def __repr__(self):
        return f"SwingPoint(idx={self.index}, price={self.price}, type={self.type})"


class DataLoader:
    """Load swing data from CSV."""
    
    @staticmethod
    def load_zigzag_csv(csv_path: str, date_cutoff: Optional[str] = None) -> List[SwingPoint]:
        """Load swing points from the zigzag CSV.
        
        Args:
            csv_path: Path to CSV file.
            date_cutoff: If set (ISO date string), only load rows on or after this date.
        """
        df = pd.read_csv(csv_path)

        if date_cutoff and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= pd.to_datetime(date_cutoff)].reset_index(drop=True)
        
        swings = []
        for idx, row in df.iterrows():
            swing = SwingPoint(
                index=idx,
                price=row['price'],
                type=row['type'].upper(),
                date=str(row.get('date', '') or '').split(' ')[0],  # keep date part only
                date_label=str(row.get('date_label', ''))
            )
            swings.append(swing)
        
        return swings


class TrendValidator:
    """Validate and identify valid trends."""
    
    @staticmethod
    def is_valid_complete_uptrend(p0: SwingPoint, p1: SwingPoint, p2: SwingPoint, p3: SwingPoint) -> bool:
        """
        Validate a COMPLETE UPTREND (0→1→2→3) with exact sequence:
        0=LOW, 1=HIGH, 2=LOW, 3=HIGH
        
        Structure Requirements:
        - HL (Higher Low): p2.LOW > p0.LOW (point 2 is higher low than point 0)
        - HH (Higher High): p3.HIGH > p1.HIGH (point 3 is higher high than point 1)
        
        This ensures the pattern forms: HL → HH (valid uptrend)
        """
        return (
            # Type sequence check
            p0.type == 'LOW' and
            p1.type == 'HIGH' and
            p2.type == 'LOW' and
            p3.type == 'HIGH' and
            # Price structure check
            p2.price > p0.price and  # HL: Higher Low
            p3.price > p1.price      # HH: Higher High
        )
    
    @staticmethod
    def is_valid_complete_downtrend(p0: SwingPoint, p1: SwingPoint, p2: SwingPoint, p3: SwingPoint) -> bool:
        """
        Validate a COMPLETE DOWNTREND (0→1→2→3) with exact sequence:
        0=HIGH, 1=LOW, 2=HIGH, 3=LOW
        
        Structure Requirements:
        - LH (Lower High): p2.HIGH < p0.HIGH (point 2 is lower high than point 0)
        - LL (Lower Low): p3.LOW < p1.LOW (point 3 is lower low than point 1)
        
        This ensures the pattern forms: LH → LL (valid downtrend)
        """
        return (
            # Type sequence check
            p0.type == 'HIGH' and
            p1.type == 'LOW' and
            p2.type == 'HIGH' and
            p3.type == 'LOW' and
            # Price structure check
            p2.price < p0.price and  # LH: Lower High
            p3.price < p1.price      # LL: Lower Low
        )
    
    @staticmethod
    def continues_uptrend(prev_swing: SwingPoint, curr_swing: SwingPoint, 
                         prev_prev: SwingPoint = None) -> bool:
        """Check if current swing continues an uptrend."""
        # In uptrend: LOW -> HIGH -> LOW(higher) -> HIGH(higher)...
        # So HIGH should be higher than previous HIGH, LOW should be higher than previous LOW
        if prev_swing.type == 'HIGH' and curr_swing.type == 'LOW':
            # LOW point: should be higher than previous LOW
            if prev_prev and prev_prev.type == 'LOW':
                return curr_swing.price > prev_prev.price
            return True  # First low in trend
        
        elif prev_swing.type == 'LOW' and curr_swing.type == 'HIGH':
            # HIGH point: should be higher than previous HIGH
            if prev_prev and prev_prev.type == 'HIGH':
                return curr_swing.price > prev_prev.price
            return True  # First high in trend
        
        return False
    
    @staticmethod
    def continues_downtrend(prev_swing: SwingPoint, curr_swing: SwingPoint,
                           prev_prev: SwingPoint = None) -> bool:
        """Check if current swing continues a downtrend."""
        # In downtrend: HIGH -> LOW -> HIGH(lower) -> LOW(lower)...
        # So HIGH should be lower than previous HIGH, LOW should be lower than previous LOW
        if prev_swing.type == 'LOW' and curr_swing.type == 'HIGH':
            # HIGH point: should be lower than previous HIGH
            if prev_prev and prev_prev.type == 'HIGH':
                return curr_swing.price < prev_prev.price
            return True  # First high in trend
        
        elif prev_swing.type == 'HIGH' and curr_swing.type == 'LOW':
            # LOW point: should be lower than previous LOW
            if prev_prev and prev_prev.type == 'LOW':
                return curr_swing.price < prev_prev.price
            return True  # First low in trend
        
        return False


class TrendDetector:
    """Detect and mark valid trends in swing sequence."""
    
    @staticmethod
    def detect_trends(swings: List[SwingPoint]) -> Tuple[Dict[int, Tuple[str, int]], List[Tuple[int, List[int], str]]]:
        """
        Detect valid trends as strict 4-point windows only:
        each valid pattern is exactly 0-1-2-3.
        Scan every next point as a potential new start (0), so windows can overlap.
        
        UPTREND (0→1→2→3): LOW → HIGH → LOW → HIGH
                 Structure: HL (Higher Low) + HH (Higher High)
                 p2.LOW > p0.LOW AND p3.HIGH > p1.HIGH
        
        DOWNTREND (0→1→2→3): HIGH → LOW → HIGH → LOW
                   Structure: LH (Lower High) + LL (Lower Low)
                   p2.HIGH < p0.HIGH AND p3.LOW < p1.LOW
        
        Any other combination is INVALID and marked as OUT.
        Returns classified swings and sequence list.
        """
        classification: Dict[int, Tuple[str, int]] = {}
        all_sequences_list: List[Tuple[int, List[int], str]] = []

        for i in range(len(swings) - 3):
            p0 = swings[i]
            p1 = swings[i + 1]
            p2 = swings[i + 2]
            p3 = swings[i + 3]

            trend_type = None
            if TrendValidator.is_valid_complete_uptrend(p0, p1, p2, p3):
                trend_type = 'UPTREND'
            elif TrendValidator.is_valid_complete_downtrend(p0, p1, p2, p3):
                trend_type = 'DOWNTREND'

            if trend_type is None:
                continue

            seq_indices = [p0.index, p1.index, p2.index, p3.index]
            all_sequences_list.append((p0.index, seq_indices, trend_type))

        # Pass ALL valid sequences, allowing overlapping sequences so that 
        # the PatternClassifier can natively process interlinking points.
        sequences_list = all_sequences_list

        membership: Dict[int, List[Tuple[str, int]]] = {}
        for seq in sequences_list:
            for role, swing_index in enumerate(seq[1]):
                membership.setdefault(swing_index, []).append((seq[2], role))

        # Decide final trend/role per swing (for basic tuple return):
        # We also prioritize role 0 here so the exporter prints it correctly.
        for swing in swings:
            entries = membership.get(swing.index, [])
            if not entries:
                classification[swing.index] = ('NONE', -1)
                continue

            zero_entries = [e for e in entries if e[1] == 0]
            nonzero_entries = [e for e in entries if e[1] > 0]
            
            if zero_entries:
                trend_type, role = zero_entries[-1]
                classification[swing.index] = (trend_type, role)
            elif nonzero_entries:
                trend_type, role = nonzero_entries[0]
                classification[swing.index] = (trend_type, role)
        
        return classification, sequences_list


class PatternClassifier:
    """Classify swings as IN or OUT."""
    
    @staticmethod
    def classify(swings: List[SwingPoint], 
                 trends: Dict[int, Tuple[str, int]],
                 sequences_list: List[Tuple[int, List[int], str]]) -> Dict[int, Tuple[str, Optional[int], str, str]]:
        """
        Classify each swing into directional IN/OUT labels.
        
        point_label values:
          - IN_UP / IN_DOWN: role 1,2,3 in an up/down valid trend
          - OUT_UP / OUT_DOWN: role 0 of an up/down valid trend
          - unlabeled: not part of any valid 0-1-2-3 trend
        """
        classification: Dict[int, Tuple[str, Optional[int], str, str]] = {}

        # Build role map per pattern and detect whether each pattern start (role 0) is IN or OUT.
        # Rule: start (0) is IN if it is also 1/2/3 of any other valid pattern.
        start_in: Dict[int, bool] = {}
        start_trend: Dict[int, str] = {}
        pattern_roles: List[Dict[int, int]] = []
        pattern_trends: List[str] = []
        for seq_idx, seq_indices, trend_type in sequences_list:
            role_map = {idx: role for role, idx in enumerate(seq_indices)}
            pattern_roles.append(role_map)
            pattern_trends.append(trend_type)
            start_trend[seq_indices[0]] = trend_type

        # Index swing membership in patterns
        swing_to_patterns: Dict[int, List[Tuple[int, int]]] = {}
        for p_idx, role_map in enumerate(pattern_roles):
            for idx, role in role_map.items():
                swing_to_patterns.setdefault(idx, []).append((p_idx, role))

        # Determine whether each pattern start is IN (overlaps another pattern's 1/2/3)
        for p_idx, role_map in enumerate(pattern_roles):
            start_idx = [k for k, v in role_map.items() if v == 0]
            if not start_idx:
                continue
            s_idx = start_idx[0]
            overlaps = swing_to_patterns.get(s_idx, [])
            start_in[p_idx] = any(role in (1, 2, 3) and other_idx != p_idx for other_idx, role in overlaps)

        # Assign labels based on pattern start rule:
        # - role 0 is OUT unless it overlaps another pattern's 1/2/3
        # - roles 1/2/3 are always IN (trend direction)
        for swing in swings:
            memberships = swing_to_patterns.get(swing.index, [])
            if not memberships:
                continue

            # Find if this swing acts as role 0 in ANY pattern
            role_0_patterns = [p for p, role in memberships if role == 0]
            
            if role_0_patterns:
                # Prioritize the pattern where it is role 0.
                # If there are multiple, just pick the first one.
                chosen_p = role_0_patterns[-1]
                role = 0
            else:
                # If not role 0 anywhere, prioritize IN patterns
                in_patterns = [p for p, _ in memberships if start_in.get(p, False)]
                chosen_p = in_patterns[0] if in_patterns else memberships[0][0]
                role = dict(memberships).get(chosen_p, memberships[0][1])

            trend_type = pattern_trends[chosen_p]

            if role == 0 and not start_in.get(chosen_p, False):
                if trend_type == 'UPTREND':
                    classification[swing.index] = ('OUT', role, trend_type, 'OUT_UP')
                else:
                    classification[swing.index] = ('OUT', role, trend_type, 'OUT_DOWN')
            else:
                if trend_type == 'UPTREND':
                    classification[swing.index] = ('IN', role, trend_type, 'IN_UP')
                else:
                    classification[swing.index] = ('IN', role, trend_type, 'IN_DOWN')

        return classification


class ResultExporter:
    """Export results to CSV and display."""
    
    @staticmethod
    def export_classification(swings: List[SwingPoint], 
                             classification: Dict[int, Tuple[str, Optional[int], str, str]],
                             output_path: str) -> None:
        """Export classification results to CSV."""
        results = []
        
        for swing in swings:
            if swing.index in classification:
                cls, role, trend_type, point_label = classification[swing.index]
            else:
                cls, role, trend_type, point_label = '', None, '', ''
            
            role_str = str(role) if role is not None and role >= 0 else ''
            
            results.append({
                'index': swing.index,
                'price': swing.price,
                'type': swing.type,
                'date': swing.date,
                'date_label': swing.date_label,
                'pattern_role': role_str,
                'classification': cls,
                'trend_type': trend_type,
                'point_label': point_label
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✓ Results exported to {output_path}")
        
        # Print summary
        in_count = (df['classification'] == 'IN').sum()
        out_count = (df['classification'] == 'OUT').sum()
        unlabeled_count = (df['classification'] == '').sum()
        in_up_count = (df['point_label'] == 'IN_UP').sum()
        in_down_count = (df['point_label'] == 'IN_DOWN').sum()
        out_up_count = (df['point_label'] == 'OUT_UP').sum()
        out_down_count = (df['point_label'] == 'OUT_DOWN').sum()
        print(f"\nSummary:")
        print(f"  IN points: {in_count}")
        print(f"  OUT points: {out_count}")
        print(f"  Unlabeled points: {unlabeled_count}")
        print(f"  IN_UP: {in_up_count}")
        print(f"  IN_DOWN: {in_down_count}")
        print(f"  OUT_UP: {out_up_count}")
        print(f"  OUT_DOWN: {out_down_count}")
        print(f"  Total swings: {len(df)}")
    
    @staticmethod
    def plot_classification(swings: List[SwingPoint], 
                           classification: Dict[int, Tuple[str, Optional[int], str, str]],
                           symbol: str, output_path: str) -> None:
        """Visualize the IN/OUT classification with pattern role labels."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import pandas as pd
        except ImportError:
            print("⚠ Matplotlib not available, skipping visualization")
            return
        
        # Prepare data
        dates = [pd.to_datetime(s.date) for s in swings]
        date_by_index = {s.index: pd.to_datetime(s.date) for s in swings}
        prices = [s.price for s in swings]

        # Adaptive canvas: keep readable but avoid excessive rendering cost.
        date_span_days = max(1, int((max(dates) - min(dates)).days)) if dates else 1
        width = min(54, max(26, date_span_days / 45.0))
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(width, 10))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        import math as _math

        price_range = max(prices) - min(prices) if len(prices) > 1 else 1
        swing_idx_list = [s.index for s in swings]
        swing_by_index = {s.index: s for s in swings}

        def _gp(idx): s = swing_by_index.get(idx); return s.price if s else float('nan')
        def _gt(idx): s = swing_by_index.get(idx); return s.type.upper() if s else ''

        # High-contrast neon palette on dark background
        COLOR = {
            'IN_UP':   '#00e676',  # neon green
            'IN_DOWN': '#29b6f6',  # bright sky blue
            'OUT_UP':  '#ffa726',  # vivid amber
            'OUT_DOWN':'#ef5350',  # bright red
        }

        # Build per-swing: (primary_label, all_roles_per_label)
        # multi_roles[swap_index] -> list of (role_num, point_label)
        multi_roles: dict = {}
        for i in range(len(swing_idx_list) - 3):
            i0,i1,i2,i3 = swing_idx_list[i],swing_idx_list[i+1],swing_idx_list[i+2],swing_idx_list[i+3]
            p0r,p1r,p2r,p3r = _gp(i0),_gp(i1),_gp(i2),_gp(i3)
            t0r,t1r,t2r,t3r = _gt(i0),_gt(i1),_gt(i2),_gt(i3)
            if any(_math.isnan(px) for px in [p0r,p1r,p2r,p3r]): continue
            is_up = t0r=='LOW' and t1r=='HIGH' and t2r=='LOW' and t3r=='HIGH' and p2r>p0r and p3r>p1r
            is_dn = t0r=='HIGH' and t1r=='LOW' and t2r=='HIGH' and t3r=='LOW' and p2r<p0r and p3r<p1r
            if not (is_up or is_dn): continue
            for rn, idx in enumerate([i0,i1,i2,i3]):
                if rn == 0:
                    c0 = classification.get(idx, ('',))[0]
                    lbl = ('IN' if c0=='IN' else 'OUT') + ('_UP' if is_up else '_DOWN')
                else:
                    lbl = 'IN_UP' if is_up else 'IN_DOWN'
                entry = (rn, lbl)
                if entry not in multi_roles.get(idx, []):
                    multi_roles.setdefault(idx, []).append(entry)

        # Thin zigzag base
        ax.plot(dates, prices, color='#95a5a6', linewidth=1.5, alpha=0.5, zorder=1)

        # Dots + role badges
        seen_dot: dict = {}  # idx -> list of labels drawn (to avoid identical repeats)
        for swing in swings:
            x = date_by_index.get(swing.index, pd.to_datetime(swing.date))
            roles = multi_roles.get(swing.index, [])
            if not roles:
                # unlabeled point
                ax.scatter(x, swing.price, color='#bdc3c7', s=60, zorder=3, alpha=0.5)
                continue

            # Draw a dot per unique label at this point
            drawn_lbls = []
            for (rn, lbl) in roles:
                color = COLOR.get(lbl, '#7f8c8d')
                if lbl not in drawn_lbls:
                    ax.scatter(x, swing.price, color=color, s=180, zorder=4,
                               alpha=0.6, edgecolors='white', linewidth=1)
                    drawn_lbls.append(lbl)

            # Single role badge — just the primary (first) role at this point
            rn, lbl = roles[0]
            color = COLOR.get(lbl, '#7f8c8d')
            y = swing.price + price_range * 0.05
            ax.text(x, y, str(rn), fontsize=7.5, fontweight='bold',
                    ha='center', va='bottom', color='white', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.18', facecolor=color,
                              alpha=0.88, edgecolor='none'))

        # Date labels: 45° rotated, alternating offset below dot
        for idx_e, swing in enumerate(swings):
            x = date_by_index.get(swing.index, pd.to_datetime(swing.date))
            y_off = price_range * (0.03 if idx_e % 2 == 0 else 0.055)
            ax.text(x, swing.price - y_off, x.strftime('%d %b'),
                    fontsize=7.5, ha='right', va='top', rotation=40,
                    color='#8b949e', alpha=0.9,
                    fontweight='bold')

        ax.set_xlabel('Year', fontsize=12, color='#c9d1d9')
        ax.set_ylabel('Price', fontsize=12, color='#c9d1d9')
        ax.set_title(f'{symbol} — IN/OUT Pattern Classification', fontsize=14,
                     fontweight='bold', color='#f0f6fc')
        ax.grid(True, alpha=0.12, linestyle='--', color='#30363d')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['IN_UP'],   markersize=10, label='IN_UP'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['IN_DOWN'],  markersize=10, label='IN_DOWN'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['OUT_UP'],   markersize=10, label='OUT_UP'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['OUT_DOWN'], markersize=10, label='OUT_DOWN'),
        ], loc='upper left', fontsize=10, framealpha=0.85)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0); lbl.set_horizontalalignment('center'); lbl.set_fontsize(10)
        plt.tight_layout()

        fig.savefig(output_path, dpi=260, bbox_inches='tight')
        print(f"✓ Classification chart saved: {output_path}")




def analyze_trend_pattern(symbol: str, zigzag_csv: str, output_csv: str, date_cutoff: Optional[str] = None):
    """
    Main entry point: run the trend-based pattern detection pipeline.
    
    Args:
        date_cutoff: ISO date string (YYYY-MM-DD). Only swing points on/after this date are analyzed.
    """
    print(f"\n{'='*60}")
    print(f"Trend-Based Pattern Detection: {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Load swing data
    print("1. Loading swing data...")
    swings = DataLoader.load_zigzag_csv(zigzag_csv, date_cutoff=date_cutoff)
    print(f"   Loaded {len(swings)} swing points")
    
    # 2. Detect trends
    print("2. Detecting valid trends (HH/HL for uptrend, LL/LH for downtrend)...")
    trends, sequences_list = TrendDetector.detect_trends(swings)
    
    trend_count = sum(1 for v in trends.values() if v[1] >= 0)
    uptrend_count = sum(1 for v in trends.values() if v[0] == 'UPTREND')
    downtrend_count = sum(1 for v in trends.values() if v[0] == 'DOWNTREND')
    print(f"   Found {trend_count} points in valid trends")
    print(f"     - Uptrend: {uptrend_count} points")
    print(f"     - Downtrend: {downtrend_count} points")
    
    # 3. Classify swings
    print("3. Classifying swing points as IN or OUT...")
    classification = PatternClassifier.classify(swings, trends, sequences_list)
    
    # 4. Export results
    print("4. Exporting results...")
    import os
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    ResultExporter.export_classification(swings, classification, output_csv)
    
    # 5. Visualize classification
    print("5. Generating classification visualization...")
    viz_path = output_csv.replace("/csv/", "/png/").replace(".csv", "_visualization.png")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    ResultExporter.plot_classification(swings, classification, symbol, viz_path)
    
    print(f"\n✓ Trend-based pattern detection complete!\n")


if __name__ == '__main__':
    import sys
    import argparse as _ap
    from datetime import datetime, timedelta

    _parser = _ap.ArgumentParser(description="Trend-based pattern detector.")
    _parser.add_argument("symbol")
    _parser.add_argument("--zigzag-csv", default=None)
    _parser.add_argument("--output-csv", default=None)
    _parser.add_argument("--years", type=int, default=None,
                         help="Limit analysis to last N years (e.g. 5)")
    _args = _parser.parse_args()

    _symbol = _args.symbol.upper()
    _zigzag_csv = _args.zigzag_csv or f'results/nepal/{_symbol}/in_out/csv/highs_lows_pattern_9_18.csv'
    _output_csv = _args.output_csv or f'results/nepal/{_symbol}/in_out/csv/in_out_pattern_9_18.csv'

    _cutoff = None
    if _args.years:
        _cutoff = (datetime.today() - timedelta(days=_args.years * 365)).strftime('%Y-%m-%d')
        print(f"  Filtering data to last {_args.years} years (from {_cutoff})")

    analyze_trend_pattern(_symbol, _zigzag_csv, _output_csv, date_cutoff=_cutoff)
