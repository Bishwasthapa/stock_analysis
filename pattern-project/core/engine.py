import pandas as pd
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import requests
import io
import os
import cv2
from skimage.morphology import skeletonize

class PatternEngine:
    # A library of named structural shapes
    PATTERNS = {
        "spring": np.array([0.5, 0.4, 0.6, 0.5, 0.45, 0.1, 0.7, 1.0, 1.2, 1.5]), # Accumulation -> Dip -> Move Up
        "head_and_shoulders": np.array([0.5, 0.8, 0.5, 1.0, 0.5, 0.8, 0.5]),
        "double_bottom": np.array([1.0, 0.2, 0.6, 0.2, 1.0]),
        "pattern1": np.array([0.5, 0.4, 0.6, 0.5, 0.45, 0.1, 0.7, 1.0, 1.2, 1.5]), # Alias for spring
    }

    def __init__(self, ema_short=9, ema_long=18):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.data = None

    def _normalize_nepse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal helper to normalise column names and date for NEPSE data."""
        df.columns = [col.capitalize() for col in df.columns]
        # Normalize date column
        date_cols = ['Date', 'Published_date', 'Datetime']
        for col in date_cols:
            if col in df.columns:
                df['Date'] = pd.to_datetime(df[col])
                break
        df = df.sort_values('Date').reset_index(drop=True)
        return df

    def load_nepse_data(self, symbol, cache=True):
        """Fetches historical data from the NEPSE GitHub repository.

        If `cache` is True the CSV will be written to `data/{symbol}.csv` and any
        existing file will be used instead of re‑downloading.  This makes it
        easier to work offline and keeps a local copy of the raw OHLC data.
        """
        print(f"Loading NEPSE data for {symbol}...")
        cache_path = os.path.join(os.getcwd(), 'data', f'{symbol}.csv')

        # try local cache first
        if cache and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                df.columns = [col.capitalize() for col in df.columns]
            except Exception:
                # if reading cache fails, fall back to remote fetch
                df = None
            else:
                print(f"Loaded {symbol} from cache {cache_path}")
                df = self._normalize_nepse(df)
                self.data = df
                return True

        url = f"https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise/{symbol}.csv"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                df = self._normalize_nepse(df)
                self.data = df
                if cache:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    df.to_csv(cache_path, index=False)
                return True
        except Exception as e:
            print(f"Error loading NEPSE data: {e}")
        return False

    def load_intl_data(self, symbol, period="5y"):
        """Fetches historical data using yfinance."""
        print(f"Loading International data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if not df.empty:
                df.reset_index(inplace=True)
                # Ensure the date column is just 'Date' for consistency
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df.rename(columns={'Datetime': 'Date'}, inplace=True)
                self.data = df
                return True
        except Exception as e:
            print(f"Error loading International data: {e}")
        return False

    def calculate_indicators(self):
        """Calculates EMAs and crossovers."""
        if self.data is None: return
        self.data[f'EMA_{self.ema_short}'] = self.data['Close'].ewm(span=self.ema_short, adjust=False).mean()
        self.data[f'EMA_{self.ema_long}'] = self.data['Close'].ewm(span=self.ema_long, adjust=False).mean()
        
        # Crossover logic
        self.data['Signal'] = 0
        self.data.loc[self.data[f'EMA_{self.ema_short}'] > self.data[f'EMA_{self.ema_long}'], 'Signal'] = 1
        self.data.loc[self.data[f'EMA_{self.ema_short}'] < self.data[f'EMA_{self.ema_long}'], 'Signal'] = -1
        self.data['Crossover'] = self.data['Signal'].diff()

    def find_ema_repeats(self, window_size=30, top_k=3):
        """Finds historical segments that had a similar EMA crossover setup."""
        if self.data is None or len(self.data) < window_size: return []
        
        current_pattern = self.data.tail(window_size)[f'EMA_{self.ema_short}'].values
        target = (current_pattern - np.mean(current_pattern)) / np.std(current_pattern)
        
        matches = []
        historical = self.data.iloc[:-window_size]
        for i in range(len(historical) - window_size):
            seg = historical.iloc[i : i + window_size][f'EMA_{self.ema_short}'].values
            norm_seg = (seg - np.mean(seg)) / np.std(seg)
            distance = dtw.distance(target, norm_seg)
            matches.append({'index': i, 'date': historical.iloc[i+window_size-1]['Date'], 'distance': distance})
        
        matches.sort(key=lambda x: x['distance'])
        return matches[:top_k]

    def find_structural_matches(self, pattern_name=None, template=None, window_size=30, top_k=3, stride=1):
        """Finds geometric matches. Can use a named pattern or a custom template array."""
        if self.data is None: return []
        
        # Determine the target template
        target_template = template
        if target_template is None:
            # Fallback to a named pattern if provided, otherwise default to "spring"
            name = pattern_name if pattern_name in self.PATTERNS else "spring"
            target_template = self.PATTERNS[name]
        
        # Optimization: If window is large and stride is 1, auto-stride to save time
        if stride == 1 and len(self.data) > 1000:
            stride = max(1, int(window_size / 10))

        results = []
        for i in range(0, len(self.data) - window_size - 10, stride):
            segment = self.data.iloc[i : i + window_size]['Close'].values
            s_min, s_max = segment.min(), segment.max()
            if s_max == s_min: continue
            
            # Normalize segment
            norm_seg = (segment - s_min) / (s_max - s_min)
            
            # RESAMPLING: Ensure template and segment are the same length for DTW
            # This makes the match "Scale-Invariant" (size doesn't matter, shape does)
            def resample(arr, target_len):
                if len(arr) == target_len: return arr
                return np.interp(
                    np.linspace(0, 1, target_len),
                    np.linspace(0, 1, len(arr)),
                    arr
                )
            
            target_len = 100 # Standardizing for comparison
            resampled_template = resample(target_template, target_len)
            resampled_seg = resample(norm_seg, target_len)
            
            # Use DTW to compare resampled shapes
            distance = dtw.distance(resampled_template, resampled_seg)
            
            # Rule-based structural check specific to "Spring"
            is_spring = False
            if pattern_name == "spring" or (pattern_name is None and template is None):
                consolidation = segment[:int(window_size*0.7)]
                spring_part = segment[int(window_size*0.7):int(window_size*0.8)]
                recovery_part = segment[int(window_size*0.8):]
                if spring_part.min() < consolidation.min() and recovery_part.max() > consolidation.max():
                    is_spring = True

            # Result formatting
            date_val = self.data.iloc[i+window_size]['Date'] if 'Date' in self.data.columns else (i+window_size)
            
            # For the pattern image itself, the match is the whole template we searched for.
            pattern_width = len(template) if template is not None else 0
            
            if is_spring:
                results.append({'index': i, 'date': date_val, 'distance': distance, 'structural': True, 'pattern_start': 0, 'pattern_end': pattern_width})
            elif distance < 1.0:
                results.append({'index': i, 'date': date_val, 'distance': distance, 'structural': False, 'pattern_start': 0, 'pattern_end': pattern_width})

        results.sort(key=lambda x: (not x.get('structural', False), x['distance']))
        return results[:top_k]

    def find_self_repeats(self, window_size=40, top_k=3):
        """Finds where the recent price action happened before in this same stock."""
        if self.data is None or len(self.data) < window_size * 2: return []
        
        current_segment = self.data.tail(window_size)['Close'].values
        cur_min, cur_max = current_segment.min(), current_segment.max()
        target = (current_segment - cur_min) / (cur_max - cur_min)
        
        return self.find_structural_matches(template=target, window_size=window_size, top_k=top_k)

    def find_all_self_patterns(self, window_size=100, num_patterns=2, matches_per_pattern=2, stride=10, distance_threshold=2.5):
        """Scans the entire chart to find distinct, repeating 'unique' patterns.
        Instead of just matching one template, it compares every segment to every other segment
        to find clusters of similar shapes.
        """
        if self.data is None or len(self.data) < window_size * 2:
            print("Not enough data points for self-discovery.")
            return []

        print(f"Scanning for {num_patterns} unique patterns that repeat at least {matches_per_pattern} times...")
        
        # 1. Extract all possible valid segments
        segments = []
        indices = []
        target_len = 100
        
        for i in range(0, len(self.data) - window_size, stride):
            seg_raw = self.data.iloc[i : i + window_size]['Close'].values
            s_min, s_max = seg_raw.min(), seg_raw.max()
            if s_max == s_min: continue
            
            # Normalize and resample to "DNA" for fair comparison
            norm_seg = (seg_raw - s_min) / (s_max - s_min)
            resampled_seg = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(norm_seg)), norm_seg)
            segments.append(resampled_seg)
            indices.append(i)
            
        n_segs = len(segments)
        if n_segs < matches_per_pattern: return []

        print(f"Extracted {n_segs} segments for comparison. Building distance matrix (this may take a moment)...")
        
        # 2. Build a distance matrix between all segments
        # Optimization: Only calculate upper triangle, it's symmetric
        dist_matrix = np.full((n_segs, n_segs), np.inf)
        for i in range(n_segs):
            for j in range(i + 1, n_segs):
                # Don't compare overlapping segments to prevent trivially matching with itself shifted by 1 pixel
                if abs(indices[i] - indices[j]) > window_size // 2:
                    dist = dtw.distance(segments[i], segments[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist # symmetric

        # 3. Find clusters (Unique Patterns)
        clusters = []
        used_indices = set()
        
        for cluster_id in range(num_patterns):
            best_cluster = []
            best_density = np.inf
            
            # Find the segment that has the closest 'matches_per_pattern' neighbors
            for i in range(n_segs):
                if i in used_indices: continue
                
                # Get distances from i to all others, ignoring already used segments
                dists = []
                for j in range(n_segs):
                    if j != i and j not in used_indices:
                        dists.append((j, dist_matrix[i, j]))
                
                # Sort by distance
                dists.sort(key=lambda x: x[1])
                
                # Check the closest required matches
                top_neighbors = dists[:matches_per_pattern]
                
                # If the furtest neighbor is still reasonably close
                if len(top_neighbors) == matches_per_pattern and top_neighbors[-1][1] < distance_threshold:
                    # The "density" is the sum of distances to these neighbors (lower is better)
                    density = sum(d[1] for d in top_neighbors)
                    if density < best_density:
                        best_density = density
                        # The cluster consists of the center segment and its neighbors
                        best_cluster = [{'index': indices[i], 'distance': 0.0, 'dtw_dna': segments[i]}]
                        for neighbor_idx, dist in top_neighbors:
                            best_cluster.append({'index': indices[neighbor_idx], 'distance': dist, 'dtw_dna': segments[neighbor_idx]})

            if best_cluster:
                # We found a valid cluster! Record it and mark these indices as used
                clusters.append({
                    'pattern_id': chr(65 + cluster_id), # A, B, C...
                    'matches': sorted(best_cluster, key=lambda x: x['index']),
                    'template_dna': best_cluster[0]['dtw_dna'] # The center segment acts as the template
                })
                print(f"Found Pattern {chr(65 + cluster_id)} with {len(best_cluster)} occurrences.")
                
                # Mark entire overlapping regions as used
                for match in best_cluster:
                    idx = match['index']
                    for j in range(n_segs):
                        if abs(indices[j] - idx) < window_size:
                            used_indices.add(j)
            else:
                print(f"Could not find any more unique repeating patterns.")
                break
                
        return clusters

    def find_swing_points(self, lookback=10):
        """Detects all local peaks and valleys (swing highs/lows) in the data.
        Returns a list of dicts: {'index': int, 'value': float, 'type': 'high'|'low'}
        """
        prices = self.data['Close'].values
        n = len(prices)
        swings = []
        
        for i in range(lookback, n - lookback):
            window = prices[i - lookback : i + lookback + 1]
            center = prices[i]
            
            if center == window.max() and center > prices[i-1] and center > prices[i+1]:
                swings.append({'index': i, 'value': float(center), 'type': 'high'})
            elif center == window.min() and center < prices[i-1] and center < prices[i+1]:
                swings.append({'index': i, 'value': float(center), 'type': 'low'})
        
        # Remove duplicates / too-close swings (keep the more extreme one)
        filtered = []
        for s in swings:
            if filtered and abs(s['index'] - filtered[-1]['index']) < lookback:
                # Replace if same type and more extreme, or skip if different type
                prev = filtered[-1]
                if s['type'] == prev['type']:
                    if (s['type'] == 'high' and s['value'] > prev['value']) or \
                       (s['type'] == 'low' and s['value'] < prev['value']):
                        filtered[-1] = s
            else:
                filtered.append(s)
        
        return filtered

    def run_inout_scanner(self, swing_points):
        """Applies the 0→1→2→3 in-out sequential labeling algorithm.
        
        Valid UP pattern: 0=low, 1=high (>0), 2=low (>0, <1), 3=high (>1)
        Valid DOWN pattern: 0=high, 1=low (<0), 2=high (<0, >1), 3=low (<1)
        
        Returns:
          valid_patterns: list of patterns, each a list of 4 swing-point dicts with role label
          out_pattern_indices: set of swing_point indices that are OUT PATTERN
        """
        n = len(swing_points)
        valid_patterns = []      # Each item: list of 4 {'point': dict, 'role': '0'|'1'|'2'|'3'}
        
        # Track which swing-point data-indices (from self.data) appear in any valid pattern
        # key = swing point 'index' value, value = list of roles it plays
        point_roles = {}   # {data_index: ['0', '1', ...]}
        
        i = 0
        while i <= n - 4:
            sp = swing_points[i]
            
            # Try to build a 4-point pattern from i..i+3
            pts = swing_points[i : i + 4]
            
            # Check alternation (high-low-high-low or low-high-low-high)
            types = [p['type'] for p in pts]
            vals = [p['value'] for p in pts]
            
            is_up = types == ['low', 'high', 'low', 'high']
            is_down = types == ['high', 'low', 'high', 'low']
            
            valid_up = False
            valid_down = False
            
            # Calculate the minimum swing size (avoid marking tiny noise as patterns)
            # At least 1% of the price range must be between consecutive swings
            prices = self.data['Close'].values
            price_range = prices.max() - prices.min()
            min_swing = price_range * 0.01
            
            if is_up:
                # Valid UP: alternating L-H-L-H with minimum swing amplitude
                swings_ok = all(abs(vals[k+1] - vals[k]) > min_swing for k in range(3))
                valid_up = swings_ok
            
            if is_down:
                # Valid DOWN: alternating H-L-H-L with minimum swing amplitude
                swings_ok = all(abs(vals[k+1] - vals[k]) > min_swing for k in range(3))
                valid_down = swings_ok
            
            if valid_up or valid_down:
                pattern = [{'point': pts[k], 'role': str(k)} for k in range(4)]
                valid_patterns.append(pattern)
                
                for k, pt in enumerate(pts):
                    idx = pt['index']
                    if idx not in point_roles:
                        point_roles[idx] = []
                    point_roles[idx].append(str(k))
                
                i += 1  # Advance one step (patterns can share points)
            else:
                i += 1  # Swing point couldn't form a pattern, move on
        
        # Now determine OUT PATTERN points:
        # A point is OUT if:
        # 1) It never appeared in any valid pattern at all, OR
        # 2) It appeared ONLY as role '0' and not in any other role in any other pattern
        out_pattern_indices = set()
        
        for sp in swing_points:
            idx = sp['index']
            roles = point_roles.get(idx, [])
            
            if not roles:
                # Never appeared in any valid pattern → OUT
                out_pattern_indices.add(idx)
            elif roles == ['0'] or all(r == '0' for r in roles):
                # Only ever appeared as a "0" (pattern start) but never as 1,2,3 in another pattern
                # Check: does this point appear as point 1, 2, or 3 in ANY pattern?
                appears_as_non_zero = any(r != '0' for r in roles)
                if not appears_as_non_zero:
                    out_pattern_indices.add(idx)
        
        print(f"In-Out Scanner: found {len(valid_patterns)} valid patterns, {len(out_pattern_indices)} OUT PATTERN points.")
        return valid_patterns, out_pattern_indices

    def draw_inout_results(self, image_path, swing_points, valid_patterns, out_pattern_indices,
                           output_file, data_width=None):
        """Draws the in-out scan results on the original image.
        
        - Valid pattern points get numbered labels (0, 1, 2, 3)
        - OUT PATTERN points get red filled circles
        Auto-crops the image to the chart region first (same as load_image_as_data).
        """
        img_full = cv2.imread(image_path)
        if img_full is None:
            print(f"Error: Could not read image {image_path}")
            return

        # Auto-crop to the chart area (same as load_image_as_data does)
        hsv = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        img, crop_x, crop_y = self._auto_crop_chart(img_full, mask)

        img_h, img_w = img.shape[:2]
        prices = self.data['Close'].values
        n_data = len(prices)
        
        # --- CORRECT COORDINATE MAPPING ---
        # During extraction: Close[i] = h - pixel_y  (see load_image_as_data)
        # So to get back: pixel_y = img_h - Close[i]
        # x: data index maps directly to column in the trimmed skeleton.
        # Since trim_start is typically ~0, data_to_x(i) ≈ i
        # But scale to the actual image width to be safe.
        def data_to_x(idx):
            return max(0, min(img_w - 1, int(idx)))
        
        def val_to_y(val):
            # val = h - pixel_y  =>  pixel_y = img_h - val
            y = int(img_h - val)
            return max(0, min(img_h - 1, y))
        
        # Collect all valid point indices and their roles for checking 0+OUT overlap
        valid_point_roles = {}  # data_index → role
        for pattern in valid_patterns:
            for p in pattern:
                didx = p['point']['index']
                valid_point_roles.setdefault(didx, []).append(p['role'])
        
        # --- Draw labels for valid pattern points (NO connecting lines) ---
        drawn_at = set()
        for pattern in valid_patterns:
            for p in pattern:
                didx = p['point']['index']
                role = p['role']
                x = data_to_x(didx)
                y = val_to_y(p['point']['value'])
                
                label_key = (didx, role)
                if label_key in drawn_at:
                    continue
                drawn_at.add(label_key)
                
                # Color per role: 0=white, 1=yellow, 2=cyan, 3=lime green
                role_colors = {'0': (220,220,220), '1': (0,200,255), '2': (0,255,255), '3': (0,255,80)}
                color = role_colors.get(role, (255,255,255))
                
                # Small dot on the line
                cv2.circle(img, (x, y), 8, color, -1, cv2.LINE_AA)
                # Label above peaks, below valleys
                offset_y = y - 20 if p['point']['type'] == 'high' else y + 30
                offset_y = max(18, min(img_h - 8, offset_y))
                cv2.putText(img, role, (x - 7, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)
        
        # --- Draw OUT PATTERN red circles (including points that are only '0') ---
        for sp in swing_points:
            if sp['index'] in out_pattern_indices:
                x = data_to_x(sp['index'])
                y = val_to_y(sp['value'])
                # Red ring around the point
                cv2.circle(img, (x, y), 16, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                
                # If this point was ALSO labeled as '0' in a valid pattern, keep the '0' label
                if sp['index'] in valid_point_roles and '0' in valid_point_roles[sp['index']]:
                    offset_y = y - 20 if sp['type'] == 'high' else y + 30
                    offset_y = max(18, min(img_h - 8, offset_y))
                    cv2.putText(img, '0', (x - 7, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (220, 220, 220), 2, cv2.LINE_AA)
        
        cv2.imwrite(output_file, img)
        print(f"In-Out scan result saved to {output_file}")

    def label_image(self, image_path, label, output_path, color=(0, 100, 255)):

        """Writes a large label on an image and saves it."""
        img = cv2.imread(image_path)
        if img is None: return
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.imwrite(output_path, img)
        print(f"Labeled image saved to {output_path}")

    def create_match_report(self, template, matches, window_size, output_file="match_report.png", target_name="Original Blueprint #1", clusters=None):
        """Vision Tasks: Create a comparison report.
        If 'clusters' is provided, it plots multiple unique self-discovered patterns.
        Otherwise, it plots a single template vs its matches.
        """
        def resample(arr, target_len=100):
            return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(arr)), arr)

        if clusters:
            # --- REDESIGNED: One row per unique pattern group ---
            # Each row shows all instances of that pattern OVERLAID on the same axes
            # so it's instantly obvious they look the same.
            n_clusters = len(clusters)
            max_instances = max(len(c['matches']) for c in clusters)
            
            # Layout: left column = all instances overlaid, right columns = individual instances
            n_cols = max_instances + 1  # 1 overlay col + N individual cols
            fig, axes = plt.subplots(n_clusters, n_cols,
                                     figsize=(5 * n_cols, 4 * n_clusters),
                                     squeeze=False)
            fig.suptitle("Self-Repeating Pattern Discovery\n(Patterns in the same row are similar to each other)",
                         fontsize=14, fontweight='bold', y=1.02)
            
            colors_per_cluster = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            # Use slightly different shades for each instance in the same cluster
            instance_alphas = [1.0, 0.65, 0.4]

            for c_idx, cluster in enumerate(clusters):
                pat_id = cluster['pattern_id']
                base_color = colors_per_cluster[c_idx]
                
                all_segments = []
                for match in cluster['matches']:
                    seg = self.data.iloc[match['index'] : match['index'] + window_size]['Close'].values
                    norm = (seg - seg.min()) / (seg.max() - seg.min()) if seg.max() != seg.min() else seg
                    all_segments.append((resample(norm), match['index']))

                # Left column (col 0): Overlay ALL instances on same axes
                ax_overlay = axes[c_idx, 0]
                for inst_i, (seg_dna, seg_idx) in enumerate(all_segments):
                    label = f"Instance {pat_id}-{inst_i+1} (starts x={seg_idx})"
                    alpha = instance_alphas[inst_i] if inst_i < len(instance_alphas) else 0.3
                    ax_overlay.plot(seg_dna, color=base_color, alpha=alpha,
                                    linewidth=2.5 - (inst_i * 0.5), label=label)
                ax_overlay.set_title(f"▶ Pattern {pat_id} — All {len(all_segments)} instances overlaid\n(They all share this shape)",
                                     fontsize=10, fontweight='bold', color=base_color)
                ax_overlay.legend(fontsize=7, loc='upper right')
                ax_overlay.set_ylim(-0.1, 1.1)
                ax_overlay.grid(True, alpha=0.2)
                ax_overlay.set_facecolor('#f9f9f9')

                # Right columns (col 1..N): Individual instances
                for inst_i, (seg_dna, seg_idx) in enumerate(all_segments):
                    ax = axes[c_idx, inst_i + 1]
                    ax.plot(seg_dna, color=base_color, linewidth=2)
                    ax.set_title(f"{pat_id}-{inst_i+1}\n(at x={seg_idx})", fontsize=9)
                    ax.set_ylim(-0.1, 1.1)
                    ax.grid(True, alpha=0.2)
                
                # Hide any empty columns
                for empty_col in range(len(all_segments) + 1, n_cols):
                    axes[c_idx, empty_col].axis('off')

            plt.tight_layout()

        else:
            # Single template match report
            num_matches = len(matches)
            fig, axes = plt.subplots(num_matches + 1, 2, figsize=(15, 4 * (num_matches + 1)))
            
            # Normalize template
            norm_template = (template - template.min()) / (template.max() - template.min())
            res_template = resample(norm_template)

            # Row 0: Reference Target
            axes[0, 0].plot(norm_template, color='blue', linewidth=2)
            axes[0, 0].set_title(target_name)
            axes[0, 1].plot(res_template, color='blue', linestyle='--', linewidth=2)
            axes[0, 1].set_title(f"{target_name} DNA (Resampled)")

            for i, m in enumerate(matches):
                segment = self.data.iloc[m['index'] : m['index'] + window_size]['Close'].values
                norm_seg = (segment - segment.min()) / (segment.max() - segment.min())
                res_seg = resample(norm_seg)
                
                axes[i+1, 0].plot(norm_seg, color='orange', linewidth=2)
                axes[i+1, 0].set_title(f"Match 1-{i+1} (Raw Segment)")
                axes[i+1, 1].plot(res_seg, color='orange', linestyle='--', linewidth=2)
                axes[i+1, 1].set_title(f"Match 1-{i+1} DNA (Resampled)")
            
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Detailed match report saved to {output_file}")

    def visualize_on_image(self, original_image_path, matches, window_size=100, output_file="highlighted_chart.png", label_prefix="Match", clusters=None):
        """Vision Point 3: Highlights found patterns directly on the original screenshot."""
        img = cv2.imread(original_image_path)
        if img is None: return
        
        h, w = img.shape[:2]
        overlay = img.copy()
        
        if clusters:
            # Multi-pattern clustering view
            import matplotlib.pyplot as plt
            colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
            # Convert mpl RGB (0-1) to OpenCV BGR (0-255)
            cv_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
            
            for c_idx, cluster in enumerate(clusters):
                color = cv_colors[c_idx]
                pat_id = cluster['pattern_id']
                
                for m_idx, match in enumerate(cluster['matches']):
                    start_x = int(match['index'])
                    end_x = int(start_x + window_size)
                    
                    # Draw a semi-transparent rectangle
                    cv2.rectangle(overlay, (start_x, 0), (end_x, h), color, -1)
                    
                    # Cross-reference label (e.g. "Pattern A-1")
                    display_label = f"Pattern {pat_id}-{m_idx+1}"
                    # Offset text based on cluster index so they don't overlap as easily
                    y_offset = 30 + (c_idx * 30) + (m_idx % 2) * 15
                    cv2.putText(img, display_label, (start_x + 5, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Single template fallback (original behavior)
            for i, m in enumerate(matches):
                start_x = int(m['index'])
                end_x = int(start_x + window_size)
                
                # Draw a semi-transparent rectangle over the match area
                cv2.rectangle(overlay, (start_x, 0), (end_x, h), (0, 165, 255), -1) # Orange
                # Cross-reference label (e.g. "Match 1" or "1a", "1b")
                display_label = f"{label_prefix} #{i+1}" if "Match" in label_prefix else f"{label_prefix}-{i+1}"
                cv2.putText(img, display_label, (start_x + 5, 30 + (i % 3) * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.imwrite(output_file, img)
        print(f"Highlighted chart saved to {output_file}")

    def visualize_pattern_match(self, pattern_image_path, match_info, output_file="reference_pattern.png", label="Pattern #1"):
        """Vision Point 3 modification: Highlights the matched portion directly on the pattern image."""
        img = cv2.imread(pattern_image_path)
        if img is None: return
        
        h, w = img.shape[:2]
        overlay = img.copy()
        
        # Currently, the whole template is used, but this supports partial template matching in the future
        start_x = int(match_info.get('pattern_start', 0))
        end_x = int(match_info.get('pattern_end', w))
        if end_x == 0: end_x = w # Fallback if not set
        
        # Draw a semi-transparent rectangle over the match area in the pattern
        cv2.rectangle(overlay, (start_x, 0), (end_x, h), (255, 0, 0), -1) # Blue
        
        # Add the main label and the specific match reference (e.g., "1-1", "1-2")
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 255), 3)
        
        # Add cross reference markers at the bottom
        markers = match_info.get('cross_references', [])
        for i, marker in enumerate(markers):
             cv2.putText(img, marker, (start_x + 5 + (i*60), h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.imwrite(output_file, img)
        print(f"Highlighted pattern saved to {output_file}")

    def visualize(self, matches, window_size=40, title="Pattern Match", output_file="match.png"):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        # Use simple indexing if Date is missing (for image charts)
        if 'Date' in self.data.columns:
            plt.plot(self.data['Date'], self.data['Close'], label='Price', alpha=0.5)
        else:
            plt.plot(self.data.index, self.data['Close'], label='Price', alpha=0.5)
        
        for i, m in enumerate(matches):
            seg = self.data.iloc[m['index'] : m['index'] + window_size]
            if 'Date' in self.data.columns:
                plt.axvspan(seg['Date'].iloc[0], seg['Date'].iloc[-1], color='orange', alpha=0.3)
                plt.text(seg['Date'].iloc[0], seg['Close'].iloc[0], f"#{i+1}", fontsize=10, fontweight='bold')
            else:
                plt.axvspan(seg.index[0], seg.index[-1], color='orange', alpha=0.3)
                plt.text(seg.index[0], seg['Close'].iloc[0], f"#{i+1}", fontsize=10, fontweight='bold')
            
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot current window (green)
        current = self.data.tail(window_size)
        if 'Date' in self.data.columns:
            plt.axvspan(current['Date'].iloc[0], current['Date'].iloc[-1], color='green', alpha=0.2, label='Current')
        else:
            plt.axvspan(current.index[0], current.index[-1], color='green', alpha=0.2, label='Current')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def _auto_crop_chart(self, img, white_mask):
        """Auto-detect and crop the main chart area from a full-screen screenshot.
        
        Strategy: Find the rectangular region with the most white pixels, excluding
        UI chrome on the edges (toolbars, sidebars, RSI panes etc.).
        """
        h, w = img.shape[:2]
        
        # Step 1: Find columns with high white pixel density (left/right bounds)
        col_density = np.sum(white_mask > 0, axis=0)  # sum per column
        
        # Remove very sparse columns (likely UI icons/text not part of the chart)
        threshold = np.max(col_density) * 0.05  # at least 5% of peak density
        active_cols = np.where(col_density > threshold)[0]
        
        if len(active_cols) < 50:
            return img, 0, 0  # Not enough to crop, return original
        
        x_start = int(active_cols[0])
        x_end = int(active_cols[-1])
        
        # Step 2: Find rows with high white pixel density (top/bottom bounds)
        row_density = np.sum(white_mask[:, x_start:x_end] > 0, axis=1)  # sum per row
        threshold_row = np.max(row_density) * 0.05
        active_rows = np.where(row_density > threshold_row)[0]
        
        if len(active_rows) < 50:
            return img, 0, 0  # Not enough rows, return original
        
        y_start = int(active_rows[0])
        y_end = int(active_rows[-1])
        
        # Step 3: Add small padding to be safe
        pad = 5
        x_start = max(0, x_start - pad)
        y_start = max(0, y_start - pad)
        x_end = min(w, x_end + pad)
        y_end = min(h, y_end + pad)
        
        print(f"Auto-cropped chart region: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}] from {w}x{h} image")
        cropped = img[y_start:y_end, x_start:x_end]
        return cropped, x_start, y_start

    def load_image_as_data(self, image_path):
        """Converts an image of a line chart into a numeric DataFrame."""
        print(f"Extracting price data from image: {image_path}...")
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image.")
            return False
        # 1. Color Masking (Isolating White Lines)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for white color
        # Low saturation, high value (brightness)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # If the mask is empty, fall back to grayscale thresholding targeting bright pixels
        if cv2.countNonZero(mask) < 100:
            print("Warning: Couldn't find enough white pixels, falling back to grayscale thresholding.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to keep only very bright pixels
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        else:
            print("Successfully isolated white lines using HSV masking.")
        
        # 1b. Auto-crop: detect and crop to just the main chart region
        img, crop_x, crop_y = self._auto_crop_chart(img, mask)
        
        # Re-run masking on the cropped image
        hsv_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_cropped, lower_white, upper_white)
        if cv2.countNonZero(mask) < 100:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
        # 2. Skeletonize to get single pixel line
        skeleton = skeletonize(mask > 0)
        
        # 3. Map x to average y
        # We want a 1D array where each index is an x-coordinate
        h, w = skeleton.shape
        sequence = []
        for x in range(w):
            pixels = np.where(skeleton[:, x])[0]
            if len(pixels) > 0:
                # Store the Y-coordinate. Since image (0,0) is top-left, 
                # we invert Y so higher up = higher price
                sequence.append(h - np.mean(pixels))
            else:
                # If gap, we'll interpolate later
                sequence.append(np.nan)
        
        # 4. Trim padding (empty space at start/end)
        s_pd = pd.Series(sequence)
        first_idx = s_pd.first_valid_index()
        last_idx = s_pd.last_valid_index()
        if first_idx is not None and last_idx is not None:
            s_pd = s_pd.iloc[first_idx : last_idx + 1]
            print(f"Trimmed drawing padding: {first_idx} to {last_idx}")
        
        # 5. Interpolate Gaps
        s_pd = s_pd.interpolate(method='linear').ffill().bfill()
        
        # Create a DataFrame to mimic the normal structure
        self.data = pd.DataFrame({
            'Close': s_pd.values
        })
        print(f"Extracted {len(self.data)} data points from drawing.")
        return True
