from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LabelResult:
    """Standardized labeling result for a swing point."""
    cls: str            # 'IN' or 'OUT'
    role: int           # 0, 1, 2, or 3
    trend_type: str     # 'UPTREND' or 'DOWNTREND'
    point_label: str    # 'IN_UP', 'IN_DOWN', 'OUT_UP', 'OUT_DOWN'

class PatternLabeler:
    """
    Service to classify swing points into IN/OUT directional labels based on 4-point patterns.
    
    Logic:
    - A valid trend is a 4-point window: 0-1-2-3.
    - Points 1, 2, and 3 are ALWAYS 'IN'.
    - Point 0 is 'OUT' UNLESS it also serves as a point 1, 2, or 3 for another overlapping valid pattern.
    """
    
    @staticmethod
    def labels_from_sequences(swings_count: int, 
                             sequences: List[Tuple[int, List[int], str]]) -> Dict[int, LabelResult]:
        """
        Processes a list of detected sequences and returns a mapping of swing index to its best label.
        
        Args:
            swings_count: Total number of swing points.
            sequences: List of (start_idx, seq_indices, trend_type) tuples.
        """
        label_map: Dict[int, LabelResult] = {}
        
        # 1. Map each swing to its appearances in different patterns
        # swing_index -> list of (pattern_index, role_within_pattern)
        swing_membership: Dict[int, List[Tuple[int, int]]] = {}
        pattern_trends: List[str] = []
        
        for p_idx, (start_idx, seq_indices, trend_type) in enumerate(sequences):
            pattern_trends.append(trend_type)
            for role, s_idx in enumerate(seq_indices):
                swing_membership.setdefault(s_idx, []).append((p_idx, role))
        
        # 2. Determine for each pattern whether its START (role 0) is IN or OUT
        # A start is IN if that same point serves as role 1, 2, or 3 in ANY OTHER pattern.
        pattern_start_is_in: Dict[int, bool] = {}
        for p_idx, (start_idx, seq_indices, trend_type) in enumerate(sequences):
            s0_index = seq_indices[0]
            others = swing_membership.get(s0_index, [])
            pattern_start_is_in[p_idx] = any(role in (1, 2, 3) and other_p != p_idx 
                                           for other_p, role in others)
            
        # 3. Assign labels to each swing
        for s_idx, memberships in swing_membership.items():
            if not memberships:
                continue
                
            # Decision Tree for the "Best" label for a point:
            # 1. If it acts as role 0 in any pattern, we prioritize that pattern for labelling it as OUT/IN.
            role_0_entries = [m for m in memberships if m[1] == 0]
            if role_0_entries:
                # Use the last detected pattern where it is role 0
                chosen_p, role = role_0_entries[-1]
            else:
                # Not a role 0? Then it must be a role 1, 2, or 3 (always IN).
                # Prioritize a pattern that makes it IN if possible, or just the first one.
                chosen_p, role = memberships[0]
            
            trend_type = pattern_trends[chosen_p]
            is_in = True
            
            if role == 0:
                is_in = pattern_start_is_in.get(chosen_p, False)
            
            # Construct the point_label string
            cls = "IN" if is_in else "OUT"
            suffix = "UP" if trend_type == "UPTREND" else "DOWN"
            point_label = f"{cls}_{suffix}"
            
            label_map[s_idx] = LabelResult(
                cls=cls,
                role=role,
                trend_type=trend_type,
                point_label=point_label
            )
            
        return label_map
