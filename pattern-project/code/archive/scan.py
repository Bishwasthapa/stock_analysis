import argparse
import os
import sys
from legacy/core.engine import PatternEngine

def main():
    parser = argparse.ArgumentParser(description="Stock Pattern Scanner")
    parser.add_argument("--symbol", type=str, help="Stock symbol (e.g., NICA, BTC-USD)")
    parser.add_argument("--market", type=str, choices=["nepal", "international"], default="nepal", help="Market type")
    parser.add_argument(
        "--pattern",
        type=str,
        default="spring",
        help="Pattern name to match (see legacy/core/engine.py PATTERNS)",
    )
    parser.add_argument("--window", type=int, default=40, help="Analysis window size")
    parser.add_argument("--self", action="store_true", help="Run self-similarity discovery on data or visual-self on images")
    parser.add_argument("--visual_self", action="store_true", help="Run visual self-discovery specifically for points in the image")
    parser.add_argument("--ema", action="store_true", help="Run EMA crossover analysis")
    parser.add_argument("--chart_image", type=str, help="Path to a screenshot/drawing of a line chart")
    parser.add_argument("--pattern_image", type=str, help="Path to a screenshot/drawing of a pattern")
    parser.add_argument("--inout", action="store_true", help="Run In-Out Pattern Scanner on chart image")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback period for swing point detection")

    args = parser.parse_args()

    engine = PatternEngine()

    # Load data source
    success = False
    results_dir = "generic"
    
    if args.chart_image:
        success = engine.load_image_as_data(args.chart_image)
        # Vision 1: Dedicated visual_analysis sub-folder
        results_dir = os.path.join(os.path.dirname(args.chart_image), "generic", "visual_analysis")
    elif args.symbol:
        stock_dir = os.path.join("stocks", args.market, args.symbol)
        results_dir = os.path.join(stock_dir, "generic")
        if args.market == "nepal":
            success = engine.load_nepse_data(args.symbol)
        else:
            success = engine.load_intl_data(args.symbol)
    else:
        print("Error: Provide either --symbol or --chart_image")
        return

    if not success: return
    os.makedirs(results_dir, exist_ok=True)
    engine.calculate_indicators()

    print(f"--- Vision Analysis Starting ---")

    # 1. Structural Match (Image-to-Image OR Label-to-Image)
    target_template = None
    pattern_id = "Match" # Default prefix
    ref_path = None
    
    # Specific output directory for Task 1: Pattern vs Chart comparison
    task1_dir = os.path.join(results_dir, "ema_pattern_matches")
    
    if args.pattern_image:
        os.makedirs(task1_dir, exist_ok=True)
        temp_engine = PatternEngine()
        if temp_engine.load_image_as_data(args.pattern_image):
            target_template = temp_engine.data['Close'].values
            print(f"Using custom pattern from {args.pattern_image}")
            pattern_id = "1"
            ref_path = os.path.join(task1_dir, "reference_pattern.png")

    print(f"Finding matches...")
    struct_matches = engine.find_structural_matches(pattern_name=args.pattern, template=target_template, window_size=args.window)
    
    # Update the reference pattern image with the specific cross references if we have them
    if ref_path and target_template is not None:
        cross_refs = [f"{pattern_id}-{i+1}" for i in range(len(struct_matches))]
        match_info = {
            'pattern_start': 0,
            'pattern_end': len(target_template),
            'cross_references': cross_refs
        }
        engine.visualize_pattern_match(args.pattern_image, match_info, output_file=ref_path, label=f"Pattern #{pattern_id}")
    
    # Highlight on original image if provided (Vision Point 3 + Task Refinement)
    if args.chart_image:
        os.makedirs(task1_dir, exist_ok=True)
        engine.visualize_on_image(args.chart_image, struct_matches, window_size=args.window,
                                  output_file=os.path.join(task1_dir, "highlighted_chart.png"),
                                  label_prefix=f"Pattern {pattern_id}")
        
        # New: Vision Task 1 - Side-by-side comparison DNA
        if target_template is not None:
            engine.create_match_report(target_template, struct_matches, window_size=args.window,
                                       output_file=os.path.join(task1_dir, "pattern_comparison_report.png"))

    # 2. Self-Discovery (Data-based OR Visual-based)
    if args.self or args.visual_self:
        print("Running Self-Pattern Discovery (echoes)...")
        if args.chart_image:
            # Task 2 Refinement: Find multiple unique pattern clusters
            task2_dir = os.path.join(results_dir, "self_patterns")
            os.makedirs(task2_dir, exist_ok=True)
            
            clusters = engine.find_all_self_patterns(window_size=args.window, num_patterns=2, matches_per_pattern=2)
            
            if clusters:
                # Highlight the found matches on the image using the new cluster-aware visualization
                engine.visualize_on_image(
                    args.chart_image, 
                    matches=[], # Matches are handled inside clusters now
                    window_size=args.window,
                    output_file=os.path.join(task2_dir, "self_patterns_highlighted.png"),
                    clusters=clusters
                )
                
                # Generate the detailed Comparison Report for all clusters
                engine.create_match_report(
                    template=None, # Multiple templates handled inside clusters
                    matches=[], 
                    window_size=args.window,
                    output_file=os.path.join(task2_dir, "self_patterns_report.png"),
                    clusters=clusters
                )
            else:
                print("No distinct repeating patterns were found in the chart.")
        else:
            # Standard data-based self-echoes
            self_repeats = engine.find_self_repeats(window_size=args.window)
            engine.visualize(self_repeats, window_size=args.window, 
                             title=f"Data Self-Comparison", 
                             output_file=os.path.join(results_dir, "self_echoes.png"))

    # 3. In-Out Pattern Scanner
    if args.inout:
        if not args.chart_image:
            print("Error: --inout requires --chart_image")
        else:
            print("Running In-Out Pattern Scanner...")
            inout_dir = os.path.join(results_dir, "inout_pattern_scan")
            os.makedirs(inout_dir, exist_ok=True)
            
            swing_points = engine.find_swing_points(lookback=args.lookback)
            print(f"Found {len(swing_points)} swing points.")
            
            valid_patterns, out_pattern_indices = engine.run_inout_scanner(swing_points)
            
            engine.draw_inout_results(
                args.chart_image,
                swing_points,
                valid_patterns,
                out_pattern_indices,
                output_file=os.path.join(inout_dir, "inout_scan_result.png")
            )

    # 4. Optional: EMA Analysis
    if args.ema:
        print("Running EMA Setup Similarity...")
        ema_matches = engine.find_ema_repeats(window_size=args.window)
        engine.visualize(ema_matches, window_size=args.window, 
                         title=f"{args.symbol} EMA Layout Repeats", 
                         output_file=os.path.join(results_dir, "ema_repeats.png"))

    print(f"Done! Results saved in {results_dir}")

if __name__ == "__main__":
    main()
