#!/usr/bin/env python3
"""
Run FSM validation on K-fold cross-validation results

This script:
1. Loads predictions from eval_with_predictions.py for each fold
2. Runs FSM tip hygiene validation on predictions vs ground truth
3. Aggregates statistics across all folds
4. Saves detailed per-video and aggregate reports

Usage:
    python run_fsm_validation_kfold.py \
        --eval_base_dir /path/to/eval_data/mydata_kfold_fixed/exp1_baseline \
        --class_index /path/to/class_index_7class.json \
        --output_dir /path/to/fsm_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to path for imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from fsm_tip_hygiene_validator import (
    TipHygieneFSM,
    compare_predictions_vs_ground_truth,
    validate_ground_truth_sequence,
)


def load_class_names(class_index_path: str) -> List[str]:
    """Load action class names from class_index.json"""
    obj = json.loads(Path(class_index_path).read_text())
    if isinstance(obj, dict):
        inv = {v: k for k, v in obj.items()}
        names = [inv[i] if i in inv else f"c{i}" for i in range(len(inv))]
    else:
        names = obj
    return names


def process_fold(
    fold_idx: int,
    predictions_dir: Path,
    class_names: List[str],
    output_dir: Path
) -> Dict:
    """
    Process FSM validation for a single fold
    
    Args:
        fold_idx: Fold index (0-based)
        predictions_dir: Directory containing {stem}_pred.npy and {stem}_gt.npy
        class_names: List of action class names
        output_dir: Output directory for this fold's results
    
    Returns:
        Dictionary with fold statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all prediction files
    pred_files = sorted(predictions_dir.glob("*_pred.npy"))
    
    if not pred_files:
        print(f"  WARNING: No prediction files found in {predictions_dir}")
        return None
    
    fold_results = {
        "fold_idx": fold_idx,
        "videos": [],
        "aggregate": {
            "total_videos": 0,
            "total_alerts_pred": 0,
            "total_alerts_gt": 0,
            "critical_alerts_pred": 0,
            "critical_alerts_gt": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
    }
    
    for pred_file in pred_files:
        stem = pred_file.stem.replace("_pred", "")
        gt_file = predictions_dir / f"{stem}_gt.npy"
        
        if not gt_file.exists():
            print(f"    WARNING: Missing ground truth for {stem}")
            continue
        
        # Load predictions and ground truth
        predictions = np.load(pred_file)
        ground_truth = np.load(gt_file)
        
        # Validate that sequences are same length
        T = min(len(predictions), len(ground_truth))
        predictions = predictions[:T]
        ground_truth = ground_truth[:T]
        
        # Run FSM validation
        comparison = compare_predictions_vs_ground_truth(
            predictions=predictions,
            ground_truth=ground_truth,
            action_names=class_names,
            context_window=10,
            num_classes=len(class_names)
        )
        
        # Store video results
        video_result = {
            "stem": stem,
            "T": int(T),
            "predictions": comparison["predictions"],
            "ground_truth": comparison["ground_truth"],
            "metrics": comparison["metrics"]
        }
        fold_results["videos"].append(video_result)
        
        # Update aggregate statistics
        agg = fold_results["aggregate"]
        agg["total_videos"] += 1
        agg["total_alerts_pred"] += comparison["predictions"]["total_alerts"]
        agg["total_alerts_gt"] += comparison["ground_truth"]["total_alerts"]
        agg["critical_alerts_pred"] += comparison["predictions"]["critical_alerts"]
        agg["critical_alerts_gt"] += comparison["ground_truth"]["critical_alerts"]
        agg["true_positives"] += comparison["metrics"]["true_positives"]
        agg["false_positives"] += comparison["metrics"]["false_positives"]
        agg["false_negatives"] += comparison["metrics"]["false_negatives"]
        
        print(f"    ✓ {stem}: "
              f"pred_critical={comparison['predictions']['critical_alerts']}, "
              f"gt_critical={comparison['ground_truth']['critical_alerts']}, "
              f"TP={comparison['metrics']['true_positives']}, "
              f"FP={comparison['metrics']['false_positives']}, "
              f"FN={comparison['metrics']['false_negatives']}")
    
    # Calculate aggregate metrics
    agg = fold_results["aggregate"]
    tp = agg["true_positives"]
    fp = agg["false_positives"]
    fn = agg["false_negatives"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # 1.0 if no violations exist
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    agg["precision"] = precision
    agg["recall"] = recall
    agg["f1"] = f1
    
    # Save fold results
    with open(output_dir / "fsm_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    
    return fold_results


def aggregate_across_folds(all_fold_results: List[Dict], output_dir: Path):
    """
    Aggregate FSM validation results across all folds
    
    Args:
        all_fold_results: List of fold result dictionaries
        output_dir: Output directory for aggregate results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate statistics
    total_videos = sum(f["aggregate"]["total_videos"] for f in all_fold_results)
    total_alerts_pred = sum(f["aggregate"]["total_alerts_pred"] for f in all_fold_results)
    total_alerts_gt = sum(f["aggregate"]["total_alerts_gt"] for f in all_fold_results)
    critical_alerts_pred = sum(f["aggregate"]["critical_alerts_pred"] for f in all_fold_results)
    critical_alerts_gt = sum(f["aggregate"]["critical_alerts_gt"] for f in all_fold_results)
    
    total_tp = sum(f["aggregate"]["true_positives"] for f in all_fold_results)
    total_fp = sum(f["aggregate"]["false_positives"] for f in all_fold_results)
    total_fn = sum(f["aggregate"]["false_negatives"] for f in all_fold_results)
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Per-fold metrics for mean/std calculation
    fold_precisions = [f["aggregate"]["precision"] for f in all_fold_results]
    fold_recalls = [f["aggregate"]["recall"] for f in all_fold_results]
    fold_f1s = [f["aggregate"]["f1"] for f in all_fold_results]
    
    aggregate_report = {
        "num_folds": len(all_fold_results),
        "total_videos": total_videos,
        "overall": {
            "total_alerts_pred": total_alerts_pred,
            "total_alerts_gt": total_alerts_gt,
            "critical_alerts_pred": critical_alerts_pred,
            "critical_alerts_gt": critical_alerts_gt,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_fold_stats": {
            "precision": {
                "mean": float(np.mean(fold_precisions)),
                "std": float(np.std(fold_precisions)),
                "values": fold_precisions,
            },
            "recall": {
                "mean": float(np.mean(fold_recalls)),
                "std": float(np.std(fold_recalls)),
                "values": fold_recalls,
            },
            "f1": {
                "mean": float(np.mean(fold_f1s)),
                "std": float(np.std(fold_f1s)),
                "values": fold_f1s,
            },
        },
        "folds": [
            {
                "fold_idx": f["fold_idx"],
                "videos": f["aggregate"]["total_videos"],
                "precision": f["aggregate"]["precision"],
                "recall": f["aggregate"]["recall"],
                "f1": f["aggregate"]["f1"],
            }
            for f in all_fold_results
        ]
    }
    
    # Save aggregate report
    with open(output_dir / "fsm_aggregate.json", "w") as f:
        json.dump(aggregate_report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FSM VALIDATION AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Total videos: {total_videos}")
    print(f"Total folds: {len(all_fold_results)}")
    print()
    print("ALERTS:")
    print(f"  Predicted (total):    {total_alerts_pred}")
    print(f"  Ground truth (total): {total_alerts_gt}")
    print(f"  Predicted (critical): {critical_alerts_pred}")
    print(f"  Ground truth (critical): {critical_alerts_gt}")
    print()
    print("DETECTION METRICS (for critical violations):")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print()
    print("PER-FOLD STATISTICS:")
    print(f"  Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"  Recall:    {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"  F1 Score:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print("=" * 80)
    
    return aggregate_report


def main():
    ap = argparse.ArgumentParser(description="Run FSM validation on K-fold results")
    ap.add_argument(
        "--eval_base_dir",
        required=True,
        help="Base evaluation directory (contains fold0/, fold1/, etc.)"
    )
    ap.add_argument(
        "--class_index",
        required=True,
        help="Path to class_index.json"
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for FSM validation results"
    )
    ap.add_argument(
        "--num_folds",
        type=int,
        default=None,
        help="Number of folds (default: auto-detect)"
    )
    
    args = ap.parse_args()
    
    eval_base_dir = Path(args.eval_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    class_names = load_class_names(args.class_index)
    print(f"Loaded {len(class_names)} action classes: {class_names}")
    print()
    
    # Auto-detect number of folds if not specified
    if args.num_folds is None:
        fold_dirs = sorted(eval_base_dir.glob("fold*"))
        num_folds = len(fold_dirs)
        print(f"Auto-detected {num_folds} folds")
    else:
        num_folds = args.num_folds
    
    # Process each fold
    all_fold_results = []
    
    for fold_idx in range(num_folds):
        fold_dir = eval_base_dir / f"fold{fold_idx}"
        predictions_dir = fold_dir / "predictions"
        fold_output_dir = output_dir / f"fold{fold_idx}"
        
        print(f"Processing Fold {fold_idx}...")
        print(f"  Predictions dir: {predictions_dir}")
        
        if not predictions_dir.exists():
            print(f"  ERROR: Predictions directory not found: {predictions_dir}")
            print(f"  Skipping fold {fold_idx}")
            print()
            continue
        
        fold_results = process_fold(
            fold_idx=fold_idx,
            predictions_dir=predictions_dir,
            class_names=class_names,
            output_dir=fold_output_dir
        )
        
        if fold_results:
            all_fold_results.append(fold_results)
            print(f"  ✓ Fold {fold_idx} complete: "
                  f"precision={fold_results['aggregate']['precision']:.4f}, "
                  f"recall={fold_results['aggregate']['recall']:.4f}, "
                  f"f1={fold_results['aggregate']['f1']:.4f}")
        print()
    
    if not all_fold_results:
        print("ERROR: No fold results to aggregate")
        sys.exit(1)
    
    # Aggregate results across folds
    aggregate_report = aggregate_across_folds(all_fold_results, output_dir)
    
    print(f"\n✓ FSM validation complete!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()




