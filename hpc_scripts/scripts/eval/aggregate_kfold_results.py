#!/usr/bin/env python3
"""
Aggregate K-fold cross-validation results.

This script:
1. Runs evaluation on each fold's validation set
2. Aggregates metrics across all folds
3. Produces a summary report with mean ± std

Usage:
    python aggregate_kfold_results.py \
        --kfold_dir /path/to/kfold_mydata \
        --exp_base_dir /work3/s204157/experiments/mydata_kfold \
        --xdir /path/to/features \
        --ydir /path/to/labels \
        --class_index /path/to/class_index.json \
        --outdir /path/to/eval_results \
        --epoch 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_summary(fold_outdir: Path) -> Dict:
    """
    Parse evaluation outputs from either:
    - summary.json (preferred; produced by eval_val_segmental.py)
    - summary.txt (legacy; produced by eval_val_timelines_plus.py)
    """
    json_path = fold_outdir / "summary.json"
    txt_path = fold_outdir / "summary.txt"

    if json_path.exists():
        obj = json.loads(json_path.read_text())
        # normalize into a stable schema for aggregation
        return {
            "micro_acc": float(obj.get("micro_acc", 0.0)),
            "macroF1": float(obj.get("macroF1_all", obj.get("macroF1", 0.0))),
            "macroF1_no_bg": float(obj.get("macroF1_no_bg", 0.0)),
            "f1_at": obj.get("segmental", {}).get("f1_at", {}),
            "edit_score_mean": float(obj.get("edit_score_mean", 0.0)),
            "per_class": obj.get("per_class", {}),  # usually absent; keep for compatibility
        }

    # legacy parsing
    results = {"micro_acc": 0.0, "macroF1": 0.0, "per_class": {}}
    if not txt_path.exists():
        return results

    with open(txt_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("micro_acc="):
            results["micro_acc"] = float(line.split("=")[1])
        elif line.startswith("macroF1="):
            results["macroF1"] = float(line.split("=")[1])
        elif line[:1].isdigit():
            # Per-class line: "00 none              0.8169 0.8891 0.8515"
            parts = line.split()
            if len(parts) >= 5:
                class_name = parts[1]
                prec = float(parts[2])
                rec = float(parts[3])
                f1 = float(parts[4])
                results["per_class"][class_name] = {"precision": prec, "recall": rec, "f1": f1}

    return results


def main():
    ap = argparse.ArgumentParser(description="Aggregate K-fold CV results")
    ap.add_argument("--kfold_dir", required=True, 
                    help="Directory containing splits_fold*.json files")
    ap.add_argument("--exp_base_dir", required=True,
                    help="Base directory for experiment outputs (contains ms_tcn_mydata_i3d_fold*)")
    ap.add_argument("--xdir", required=True, help="Features directory")
    ap.add_argument("--ydir", required=True, help="Labels directory")
    ap.add_argument("--class_index", required=True, help="Path to class_index.json")
    ap.add_argument("--outdir", required=True, help="Output directory for aggregated results")
    ap.add_argument("--epoch", type=int, default=100, help="Epoch to evaluate (default: 100)")
    ap.add_argument("--num_layers_PG", type=int, default=10)
    ap.add_argument("--num_layers_R", type=int, default=10)
    ap.add_argument("--num_R", type=int, default=6)
    ap.add_argument("--num_f_maps", type=int, default=64)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--run_eval", action="store_true", 
                    help="Run evaluation for each fold (otherwise just aggregate existing results)")
    ap.add_argument(
        "--segmental",
        action="store_true",
        help="Use segmental eval (F1@k, edit) via eval_val_segmental.py and aggregate those metrics.",
    )
    args = ap.parse_args()

    kfold_dir = Path(args.kfold_dir)
    exp_base_dir = Path(args.exp_base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load K-fold metadata
    meta_path = kfold_dir / "kfold_meta.json"
    if not meta_path.exists():
        print(f"ERROR: K-fold metadata not found: {meta_path}")
        sys.exit(1)
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    K = meta["K"]
    print(f"Aggregating results for {K}-fold cross-validation")
    print(f"Type: {meta['type']}")
    print()

    # Find the eval script
    eval_script = Path(__file__).parent / ("eval_val_segmental.py" if args.segmental else "eval_val_timelines_plus.py")
    if not eval_script.exists():
        print(f"WARNING: Eval script not found at {eval_script}")
        print("Will only aggregate existing results")
        args.run_eval = False

    # Collect results for each fold
    all_results = []
    
    for fold_idx in range(K):
        fold_exp_dir = exp_base_dir / f"ms_tcn_mydata_i3d_fold{fold_idx}"
        fold_splits = kfold_dir / f"splits_fold{fold_idx}.json"
        fold_outdir = outdir / f"fold{fold_idx}"
        fold_outdir.mkdir(parents=True, exist_ok=True)
        
        # Get validation video name
        with open(fold_splits) as f:
            splits = json.load(f)
        val_video = Path(splits["val"][0]).stem if splits["val"] else "unknown"
        
        print(f"Fold {fold_idx}: val={val_video}")
        
        # Check if model exists
        model_path = fold_exp_dir / f"epoch-{args.epoch}.model"
        if not model_path.exists():
            print(f"  WARNING: Model not found: {model_path}")
            print(f"  Skipping fold {fold_idx}")
            continue
        
        # eval_val_segmental writes summary.json (+ summary.txt), legacy writes summary.txt only
        summary_path = fold_outdir / ("summary.json" if args.segmental else "summary.txt")
        
        # Run evaluation if requested and summary doesn't exist
        if args.run_eval and not summary_path.exists():
            print(f"  Running evaluation...")
            import subprocess
            cmd = [
                sys.executable, str(eval_script),
                "--splits", str(fold_splits),
                "--xdir", args.xdir,
                "--ydir", args.ydir,
                "--exp_dir", str(fold_exp_dir),
                "--class_index", args.class_index,
                "--weights", str(model_path),
                "--num_layers_PG", str(args.num_layers_PG),
                "--num_layers_R", str(args.num_layers_R),
                "--num_R", str(args.num_R),
                "--num_f_maps", str(args.num_f_maps),
                "--outdir", str(fold_outdir),
                "--split_name", "val",
            ]
            if args.causal:
                cmd.append("--causal")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR running eval: {result.stderr}")
                continue
            print(f"  {result.stdout.strip()}")
        
        # Parse results
        if summary_path.exists():
            results = parse_summary(fold_outdir)
            results["fold_idx"] = fold_idx
            results["val_video"] = val_video
            all_results.append(results)
            msg = f"  micro_acc={results['micro_acc']:.4f}, macroF1={results['macroF1']:.4f}"
            if args.segmental:
                f1_50 = float(results.get("f1_at", {}).get("50", 0.0))
                msg += f", F1@50={f1_50:.4f}, edit={float(results.get('edit_score_mean', 0.0)):.2f}"
            print(msg)
        else:
            print(f"  WARNING: Summary not found: {summary_path}")
    
    print()
    
    if not all_results:
        print("ERROR: No results to aggregate")
        sys.exit(1)
    
    # Aggregate results
    print("=" * 70)
    print(f"K-Fold Cross-Validation Results ({len(all_results)}/{K} folds)")
    print("=" * 70)
    
    micro_accs = [r["micro_acc"] for r in all_results]
    macro_f1s = [r["macroF1"] for r in all_results]
    macro_f1s_no_bg = [r.get("macroF1_no_bg", 0.0) for r in all_results if "macroF1_no_bg" in r]
    
    print(f"\nOverall Metrics (mean ± std):")
    print(f"  micro-acc:  {np.mean(micro_accs):.4f} ± {np.std(micro_accs):.4f}  (range: {min(micro_accs):.4f} - {max(micro_accs):.4f})")
    print(f"  macro-F1:   {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}  (range: {min(macro_f1s):.4f} - {max(macro_f1s):.4f})")
    if args.segmental and macro_f1s_no_bg:
        print(
            f"  macro-F1(no-bg): {np.mean(macro_f1s_no_bg):.4f} ± {np.std(macro_f1s_no_bg):.4f}  "
            f"(range: {min(macro_f1s_no_bg):.4f} - {max(macro_f1s_no_bg):.4f})"
        )
    
    # Per-fold breakdown
    print(f"\nPer-fold Results:")
    if args.segmental:
        print(f"  {'Fold':<6} {'Val Video':<30} {'micro-acc':<12} {'macro-F1':<12} {'F1@50':<12} {'Edit':<8}")
        print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    else:
        print(f"  {'Fold':<6} {'Val Video':<30} {'micro-acc':<12} {'macro-F1':<12}")
        print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*12}")
    for r in all_results:
        if args.segmental:
            f1_50 = float(r.get("f1_at", {}).get("50", 0.0))
            ed = float(r.get("edit_score_mean", 0.0))
            print(
                f"  {r['fold_idx']:<6} {r['val_video']:<30} {r['micro_acc']:<12.4f} {r['macroF1']:<12.4f} {f1_50:<12.4f} {ed:<8.2f}"
            )
        else:
            print(f"  {r['fold_idx']:<6} {r['val_video']:<30} {r['micro_acc']:<12.4f} {r['macroF1']:<12.4f}")
    
    per_class_summary = {}
    if all_results and all_results[0].get("per_class"):
        # Per-class metrics (only available for legacy eval)
        print(f"\nPer-class F1 Scores (mean ± std):")
        class_names = list(all_results[0]["per_class"].keys())
        for class_name in class_names:
            f1_scores = [r["per_class"].get(class_name, {}).get("f1", 0.0) for r in all_results]
            prec_scores = [r["per_class"].get(class_name, {}).get("precision", 0.0) for r in all_results]
            rec_scores = [r["per_class"].get(class_name, {}).get("recall", 0.0) for r in all_results]
            per_class_summary[class_name] = {
                "f1_mean": np.mean(f1_scores),
                "f1_std": np.std(f1_scores),
                "prec_mean": np.mean(prec_scores),
                "rec_mean": np.mean(rec_scores),
            }
            print(
                f"  {class_name:<20} F1={np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}  "
                f"(P={np.mean(prec_scores):.4f}, R={np.mean(rec_scores):.4f})"
            )
    
    print("=" * 70)
    
    # Save aggregated results
    aggregated = {
        "num_folds": len(all_results),
        "total_folds": K,
        "micro_acc": {
            "mean": float(np.mean(micro_accs)),
            "std": float(np.std(micro_accs)),
            "min": float(min(micro_accs)),
            "max": float(max(micro_accs)),
            "values": micro_accs
        },
        "macroF1": {
            "mean": float(np.mean(macro_f1s)),
            "std": float(np.std(macro_f1s)),
            "min": float(min(macro_f1s)),
            "max": float(max(macro_f1s)),
            "values": macro_f1s
        },
        "macroF1_no_bg": {
            "mean": float(np.mean(macro_f1s_no_bg)) if macro_f1s_no_bg else None,
            "std": float(np.std(macro_f1s_no_bg)) if macro_f1s_no_bg else None,
            "min": float(min(macro_f1s_no_bg)) if macro_f1s_no_bg else None,
            "max": float(max(macro_f1s_no_bg)) if macro_f1s_no_bg else None,
            "values": macro_f1s_no_bg if macro_f1s_no_bg else None,
        },
        "per_class": per_class_summary,
        "per_fold": all_results
    }

    if args.segmental:
        # Aggregate F1@k and edit score if present
        ks = ["10", "25", "50", "75"]
        f1k = {k: [float(r.get("f1_at", {}).get(k, 0.0)) for r in all_results] for k in ks}
        aggregated["segmental"] = {
            "f1_at": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "min": float(min(v)),
                    "max": float(max(v)),
                    "values": v,
                }
                for k, v in f1k.items()
            },
            "edit_score_mean": {
                "mean": float(np.mean([float(r.get("edit_score_mean", 0.0)) for r in all_results])),
                "std": float(np.std([float(r.get("edit_score_mean", 0.0)) for r in all_results])),
            },
        }
    
    agg_path = outdir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved: {agg_path}")
    
    # Save summary text
    summary_path = outdir / "kfold_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"K-Fold Cross-Validation Results ({len(all_results)}/{K} folds)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"micro_acc: {np.mean(micro_accs):.4f} ± {np.std(micro_accs):.4f}\n")
        f.write(f"macroF1:   {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}\n\n")
        if args.segmental:
            ks = ["10", "25", "50", "75"]
            for k in ks:
                vals = [float(r.get("f1_at", {}).get(k, 0.0)) for r in all_results]
                f.write(f"F1@{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")
            edits = [float(r.get("edit_score_mean", 0.0)) for r in all_results]
            f.write(f"Edit: {np.mean(edits):.2f} ± {np.std(edits):.2f}\n")
        elif per_class_summary:
            f.write(f"Per-class F1 (mean ± std):\n")
            for class_name, stats in per_class_summary.items():
                f.write(f"  {class_name}: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}\n")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

