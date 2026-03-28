#!/usr/bin/env python3
"""
Evaluate multiple checkpoints (epoch-*.model) and write metrics vs epoch.

This is designed to support early stopping / model selection using segmental metrics
(F1@k, edit score) from `eval_val_segmental.py`.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_epochs(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _find_epochs_from_dir(model_dir: Path) -> List[int]:
    epochs = []
    for p in model_dir.glob("epoch-*.model"):
        try:
            ep = int(p.stem.split("-")[1])
            epochs.append(ep)
        except Exception:
            continue
    return sorted(set(epochs))


def _select_metric(row: Dict, select: str) -> float:
    select = select.strip().lower()
    if select == "micro_acc":
        return float(row.get("micro_acc", 0.0))
    if select == "macrof1_all":
        return float(row.get("macroF1_all", 0.0))
    if select == "macrof1_no_bg":
        return float(row.get("macroF1_no_bg", 0.0))
    if select.startswith("f1@"):
        k = select.split("@", 1)[1]
        return float(row.get("segmental", {}).get("f1_at", {}).get(str(k).zfill(2), 0.0))
    if select == "edit":
        return float(row.get("edit_score_mean", 0.0))
    raise ValueError(f"Unknown --select metric: {select}")


def main():
    ap = argparse.ArgumentParser(description="Sweep epochs and evaluate segmental metrics")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--xdir", required=True)
    ap.add_argument("--ydir", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--exp_dir", required=True, help="Experiment directory (used for default model_dir)")
    ap.add_argument("--model_dir", default=None, help="Directory containing epoch-*.model (default: exp_dir)")
    ap.add_argument("--outdir", required=True, help="Where to write sweep outputs")
    ap.add_argument("--split_name", default="val", choices=["train", "val", "test"])
    ap.add_argument("--causal", action="store_true", help="Use causal MS-TCN2")
    ap.add_argument("--epochs", default="", help="Comma-separated epochs to evaluate (e.g. '2,4,6,8')")
    ap.add_argument("--every", type=int, default=0, help="If >0, evaluate every N epochs up to --max_epoch")
    ap.add_argument("--max_epoch", type=int, default=0, help="Used with --every")
    ap.add_argument("--select", default="f1@50", help="Selection metric: f1@50 (default), micro_acc, macrof1_no_bg, edit, ...")

    # Forwarded to eval script
    ap.add_argument("--num_layers_PG", type=int, default=10)
    ap.add_argument("--num_layers_R", type=int, default=10)
    ap.add_argument("--num_R", type=int, default=3)
    ap.add_argument("--num_f_maps", type=int, default=64)
    ap.add_argument("--background_id", type=int, default=0)
    ap.add_argument("--include_background_in_segmental", action="store_true")
    ap.add_argument("--ignore_label", type=int, default=-100)
    ap.add_argument("--thresholds", default="0.10,0.25,0.50,0.75")

    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path(args.exp_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    epochs = _parse_epochs(args.epochs)
    if args.every and args.max_epoch:
        epochs = list(range(args.every, args.max_epoch + 1, args.every))
    if not epochs:
        epochs = _find_epochs_from_dir(model_dir)
    if not epochs:
        raise SystemExit(f"No epochs found in {model_dir} (expected epoch-*.model).")

    eval_script = Path(__file__).parent / "eval_val_segmental.py"
    if not eval_script.exists():
        raise SystemExit(f"Missing eval script: {eval_script}")

    rows: List[Dict] = []
    best: Optional[Tuple[int, float]] = None

    for ep in epochs:
        weights = model_dir / f"epoch-{ep}.model"
        if not weights.exists():
            print(f"[skip] missing {weights}")
            continue

        epoch_out = outdir / f"epoch{ep:03d}"
        epoch_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(eval_script),
            "--splits",
            str(args.splits),
            "--xdir",
            str(args.xdir),
            "--ydir",
            str(args.ydir),
            "--exp_dir",
            str(args.exp_dir),
            "--class_index",
            str(args.class_index),
            "--weights",
            str(weights),
            "--outdir",
            str(epoch_out),
            "--split_name",
            str(args.split_name),
            "--num_layers_PG",
            str(args.num_layers_PG),
            "--num_layers_R",
            str(args.num_layers_R),
            "--num_R",
            str(args.num_R),
            "--num_f_maps",
            str(args.num_f_maps),
            "--background_id",
            str(args.background_id),
            "--ignore_label",
            str(args.ignore_label),
            "--thresholds",
            str(args.thresholds),
        ]
        if args.causal:
            cmd.append("--causal")
        if args.include_background_in_segmental:
            cmd.append("--include_background_in_segmental")

        # Set up environment with PYTHONPATH to include project root
        import os
        env = os.environ.copy()
        project_root = str(Path(__file__).parent.parent.parent.parent)
        scripts_dir = str(Path(__file__).parent.parent)  # scripts/ directory
        env["PYTHONPATH"] = f"{scripts_dir}:{project_root}:{env.get('PYTHONPATH', '')}"
        
        # Run from scripts directory so causal_model import works
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=scripts_dir)
        if res.returncode != 0:
            print(f"[error] epoch {ep}:\n{res.stderr}")
            continue
        print(res.stdout.strip())

        summary_path = epoch_out / "summary.json"
        if not summary_path.exists():
            print(f"[error] missing summary.json for epoch {ep} at {summary_path}")
            continue

        row = json.loads(summary_path.read_text())
        row["_epoch"] = int(ep)
        rows.append(row)

        score = _select_metric(row, args.select)
        if best is None or score > best[1]:
            best = (int(ep), float(score))

    if not rows:
        raise SystemExit("No successful evaluations.")

    # Write JSONL (full summaries)
    with open(outdir / "metrics_vs_epoch.jsonl", "w") as f:
        for r in sorted(rows, key=lambda x: x["_epoch"]):
            f.write(json.dumps(r) + "\n")

    # Write CSV (compact)
    csv_path = outdir / "metrics_vs_epoch.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "micro_acc",
                "macroF1_all",
                "macroF1_no_bg",
                "F1@10",
                "F1@25",
                "F1@50",
                "F1@75",
                "edit_score_mean",
            ]
        )
        for r in sorted(rows, key=lambda x: x["_epoch"]):
            f1_at = r.get("segmental", {}).get("f1_at", {})
            w.writerow(
                [
                    r["_epoch"],
                    r.get("micro_acc", 0.0),
                    r.get("macroF1_all", 0.0),
                    r.get("macroF1_no_bg", 0.0),
                    f1_at.get("10", 0.0),
                    f1_at.get("25", 0.0),
                    f1_at.get("50", 0.0),
                    f1_at.get("75", 0.0),
                    r.get("edit_score_mean", 0.0),
                ]
            )

    if best is not None:
        (outdir / "best_epoch.txt").write_text(f"best_epoch={best[0]}\nmetric={args.select}\nscore={best[1]:.6f}\n")
        print(f"Best epoch: {best[0]} ({args.select}={best[1]:.6f})")


if __name__ == "__main__":
    main()


