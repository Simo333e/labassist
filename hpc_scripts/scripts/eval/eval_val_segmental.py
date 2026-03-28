#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# Ensure project root is importable so `hpc_scripts/MS-TCN` can be added below
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import local eval utilities without requiring `hpc_scripts` to be a Python package
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from segmentation_metrics import edit_score, segmental_f1_at_thresholds

# Action class colors (for 5-class PCR)
# Supports both old and new (v2) taxonomy names
# Keep consistent with dataset_statistics.py
ACTION_COLORS = {
    'none': '#CCCCCC',
    # Old names (v1)
    'Pipette pcr tube': '#E74C3C',
    'Aspirate reagent': '#3498DB', 
    # New names (v2)
    'Manipulate PCR tube': '#E74C3C',
    'Manipulate reagent': '#3498DB',
    # Unchanged
    'Attach pipette tip': '#2ECC71',
    'Eject pipette tip': '#F39C12',
}


def plot_timeline_with_legend(y: np.ndarray, pred: np.ndarray, names, outfile: str, title: str = ""):
    ncls = len(names)
    
    # Use consistent colors from ACTION_COLORS when available
    colors = []
    for name in names:
        if name in ACTION_COLORS:
            # Convert hex to RGB tuple
            hex_color = ACTION_COLORS[name]
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
            colors.append(rgb)
        else:
            # Fallback to tab20 for unknown classes
            base = matplotlib.colormaps.get_cmap("tab20")
            colors.append(base(len(colors) % 20))
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, ncls + 0.5, 1.0), cmap.N)

    def boundaries(arr):
        return np.where(arr[1:] != arr[:-1])[0] + 1

    fig = plt.figure(figsize=(14, 4), layout="constrained")
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 0.28], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[2, 0])

    ax0.imshow(y[None, :], aspect="auto", cmap=cmap, norm=norm)
    ax0.set_title("GT" + (f" – {title}" if title else ""))
    ax0.set_yticks([])

    ax1.imshow(pred[None, :], aspect="auto", cmap=cmap, norm=norm)
    ax1.set_title("Pred")
    ax1.set_yticks([])

    for ax, arr in ((ax0, y), (ax1, pred)):
        for b in boundaries(arr):
            ax.axvline(b, color="k", alpha=0.15, lw=0.75)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks(range(ncls))
    cb.set_ticklabels([str(n) for n in names])
    for lab in cax.get_xticklabels():
        lab.set_rotation(40)
        lab.set_ha("right")
        lab.set_fontsize(8)

    out = pathlib.Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)


def load_class_names(class_index_path: str) -> List[str]:
    obj = json.loads(pathlib.Path(class_index_path).read_text())
    if isinstance(obj, dict):
        inv = {v: k for k, v in obj.items()}
        names = [inv[i] if i in inv else f"c{i}" for i in range(len(inv))]
    else:
        names = obj
    return names


def load_stems(splits_json: str, split_name: str) -> List[str]:
    sp = json.loads(pathlib.Path(splits_json).read_text())
    if "train" not in sp:
        sp = next(iter(sp.values()))
    return [pathlib.Path(p).stem for p in sp.get(split_name, [])]


def macro_f1_from_cm(cm: np.ndarray, exclude_ids: Optional[List[int]] = None) -> float:
    cm = np.asarray(cm)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    if exclude_ids:
        mask = np.ones((cm.shape[0],), dtype=bool)
        for i in exclude_ids:
            if 0 <= i < mask.shape[0]:
                mask[i] = False
        f1 = f1[mask]
    return float(np.nanmean(f1)) if f1.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--xdir", required=True)
    ap.add_argument("--ydir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--num_layers_PG", type=int, default=10)
    ap.add_argument("--num_layers_R", type=int, default=10)
    ap.add_argument("--num_R", type=int, default=3)
    ap.add_argument("--num_f_maps", type=int, default=64)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--make_timelines", action="store_true")
    ap.add_argument("--split_name", choices=["train", "val", "test"], default="val")
    ap.add_argument("--causal", action="store_true", help="use causal MS-TCN2; default is non-causal")

    # Segmental metric options
    ap.add_argument("--ignore_label", type=int, default=-100, help="Label id to ignore (padding), default -100")
    ap.add_argument("--background_id", type=int, default=0, help="Background/'none' class id to exclude from segmental metrics")
    ap.add_argument(
        "--include_background_in_segmental",
        action="store_true",
        help="If set, include background in segmental metrics (default excludes it).",
    )
    ap.add_argument(
        "--thresholds",
        type=str,
        default="0.10,0.25,0.50,0.75",
        help="Comma-separated IoU thresholds for F1@k, e.g. '0.10,0.25,0.50,0.75'",
    )

    args = ap.parse_args()

    stems = load_stems(args.splits, args.split_name)
    names = load_class_names(args.class_index)
    ncls = len(names)

    # load model (MS-TCN2)
    MS_TCN_DIR = (pathlib.Path(__file__).resolve().parents[2] / "MS-TCN")
    if str(MS_TCN_DIR) not in sys.path:
        sys.path.insert(0, str(MS_TCN_DIR))
    if args.causal:
        from causal_model import MS_TCN2
    else:
        from model import MS_TCN2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer D from first available X
    D = None
    for s in stems:
        p = pathlib.Path(args.xdir) / f"{s}_x.npy"
        if p.exists():
            D = int(np.load(p, mmap_mode="r").shape[1])
            break
    assert D is not None, "Could not infer feature dim (no X found)."

    model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, D, ncls).to(device)
    sd = torch.load(pathlib.Path(args.weights), map_location=device)
    model.load_state_dict(sd)
    model.eval()

    outdir = pathlib.Path(args.outdir or (pathlib.Path(args.exp_dir) / f"eval_{args.split_name}"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "timelines").mkdir(parents=True, exist_ok=True)

    cm = np.zeros((ncls, ncls), np.int64)
    per_video = []

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    exclude_segment_labels = []
    if not args.include_background_in_segmental:
        exclude_segment_labels.append(int(args.background_id))

    # accumulate segmental counts by threshold across videos
    seg_counts: Dict[float, Dict[str, float]] = {
        thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for thr in thresholds
    }
    edit_scores: List[float] = []

    for stem in stems:
        xpath = pathlib.Path(args.xdir) / f"{stem}_x.npy"
        ypath = pathlib.Path(args.ydir) / f"{stem}_y.npy"
        if not xpath.exists() or not ypath.exists():
            continue
        X = np.load(xpath)
        y = np.load(ypath)
        T = min(len(X), len(y))
        X, y = X[:T], y[:T]

        xb = torch.from_numpy(X.T).float().unsqueeze(0).to(device)  # [1,D,T]
        with torch.no_grad():
            logits_bct = model(xb)[-1][0]  # [C,T]
            logits = logits_bct.transpose(0, 1).cpu().numpy()  # [T,C]
        pred = logits.argmax(-1).astype(int)

        # framewise metrics
        acc = float((pred == y).mean()) if T > 0 else 0.0
        for g, p in zip(y, pred):
            if 0 <= int(g) < ncls:
                cm[int(g), int(p)] += 1

        # segmental metrics
        seg = segmental_f1_at_thresholds(
            gt=y,
            pred=pred,
            thresholds=thresholds,
            ignore_labels=(args.ignore_label,),
            exclude_labels=exclude_segment_labels,
        )
        for thr, d in seg.items():
            seg_counts[thr]["tp"] += float(d["tp"])
            seg_counts[thr]["fp"] += float(d["fp"])
            seg_counts[thr]["fn"] += float(d["fn"])

        ed = edit_score(
            gt=y,
            pred=pred,
            ignore_labels=(args.ignore_label,),
            exclude_labels=exclude_segment_labels,
        )
        edit_scores.append(ed)

        per_video.append(
            {
                "stem": stem,
                "T": int(T),
                "frame_acc": float(acc),
                "edit_score": float(ed),
            }
        )

        if args.make_timelines:
            out_png = outdir / "timelines" / f"{stem}.png"
            plot_timeline_with_legend(y=y, pred=pred, names=names, outfile=str(out_png), title=stem)

    micro_acc = float(np.diag(cm).sum() / max(1, cm.sum()))
    macroF1_all = macro_f1_from_cm(cm, exclude_ids=None)
    macroF1_no_bg = macro_f1_from_cm(cm, exclude_ids=[int(args.background_id)])

    # finalize segmental metrics from aggregated TP/FP/FN
    f1_at: Dict[str, float] = {}
    seg_details: Dict[str, Dict[str, float]] = {}
    for thr in thresholds:
        tp = seg_counts[thr]["tp"]
        fp = seg_counts[thr]["fp"]
        fn = seg_counts[thr]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        key = f"{int(round(thr * 100)):02d}"
        f1_at[key] = float(f1)
        seg_details[key] = {
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }

    edit_mean = float(np.mean(edit_scores)) if edit_scores else 0.0

    print(
        f"{args.split_name.upper()} "
        f"micro-acc={micro_acc:.4f}  "
        f"macroF1(all)={macroF1_all:.4f}  "
        f"macroF1(no-bg)={macroF1_no_bg:.4f}  "
        + "  ".join([f"F1@{k}={v:.4f}" for k, v in sorted(f1_at.items())])
        + f"  Edit={edit_mean:.2f}"
    )

    # save metrics
    np.savetxt(outdir / f"confusion_{args.split_name}.csv", cm, fmt="%d", delimiter=",")
    summary = {
        "split_name": args.split_name,
        "micro_acc": float(micro_acc),
        "macroF1_all": float(macroF1_all),
        "macroF1_no_bg": float(macroF1_no_bg),
        "segmental": {
            "include_background": bool(args.include_background_in_segmental),
            "background_id": int(args.background_id),
            "ignore_label": int(args.ignore_label),
            "f1_at": f1_at,  # keys: "10","25","50","75"
            "details": seg_details,
        },
        "edit_score_mean": float(edit_mean),
        "num_videos": int(len(per_video)),
        "per_video": per_video,
        "weights": str(args.weights),
        "class_index": str(args.class_index),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # keep a minimal text summary for quick grepping
    with open(outdir / "summary.txt", "w") as f:
        f.write(f"micro_acc={micro_acc:.6f}\n")
        f.write(f"macroF1_all={macroF1_all:.6f}\n")
        f.write(f"macroF1_no_bg={macroF1_no_bg:.6f}\n")
        for k in sorted(f1_at.keys()):
            f.write(f"F1@{k}={f1_at[k]:.6f}\n")
        f.write(f"edit_score_mean={edit_mean:.6f}\n")


if __name__ == "__main__":
    main()


