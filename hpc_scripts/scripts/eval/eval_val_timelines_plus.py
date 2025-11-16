#!/usr/bin/env python3
import argparse, json, pathlib, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
from typing import List, Dict
import sys, pathlib
import numpy as np

# Ensure project root is importable so `hpc_scripts.*` works
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def plot_timeline_with_legend(y: np.ndarray, pred: np.ndarray, names, outfile: str, title: str = ""):
    ncls = len(names)
    base = matplotlib.colormaps.get_cmap("tab20")
    colors = [base(i % 20) for i in range(ncls)]
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
        lab.set_rotation(40); lab.set_ha("right"); lab.set_fontsize(8)

    out = pathlib.Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)
def load_class_names(class_index_path):
    obj = json.loads(pathlib.Path(class_index_path).read_text())
    if isinstance(obj, dict):
        inv = {v:k for k,v in obj.items()}
        names = [inv[i] if i in inv else f"c{i}" for i in range(len(inv))]
    else:
        names = obj
    return names
 
def load_stems(splits_json: str, split_name: str) -> List[str]:
    sp = json.loads(pathlib.Path(splits_json).read_text())
    if "train" not in sp: sp = next(iter(sp.values()))
    return [pathlib.Path(p).stem for p in sp.get(split_name, [])]

def load_order(names):
    # assumes 'none' is first or present; order by index in class_index
    return list(range(len(names)))

def monotone_decode(probs, order):
    curr = 0
    T, C = probs.shape
    path = np.zeros((T,), np.int32)
    for t in range(T):
        cand = [curr, min(curr+1, C-1)]
        path[t] = cand[int(np.argmax(probs[t, cand]))]
        curr = path[t]
    return path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--xdir", required=True)
    ap.add_argument("--ydir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--num_layers_PG", type=int, default=10)
    ap.add_argument("--num_layers_R",  type=int, default=10)
    ap.add_argument("--num_R",         type=int, default=3)
    ap.add_argument("--num_f_maps",    type=int, default=64)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--make_timelines", action="store_true")
    ap.add_argument("--split_name", choices=["val", "test"], default="val")
    ap.add_argument("--dilations", default="", help="unused for MS_TCN2 causal eval")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--causal", action="store_true", help="use causal MS-TCN2; default is non-causal")

    args = ap.parse_args()

    stems = load_stems(args.splits, args.split_name)
    names = load_class_names(args.class_index)
    ncls = len(names)

    # load model (causal MS-TCN2)
    import sys
    MS_TCN_DIR = (pathlib.Path(__file__).resolve().parents[2] / "hpc_scripts" / "MS-TCN")
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
        p = pathlib.Path(args.xdir)/f"{s}_x.npy"
        if p.exists():
            D = int(np.load(p, mmap_mode="r").shape[1]); break
    assert D is not None, "Could not infer feature dim (no X found)."

    model = MS_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, D, ncls).to(device)

    sd = torch.load(pathlib.Path(args.weights), map_location=device)
    model.load_state_dict(sd); model.eval()

    outdir = pathlib.Path(args.outdir or (pathlib.Path(args.exp_dir)/f"eval_{args.split_name}"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"timelines").mkdir(parents=True, exist_ok=True)

    cm = np.zeros((ncls,ncls), np.int64)
    per_video = []

    for stem in stems:
        xpath = pathlib.Path(args.xdir)/f"{stem}_x.npy"
        ypath = pathlib.Path(args.ydir)/f"{stem}_y.npy"
        if not xpath.exists() or not ypath.exists(): continue
        X = np.load(xpath); y = np.load(ypath)
        T = min(len(X), len(y)); X, y = X[:T], y[:T]

        xb = torch.from_numpy(X.T).float().unsqueeze(0).to(device)  # [1,D,T]
        with torch.no_grad():
            logits_bct = model(xb)[-1][0]            # [C,T]
            logits = logits_bct.transpose(0,1).cpu().numpy()  # [T,C]
        pred = logits.argmax(-1)

        # decoding: use argmax only (no postprocessing)
        acc = float((pred == y).mean())
        for g,p in zip(y, pred):
            if 0 <= g < ncls: cm[g,p]+=1
        per_video.append((stem, T, acc))

        if args.make_timelines:
            out_png = outdir/"timelines"/f"{stem}.png"
            plot_timeline_with_legend(y=y, pred=pred, names=names, outfile=str(out_png), title=stem)

    # metrics
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec  = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1   = 2*prec*rec/np.maximum(prec+rec,1e-9)
    macroF1 = float(np.nanmean(f1))
    micro_acc = float(np.diag(cm).sum() / max(1, cm.sum()))
    print(f"{args.split_name.upper()} micro-acc={micro_acc:.4f}  macroF1={macroF1:.4f}")

    # save metrics
    np.savetxt(outdir/f"confusion_{args.split_name}.csv", cm, fmt="%d", delimiter=",")
    with open(outdir/"summary.txt","w") as f:
        f.write(f"micro_acc={micro_acc:.6f}\nmacroF1={macroF1:.6f}\n")
        f.write("per-class (prec, rec, f1):\n")
        for i,nm in enumerate(names):
            f.write(f"{i:02d} {nm:16s}  {prec[i]:.4f} {rec[i]:.4f} {f1[i]:.4f}\n")
        f.write("\nper-video acc:\n")
        for stem,T,acc in per_video:
            f.write(f"{stem},{T},{acc:.6f}\n")

    # confusion heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion ({args.split_name})")
    plt.xlabel("Pred"); plt.ylabel("GT")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir/f"confusion_{args.split_name}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()