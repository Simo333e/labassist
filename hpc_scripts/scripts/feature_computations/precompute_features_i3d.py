#!/usr/bin/env python3
import argparse, pathlib, json, math, numpy as np, torch, cv2
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torchvision.transforms as T

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FineBio root containing videos/")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--view", default="T1")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--win", type=int, default=32)      # frames per clip
    ap.add_argument("--stride", type=int, default=2)    # sample every 2 src frames inside window
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="default: root/cache/features_10hz_i3d")
    
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--center_step", type=int, default=1, help="sample every kth center timestamp")
    ap.add_argument("--shards", type=int, default=1, help="split stems across N shards")
    ap.add_argument("--shard_idx", type=int, default=0, help="0-based shard index")
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root)
    OUT  = pathlib.Path(args.out or (ROOT/"cache/features_10hz_i3d")); OUT.mkdir(parents=True, exist_ok=True)

    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems_all = [pathlib.Path(p).stem for p in sp["train"] + sp["val"] + sp["test"]]
    # sharding for job arrays
    S = max(1, int(args.shards))
    I = int(args.shard_idx)
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]

    vids = {v.stem: v for v in (ROOT/"videos").glob(f"*_{args.view}.mp4")}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights).to(device).eval()
    # get 512-D features via forward hook
    feat_dim = 512
    features = {}
    def hook(module, inp, out):
        # out is [B, 512, 1, 1, 1] after avgpool
        features["z"] = out.flatten(1)
    model.avgpool.register_forward_hook(hook)
    # Usual normalization did not work
    mean = [0.43216, 0.394666, 0.37645]
    std  = [0.22803, 0.22145, 0.216989]
    tfm = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Resize((args.imgsz, args.imgsz)),
        T.Normalize(mean=mean, std=std),
    ])

    for stem in stems:
        vpath = vids.get(stem)
        if not vpath: continue
        out_npy = OUT/f"{stem}_x.npy"
        out_meta = OUT/f"{stem}_meta.json"
        if out_npy.exists(): continue

        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nframes <= 0:
            cap.release(); continue

        # read all frames 
        frames = []
        ok = True
        while ok:
            ok, f = cap.read()
            if ok: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: continue

        # map target fps timeline to center indices in source frames
        T_out = int(math.ceil((nframes / src_fps) * args.fps))
        centers = [min(nframes-1, int(round((t/args.fps) * src_fps))) for t in range(T_out)]
        step = max(1, int(args.center_step))
        centers = centers[::step]

        half = args.win // 2
        X = []
        
        tframes = [tfm(f) for f in frames]  # each [3,H,W]

        def clip_tensor(center):
            idxs = np.arange(center - half*args.stride, center + half*args.stride, args.stride)
            idxs = np.clip(idxs, 0, nframes-1).astype(int)
            return torch.stack([tframes[i] for i in idxs], dim=1)  # [3,T,H,W]

        B = max(1, int(args.batch_size))
        for s in range(0, len(centers), B):
            batch_centers = centers[s:s+B]
            if not batch_centers:
                continue
            clips = torch.stack([clip_tensor(c) for c in batch_centers], dim=0).to(device)  # [B,3,T,H,W]
            features.clear()
            _ = model(clips)
            Z = features["z"].detach().cpu().numpy()  # [B,512]
            X.extend([z.astype(np.float32) for z in Z])

        X = np.stack(X, axis=0)  # [T,512]
        np.save(out_npy, X)
        out_meta.write_text(json.dumps({"stem": stem, "fps": args.fps, "src_fps": src_fps, "T": int(X.shape[0]), "D": int(X.shape[1])}))
        print(f"{stem}: X{tuple(X.shape)} saved → {out_npy}")

if __name__ == "__main__":
    main()