#!/usr/bin/env python3
"""
Finetune MS-TCN2 on a new dataset with different action classes.

Key simplifications:
* Reuse Trainer from MS-TCN for the main optimization loop
* Always train the full network (no backbone freezing)
* Share batch generation + pretrained transfer utilities with the base repo
"""

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch
from loguru import logger

# Add MS-TCN to sys.path so we can import trainer/utilities.
MS_TCN_PATH = pathlib.Path(__file__).resolve().parents[1] / "MS-TCN"
if str(MS_TCN_PATH) not in sys.path:
    sys.path.insert(0, str(MS_TCN_PATH))

from causal_model import Trainer  # noqa: E402
from npy_batch_gen import NpyBatchGenerator  # noqa: E402
from pretrained_utils import transfer_pretrained_weights  # noqa: E402


def load_config_file(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required to parse YAML configs. Install with `pip install pyyaml`.") from exc
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    return cfg


def _maybe_float(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _maybe_int(value):
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def apply_config(args, cfg):
    if not cfg:
        return

    args.exp_dir = cfg.get("exp_dir", args.exp_dir)
    args.exp_name = cfg.get("exp_name", args.exp_name)
    args.device = cfg.get("device", args.device)

    pretrained_cfg = cfg.get("pretrained") or {}
    args.pretrained = pretrained_cfg.get("checkpoint", args.pretrained)
    args.pretrained_num_classes = _maybe_int(pretrained_cfg.get("num_classes", args.pretrained_num_classes))

    for key, value in (cfg.get("data") or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)

    for key, value in (cfg.get("model") or {}).items():
        if hasattr(args, key):
            if key in {"num_layers_PG", "num_layers_R", "num_R", "num_f_maps", "features_dim"}:
                setattr(args, key, _maybe_int(value))
            else:
                setattr(args, key, value)

    for key, value in (cfg.get("training") or {}).items():
        if hasattr(args, key):
            if key in {"num_epochs", "batch_size", "window_size", "window_step", "max_windows_per_video"}:
                setattr(args, key, _maybe_int(value))
            elif key in {"lr", "lr_finetune", "feature_dropout_p", "max_class_weight", "balanced_mix_ratio", "focal_gamma"}:
                setattr(args, key, _maybe_float(value))
            elif key in {"freeze_pg", "linear_probe", "use_focal_loss", "balanced_windows", "use_class_weights"}:
                # Boolean flags
                setattr(args, key, bool(value))
            else:
                setattr(args, key, value)


def save_resolved_config(args):
    resolved = {
        "config_source": args.config or "cli",
        "exp_name": args.exp_name,
        "exp_dir": args.exp_dir,
        "device": args.device,
        "pretrained": {
            "checkpoint": args.pretrained,
            "num_classes": args.pretrained_num_classes,
        },
        "data": {
            "splits": args.splits,
            "class_index": args.class_index,
            "xdir": args.xdir,
            "ydir": args.ydir,
        },
        "model": {
            "num_layers_PG": args.num_layers_PG,
            "num_layers_R": args.num_layers_R,
            "num_R": args.num_R,
            "num_f_maps": args.num_f_maps,
            "features_dim": args.features_dim,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_finetune": args.lr_finetune,
            "use_class_weights": args.use_class_weights,
            "feature_dropout_p": args.feature_dropout_p,
            "max_class_weight": args.max_class_weight,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "balanced_windows": args.balanced_windows,
            "background_label": args.background_label,
            "max_windows_per_video": args.max_windows_per_video,
            "balanced_mix_ratio": args.balanced_mix_ratio,
            "use_focal_loss": args.use_focal_loss,
            "focal_gamma": args.focal_gamma,
            "freeze_pg": args.freeze_pg,
            "linear_probe": args.linear_probe,
        },
    }
    out_path = pathlib.Path(args.exp_dir) / "resolved_config.yaml"
    try:
        import yaml
        out_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    except Exception:
        out_path.write_text(json.dumps(resolved, indent=2))


def load_class_index(class_index_path):
    cls_obj = json.loads(pathlib.Path(class_index_path).read_text())
    if isinstance(cls_obj, dict):
        return list(cls_obj.keys())
    return list(cls_obj)


def build_training_pairs(splits_path, xdir, ydir, num_classes):
    sp = json.loads(pathlib.Path(splits_path).read_text())
    stems = [pathlib.Path(p).stem for p in sp.get("train", [])]
    xdir, ydir = pathlib.Path(xdir), pathlib.Path(ydir)

    pairs = []
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for stem in stems:
        xp = xdir / f"{stem}_x.npy"
        yp = ydir / f"{stem}_y.npy"
        if xp.exists() and yp.exists():
            pairs.append((str(xp), str(yp)))
            y = np.load(yp)
            valid = y[(y >= 0) & (y < num_classes)]
            if valid.size:
                class_counts += np.bincount(valid, minlength=num_classes)
        else:
            logger.warning(f"Missing data for {stem}: features={xp.exists()} labels={yp.exists()}")

    return pairs, class_counts


def verify_feature_dim(pairs, expected_dim):
    sample_x = np.load(pairs[0][0])
    actual_dim = sample_x.shape[1]
    if actual_dim != expected_dim:
        raise SystemExit(f"Feature dimension mismatch! Expected {expected_dim}, got {actual_dim}")


def compute_class_weights(class_counts, max_weight=None):
    total_frames = class_counts.sum()
    if total_frames == 0:
        return None, {}

    weights = np.zeros_like(class_counts, dtype=np.float32)
    stats = {}
    for idx, cnt in enumerate(class_counts):
        frac = cnt / total_frames if total_frames else 0.0
        stats[idx] = {"frames": int(cnt), "fraction": float(frac)}
        if cnt > 0:
            weight = total_frames / (len(class_counts) * cnt)
            if max_weight is not None:
                weight = min(weight, max_weight)
            weights[idx] = weight
    if not weights.any():
        return None, stats
    return torch.tensor(weights, dtype=torch.float32), stats


def resolve_learning_rate(args):
    if args.lr_finetune:
        logger.warning("--lr_finetune is deprecated; using it as the training lr")
        return args.lr_finetune
    return args.lr


def parse_args():
    ap = argparse.ArgumentParser(description="Finetune MS-TCN2 on custom data")
    ap.add_argument("--config", type=str, help="Path to YAML/JSON config file")

    # Data paths
    ap.add_argument("--pretrained", default="", help="Path to pretrained .model checkpoint")
    ap.add_argument("--pretrained_num_classes", type=int, default=0,
                    help="Number of classes in pretrained model (33 for FineBio)")
    ap.add_argument("--splits", default="", help="Path to splits.json")
    ap.add_argument("--xdir", default="", help="Directory with *_x.npy features")
    ap.add_argument("--ydir", default="", help="Directory with *_y.npy labels")
    ap.add_argument("--class_index", default="", help="Path to class_index.json")
    ap.add_argument("--exp_dir", default="", help="Output experiment directory")
    ap.add_argument("--exp_name", default="", help="Experiment name (for logging)")

    # Architecture (must match pretrained for backbone)
    ap.add_argument("--num_layers_PG", type=int, default=10)
    ap.add_argument("--num_layers_R", type=int, default=10)
    ap.add_argument("--num_R", type=int, default=5)
    ap.add_argument("--num_f_maps", type=int, default=64)
    ap.add_argument("--features_dim", type=int, default=5120,
                    help="Feature dimension (5120 for ResNet18+Motion+ctx5)")

    # Training
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--lr_finetune", type=float, default=None,
                    help="Deprecated alias for --lr (kept for backwards compatibility)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_class_weights", action="store_true",
                    help="Reweight CE loss by inverse class frequency")
    ap.add_argument("--feature_dropout_p", type=float, default=0.0,
                    help="Probability of zeroing entire feature frames during training")
    ap.add_argument("--max_class_weight", type=float, default=3.0,
                    help="Maximum class weight when --use_class_weights is enabled")
    ap.add_argument("--background_label", default="none",
                    help="Label treated as background for balanced window sampling")
    ap.add_argument("--window_size", type=int, default=0,
                    help="Fixed window size (frames) for training samples (0 = full video)")
    ap.add_argument("--window_step", type=int, default=0,
                    help="Stride between windows when --window_size > 0 (0 = auto)")
    ap.add_argument("--balanced_windows", action="store_true",
                    help="Sample windows centered on non-background frames")
    ap.add_argument("--max_windows_per_video", type=int, default=None,
                    help="Max training windows per video when windowing (None = no cap)")
    ap.add_argument("--balanced_mix_ratio", type=float, default=1.0,
                    help="Fraction of capped windows that are action-centered when balanced_windows is true (0-1)")
    
    # Focal Loss options
    ap.add_argument("--use_focal_loss", action="store_true",
                    help="Use Focal Loss instead of CrossEntropy (helps with class imbalance)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma parameter (higher = more focus on hard examples)")

    # Freezing options for stable training
    ap.add_argument("--freeze_pg", action="store_true",
                    help="Freeze Prediction Generator (PG) stage, only train Refinement stages")
    ap.add_argument("--linear_probe", action="store_true",
                    help="Linear probing: freeze all layers except final conv_out heads")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_config_file(args.config)
        apply_config(args, cfg)

    required_fields = {
        "splits": args.splits,
        "xdir": args.xdir,
        "ydir": args.ydir,
        "class_index": args.class_index,
        "exp_dir": args.exp_dir,
    }
    missing = [name for name, value in required_fields.items()
               if (value is None) or (isinstance(value, str) and not value.strip())]
    if missing:
        raise SystemExit(f"Missing required settings (pass via CLI or config): {missing}")

    # Setup
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.exp_name:
        args.exp_name = pathlib.Path(args.exp_dir).name

    os.makedirs(args.exp_dir, exist_ok=True)
    logger.add(f"{args.exp_dir}/train.log")
    logger.add(sys.stdout, colorize=True, format="{message}")

    save_resolved_config(args)

    class_names = load_class_index(args.class_index)
    num_classes = len(class_names)
    try:
        background_idx = class_names.index(args.background_label)
    except ValueError:
        background_idx = 0
        logger.warning(f"Background label '{args.background_label}' not found. Falling back to index 0 ({class_names[0]}).")

    logger.info("=" * 60)
    logger.info("Finetuning MS-TCN2")
    logger.info("=" * 60)
    logger.info(f"Pretrained checkpoint : {args.pretrained}")
    logger.info(f"Pretrained classes    : {args.pretrained_num_classes}")
    logger.info(f"New classes           : {num_classes}")
    for idx, name in enumerate(class_names):
        logger.info(f"  {idx:02d}: {name}")
    logger.info(f"Features dim          : {args.features_dim}")
    logger.info(f"Total epochs          : {args.num_epochs}")
    logger.info("=" * 60)

    pairs, class_counts = build_training_pairs(args.splits, args.xdir, args.ydir, num_classes)
    if not pairs:
        raise SystemExit("No training pairs found. Check splits/xdir/ydir.")

    verify_feature_dim(pairs, args.features_dim)
    logger.info(f"Training on {len(pairs)} videos")

    class_weights = None
    if args.use_class_weights:
        tensor, stats = compute_class_weights(class_counts, max_weight=args.max_class_weight)
        class_weights = tensor
        logger.info("Class distribution (frames):")
        total_frames = class_counts.sum()
        for idx, name in enumerate(class_names):
            cnt = stats.get(idx, {}).get("frames", 0)
            frac = stats.get(idx, {}).get("fraction", 0.0)
            logger.info(f"  {idx:02d} {name:20s} {cnt:8d} ({frac:.2%})")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args.num_layers_PG, args.num_layers_R, args.num_R,
                      args.num_f_maps, args.features_dim, num_classes,
                      dataset=args.exp_name, run_dir=args.exp_dir)
    logger.info(f"Model size: {sum(p.numel() for p in trainer.model.parameters()):,} params")

    if args.pretrained and pathlib.Path(args.pretrained).exists():
        trainer.model = transfer_pretrained_weights(trainer.model, args.pretrained)
    else:
        logger.info("No pretrained checkpoint provided; training from scratch")

    # Apply freezing options
    if args.linear_probe:
        # Freeze everything except conv_out layers (true linear probing)
        frozen_count = 0
        trainable_count = 0
        for name, param in trainer.model.named_parameters():
            if "conv_out" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        logger.info(f"Linear probing: frozen {frozen_count:,} params, training {trainable_count:,} conv_out params")
    elif args.freeze_pg:
        # Freeze only PG stage, train Rs stages
        frozen_count = 0
        trainable_count = 0
        for name, param in trainer.model.named_parameters():
            if name.startswith("PG."):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        logger.info(f"Freeze PG: frozen {frozen_count:,} params (PG), training {trainable_count:,} params (Rs)")

    learning_rate = resolve_learning_rate(args)
    batch_gen = NpyBatchGenerator(
        pairs,
        num_classes,
        window_size=args.window_size,
        window_step=args.window_step,
        balanced_windows=args.balanced_windows,
        background_idx=background_idx,
        max_windows_per_video=args.max_windows_per_video,
        balanced_mix_ratio=args.balanced_mix_ratio,
    )
    logger.info(f"Batch generator prepared with {len(batch_gen.list_of_examples)} samples")
    if args.use_focal_loss:
        logger.info(f"Using Focal Loss with gamma={args.focal_gamma}")
    trainer.train(
        args.exp_dir,
        batch_gen,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        device=device,
        class_weights=class_weights,
        feature_dropout_p=args.feature_dropout_p,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    torch.save(trainer.model.state_dict(), f"{args.exp_dir}/final.model")
    logger.info(f"✓ Training complete. Models saved to {args.exp_dir}")


if __name__ == "__main__":
    main()




