#!/usr/bin/python2.7

import torch
from batch_gen import BatchGenerator
import os
import argparse
import random
from glob import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='')
parser.add_argument('--causal', action='store_true', help='Use causal MS-TCN model from causal_model.py')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)

parser.add_argument('--run_dir', type=str)

parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

# FineBio NPY dataset mode (optional). If --splits is provided, overrides legacy bundle/mapping mode
parser.add_argument('--splits', type=str, default='', help='Path to splits JSON for FineBio npy features (enables NPY mode)')
parser.add_argument('--xdir', type=str, default='', help='Directory containing *_x.npy feature files')
parser.add_argument('--ydir', type=str, default='', help='Directory containing *_y.npy label files')
parser.add_argument('--class_index', type=str, default='', help='JSON mapping of class names to indices (or list of names)')
parser.add_argument('--exp_dir', type=str, default='', help='Output experiment directory (used in NPY mode)')

args = parser.parse_args()

# Conditional import to select vanilla vs causal Trainer
if args.causal:
    from causal_model import Trainer
else:
    from model import Trainer

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1

if args.splits:
    # ---- NPY dataset mode (FineBio cache with *_x.npy and *_y.npy) ----
    import json
    import pathlib
    import numpy as np

    class NpyBatchGenerator(object):
        def __init__(self, pairs, num_classes, sample_rate):
            self.list_of_examples = pairs  # list of (x_path, y_path)
            self.index = 0
            self.num_classes = num_classes
            self.sample_rate = sample_rate

        def reset(self):
            self.index = 0
            random.shuffle(self.list_of_examples)

        def has_next(self):
            return self.index < len(self.list_of_examples)

        def next_batch(self, batch_size):
            batch = self.list_of_examples[self.index:self.index + batch_size]
            self.index += batch_size

            batch_input = []
            batch_target = []
            for x_path, y_path in batch:
                x = np.load(x_path)  # [T, D]
                y = np.load(y_path)  # [T]
                T = min(x.shape[0], y.shape[0])
                x = x[:T]
                y = y[:T]
                batch_input.append(x[::self.sample_rate].T)   # [D, T]
                batch_target.append(y[::self.sample_rate])     # [T]

            length_of_sequences = list(map(len, batch_target))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
                mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

    # load splits and build pairs
    sp = json.loads(pathlib.Path(args.splits).read_text())
    if "train" not in sp:
        sp = next(iter(sp.values()))
    stems = [pathlib.Path(p).stem for p in sp.get("train", [])]
    xdir = pathlib.Path(args.xdir)
    ydir = pathlib.Path(args.ydir)
    pairs = []
    for s in stems:
        xp = xdir / (s + "_x.npy")
        yp = ydir / (s + "_y.npy")
        if xp.exists() and yp.exists():
            pairs.append((str(xp), str(yp)))

    # infer feature dim and classes
    assert len(pairs) > 0, "No training pairs found—check xdir/ydir/splits"
    features_dim = int(np.load(pairs[0][0], mmap_mode='r').shape[1])
    cls_obj = json.loads(pathlib.Path(args.class_index).read_text()) if args.class_index else None
    if isinstance(cls_obj, dict):
        num_classes = len(cls_obj)
    elif isinstance(cls_obj, list):
        num_classes = len(cls_obj)
    else:
        # fallback: max label in first y + 1
        num_classes = int(np.max(np.load(pairs[0][1], mmap_mode='r')) + 1)

    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "finebio", run_dir=args.run_dir or "./")
    model_dir = args.exp_dir or (args.run_dir or "./") + "/MS-TCN2/models/finebio_npy_" + f"lr{lr:.0e}_PG{num_layers_PG}_R{num_layers_R}*{num_R}"
    os.makedirs(model_dir, exist_ok=True)
    batch_gen = NpyBatchGenerator(pairs, num_classes, sample_rate)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
else:
    # ---- Legacy bundle/mapping mode ----
    if len(args.split):
        vid_list_file = args.run_dir + "/data/splits/train_split" + args.split + ".bundle"
        mapping_file = args.run_dir + "/data/mapping_" + args.split + ".txt"
        model_dir = args.run_dir + f"/MS-TCN2/models/split_" + args.split + "_lr{:.0e}_PG{}_R{}*{}".format(lr, num_layers_PG, num_layers_R, num_R)
    else:
        vid_list_file = args.run_dir + "/data/splits/train.bundle"
        mapping_file = args.run_dir + "/data/mapping.txt"
        model_dir = args.run_dir + f"/MS-TCN2/models/all" + args.split + "_lr{:.0e}_PG{}_R{}*{}".format(lr, num_layers_PG, num_layers_R, num_R)
    features_path = args.run_dir + f"/data/rgbflow_i3d_features/"
    gt_path = args.run_dir + "/data/groundTruth/"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)
    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "finebio", run_dir=args.run_dir)
    os.makedirs(model_dir, exist_ok=True)
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)