#!/usr/bin/env python3
"""
Utilities for temporal action segmentation metrics.

Implements:
- Segment extraction from framewise labels
- Segmental F1@k (IoU overlap thresholds), as used in MS-TCN-style literature
- Edit score (normalized Levenshtein similarity on action sequences)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Segment:
    label: int
    start: int  # inclusive
    end: int  # exclusive

    def length(self) -> int:
        return max(0, self.end - self.start)


def labels_to_segments(
    y: np.ndarray,
    *,
    ignore_labels: Iterable[int] = (-100,),
) -> List[Segment]:
    """
    Convert framewise labels into contiguous segments.

    Notes:
    - Segments are [start, end) half-open intervals in frame indices.
    - Frames with ignore labels are treated as boundaries and not returned as segments.
    """
    y = np.asarray(y).astype(int)
    ignore_set = set(int(x) for x in ignore_labels)

    segs: List[Segment] = []
    if y.size == 0:
        return segs

    start = 0
    prev = int(y[0])
    for i in range(1, int(y.shape[0])):
        cur = int(y[i])
        if cur != prev:
            if prev not in ignore_set:
                segs.append(Segment(label=prev, start=start, end=i))
            start = i
            prev = cur
    # last
    if prev not in ignore_set:
        segs.append(Segment(label=prev, start=start, end=int(y.shape[0])))
    return segs


def filter_segments(
    segs: Sequence[Segment],
    *,
    exclude_labels: Optional[Iterable[int]] = None,
) -> List[Segment]:
    if not exclude_labels:
        return list(segs)
    ex = set(int(x) for x in exclude_labels)
    return [s for s in segs if s.label not in ex]


def segment_iou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    # Standard DP edit distance
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, cur = cur, prev
    return prev[m]


def collapse_to_action_sequence(
    y: np.ndarray,
    *,
    ignore_labels: Iterable[int] = (-100,),
    exclude_labels: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Collapse framewise labels into an action sequence by removing consecutive duplicates.
    Optionally removes background labels and ignore labels.
    """
    y = np.asarray(y).astype(int).tolist()
    ignore_set = set(int(x) for x in ignore_labels)
    exclude_set = set(int(x) for x in (exclude_labels or ()))

    seq: List[int] = []
    prev: Optional[int] = None
    for v in y:
        if v in ignore_set:
            prev = None  # break segments across ignored areas
            continue
        if v == prev:
            continue
        prev = v
        if v in exclude_set:
            continue
        seq.append(v)
    return seq


def edit_score(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    ignore_labels: Iterable[int] = (-100,),
    exclude_labels: Optional[Iterable[int]] = None,
) -> float:
    """
    Normalized edit score in [0, 100], common in action segmentation papers.
    """
    gt_seq = collapse_to_action_sequence(gt, ignore_labels=ignore_labels, exclude_labels=exclude_labels)
    pr_seq = collapse_to_action_sequence(pred, ignore_labels=ignore_labels, exclude_labels=exclude_labels)
    denom = max(len(gt_seq), len(pr_seq))
    if denom == 0:
        return 100.0
    dist = _levenshtein_distance(gt_seq, pr_seq)
    score = (1.0 - (float(dist) / float(denom))) * 100.0
    return float(max(0.0, min(100.0, score)))


def segmental_f1_at_thresholds(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.10, 0.25, 0.50, 0.75),
    ignore_labels: Iterable[int] = (-100,),
    exclude_labels: Optional[Iterable[int]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Compute segmental F1@k at multiple IoU overlap thresholds.

    Returns dict keyed by threshold -> {precision, recall, f1, tp, fp, fn}

    Matching:
    - Only segments of the same class can match.
    - Each GT segment can be matched at most once per threshold.
    - Greedy per predicted segment: match to the best IoU among currently-unmatched GTs.
    """
    gt_segs = filter_segments(
        labels_to_segments(gt, ignore_labels=ignore_labels),
        exclude_labels=exclude_labels,
    )
    pr_segs = filter_segments(
        labels_to_segments(pred, ignore_labels=ignore_labels),
        exclude_labels=exclude_labels,
    )

    gt_by_cls: Dict[int, List[Segment]] = {}
    for s in gt_segs:
        gt_by_cls.setdefault(s.label, []).append(s)
    pr_by_cls: Dict[int, List[Segment]] = {}
    for s in pr_segs:
        pr_by_cls.setdefault(s.label, []).append(s)

    out: Dict[float, Dict[str, float]] = {}
    for thr in thresholds:
        tp = 0
        fp = 0
        fn = 0

        for cls, gt_cls in gt_by_cls.items():
            pr_cls = pr_by_cls.get(cls, [])
            matched = np.zeros((len(gt_cls),), dtype=bool)

            for pseg in pr_cls:
                best_iou = 0.0
                best_j = -1
                for j, gseg in enumerate(gt_cls):
                    if matched[j]:
                        continue
                    iou = segment_iou(pseg, gseg)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= float(thr):
                    tp += 1
                    matched[best_j] = True
                else:
                    fp += 1

            fn += int((~matched).sum())

        # classes that exist only in prediction contribute only FP
        for cls, pr_cls in pr_by_cls.items():
            if cls in gt_by_cls:
                continue
            fp += len(pr_cls)

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[float(thr)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


