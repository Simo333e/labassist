#!/usr/bin/env python3
"""
NPY-based batch generator for MS-TCN training/finetuning workflows.
Shared between the legacy train script and finetuning utilities.
"""

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

IGNORE_INDEX = -100


@dataclass
class SampleWindow:
    x_path: str
    y_path: str
    start: int = 0
    length: int = 0  # 0 = use full length


class NpyBatchGenerator(object):
    """Batch generator for feature/label NPY files."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        num_classes: int,
        sample_rate: int = 1,
        window_size: int = 0,
        window_step: int = 0,
        balanced_windows: bool = False,
        background_idx: int = 0,
        ignore_index: int = IGNORE_INDEX,
        max_windows_per_video: int | None = None,
        balanced_mix_ratio: float = 1.0,
    ):
        self.num_classes = num_classes
        self.sample_rate = max(1, sample_rate)
        self.window_size = max(0, window_size)
        self.window_step = max(1, window_step or (self.window_size if self.window_size else 1))
        self.balanced_windows = balanced_windows
        self.background_idx = background_idx
        self.ignore_index = ignore_index
        self.max_windows_per_video = max_windows_per_video
        self.balanced_mix_ratio = max(0.0, min(1.0, float(balanced_mix_ratio)))

        self.list_of_examples: List[SampleWindow] = self._build_examples(pairs)
        self.index = 0

    def _build_examples(self, pairs: Sequence[Tuple[str, str]]) -> List[SampleWindow]:
        samples: List[SampleWindow] = []
        for x_path, y_path in pairs:
            if self.window_size <= 0:
                samples.append(SampleWindow(x_path, y_path, start=0, length=0))
                continue
            labels = np.load(y_path)
            seq_len = len(labels)
            if seq_len <= self.window_size:
                samples.append(SampleWindow(x_path, y_path, start=0, length=self.window_size))
                continue
            starts = self._compute_window_starts(labels, seq_len)
            for start in starts:
                samples.append(SampleWindow(x_path, y_path, start=start, length=self.window_size))
        if not samples:
            # Fallback to original pairs if something went wrong
            samples = [SampleWindow(x, y, start=0, length=0) for x, y in pairs]
        random.shuffle(samples)
        return samples

    def _compute_window_starts(self, labels: np.ndarray, seq_len: int) -> List[int]:
        # Uniform grid of windows across the sequence
        uniform_starts = set(range(0, max(1, seq_len - self.window_size + 1), self.window_step))

        if not self.balanced_windows:
            # No action centering; just return the uniform windows
            return sorted(uniform_starts)

        # Action-centered windows around non-background, non-ignored frames
        valid = np.where((labels != self.background_idx) & (labels != self.ignore_index))[0]
        balanced_starts = set()
        for idx in valid:
            start = idx - self.window_size // 2
            start = max(0, min(start, seq_len - self.window_size))
            balanced_starts.add(start)

        # If no actions found, fall back to uniform windows only
        if not balanced_starts:
            return sorted(uniform_starts)

        total_candidates = list(balanced_starts | uniform_starts)
        if self.max_windows_per_video is None or len(total_candidates) <= self.max_windows_per_video:
            return sorted(total_candidates)

        # Mix balanced (action-centered) and uniform windows up to the cap
        num_balanced = min(len(balanced_starts),
                           int(self.max_windows_per_video * self.balanced_mix_ratio))
        num_uniform = min(len(uniform_starts),
                          self.max_windows_per_video - num_balanced)

        chosen = []
        if num_balanced > 0:
            chosen += random.sample(list(balanced_starts), num_balanced)
        if num_uniform > 0:
            chosen += random.sample(list(uniform_starts), num_uniform)

        if not chosen:
            chosen = random.sample(total_candidates, self.max_windows_per_video)

        return sorted(chosen)

    def reset(self) -> None:
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input, batch_target = [], []
        for sample in batch:
            x = np.load(sample.x_path)  # [T, D]
            y = np.load(sample.y_path)  # [T]
            T = min(x.shape[0], y.shape[0])

            if sample.length and T > 0:
                start = min(sample.start, max(0, T - 1))
                end = min(start + sample.length, T)
                x = x[start:end]
                y = y[start:end]
                T = min(x.shape[0], y.shape[0])

            x_window = x[:T:self.sample_rate].T  # [D, T]
            y_window = y[:T:self.sample_rate]    # [T]

            batch_input.append(x_window)
            batch_target.append(y_window)

        lengths = list(map(len, batch_target))
        max_len = max(lengths)
        feat_dim = batch_input[0].shape[0]
        batch_input_tensor = torch.zeros(len(batch_input), feat_dim, max_len, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_len, dtype=torch.long) * self.ignore_index
        mask = torch.zeros(len(batch_input), self.num_classes, max_len, dtype=torch.float)

        for i in range(len(batch_input)):
            length = batch_input[i].shape[1]
            batch_input_tensor[i, :, :length] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :length] = torch.from_numpy(batch_target[i])
            valid_mask = torch.from_numpy((batch_target[i] != self.ignore_index).astype(np.float32))
            mask[i, :, :length] = valid_mask.unsqueeze(0)

        return batch_input_tensor, batch_target_tensor, mask

