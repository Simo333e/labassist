#!/usr/bin/env python3
"""Helpers for loading/transferring pretrained MS-TCN weights."""

import torch
from loguru import logger


SKIP_PATTERNS = [
    "conv_out",  # Classification heads depend on num_classes
    "Rs.0.conv_1x1",
    "Rs.1.conv_1x1",
    "Rs.2.conv_1x1",
    "Rs.3.conv_1x1",
    "Rs.4.conv_1x1",
]


def get_transferable_keys(state_dict):
    transferable = {}
    skipped = []
    for key, value in state_dict.items():
        if any(pattern in key for pattern in SKIP_PATTERNS):
            skipped.append(key)
        else:
            transferable[key] = value
    return transferable, skipped


def transfer_pretrained_weights(model, checkpoint_path):
    """
    Load a checkpoint and copy matching backbone weights into the provided model.
    Classification heads are skipped since they depend on the new num_classes.
    """
    pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
    transferable, skipped = get_transferable_keys(pretrained_dict)

    logger.info(f"Pretrained checkpoint contains {len(pretrained_dict)} tensors")
    logger.info(f"Transferring {len(transferable)} tensors to new model")
    if skipped:
        logger.info(f"Skipping {len(skipped)} classification tensors:")
        for name in skipped:
            logger.info(f"  - {name}")

    model_dict = model.state_dict()
    matched = {}
    mismatched = []
    for key, value in transferable.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            matched[key] = value
        else:
            mismatched.append(key)

    if mismatched:
        logger.warning(f"Mismatched tensors ({len(mismatched)}); skipping:")
        for name in mismatched:
            logger.warning(f"  - {name}")

    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return model

