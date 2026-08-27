"""
shared_utils.py
---------------
Shared utilities for fair comparison between U-shape and Normal split learning.
Import this in all client scripts to ensure identical seeds and data splits.
"""

import os
import random
import numpy as np
import torch

# ============================================================
# GLOBAL SEED — set this once; both experiments must use the same value
# ============================================================
GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    """Fix all random sources so both experiments start identically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes Conv layers deterministic (slight perf cost, worth it for comparison)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_shuffled_indices(n: int, seed: int = GLOBAL_SEED) -> np.ndarray:
    """
    Return a reproducible shuffled index array of length n.
    Both methods call this with the same n and seed → identical data order.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def save_indices(indices: np.ndarray, machine_id: str, split: str = "train"):
    """Persist indices to disk so they can be reused across separate runs."""
    os.makedirs("indices", exist_ok=True)
    path = f"indices/{machine_id}_{split}.npy"
    np.save(path, indices)


def load_indices(machine_id: str, split: str = "train") -> np.ndarray | None:
    """Load previously saved indices, or return None if not yet generated."""
    path = f"indices/{machine_id}_{split}.npy"
    if os.path.exists(path):
        return np.load(path)
    return None


def get_or_create_indices(n: int, machine_id: str, split: str = "train") -> np.ndarray:
    """
    Load saved indices if they exist; otherwise generate, save, and return them.
    This guarantees BOTH experiments see the same data order even when run
    as separate processes on separate days.
    """
    idx = load_indices(machine_id, split)
    if idx is not None and len(idx) == n:
        return idx
    idx = get_shuffled_indices(n)
    save_indices(idx, machine_id, split)
    return idx