# utils/preprocess.py
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def _load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Cache the whitening stats so we hit disk only once."""
    stats = json.loads(stats_path.read_text())
    mu  = np.asarray(stats["mu"], dtype=np.float64)          # (3,)
    inv = np.asarray(stats["inv"], dtype=np.float64)         # (3,3)
    return mu, inv


def unwhiten(arr: np.ndarray, stats_path: Path | None) -> np.ndarray:
    """
    Invert the whitening transform:

        x_raw = x_white @ inv^{-1} + mu

    Parameters
    ----------
    arr : (N,3) whitened array
    stats_path : Path or None
        JSON file with keys "mu" and "inv".
        If None, the array is returned unchanged.

    Returns
    -------
    (N,3) raw array (float32)
    """
    if stats_path is None:
        return arr.astype(np.float32, copy=False)

    mu, inv = _load_stats(stats_path)
    raw = arr @ np.linalg.inv(inv) + mu              # (N,3)
    return raw.astype(np.float32, copy=False)
