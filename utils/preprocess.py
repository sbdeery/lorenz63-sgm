# utils/preprocess.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=4)
def _load_stats(stats_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cache the whitening stats so we hit disk only once."""
    stats = json.loads(stats_path.read_text())
    mu = np.asarray(stats["mu"], dtype=np.float64)  # (3,)
    inv = np.asarray(stats["inv"], dtype=np.float64)  # (3,3)
    return mu, inv


def unwhiten(arr: NDArray[np.float32], stats_path: Path | None) -> NDArray[np.float32]:
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
    raw = arr @ np.linalg.inv(inv) + mu  # (N,3)
    return cast(NDArray[np.float32], raw.astype(np.float32, copy=False))
