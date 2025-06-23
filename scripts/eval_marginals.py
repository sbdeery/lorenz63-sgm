"""
Evaluate sample fidelity along 1-D marginals (x, y, z).

Usage
-----
python -m scripts.eval_marginals \
    --data    data/lorenz_train_norm.npy \
    --samples outputs/pc_lorenz_samples.npz \
    --stats   data/lorenz_stats.json \
    --outdir  outputs/marginals \
    --bins    150
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

from utils.preprocess import unwhiten

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def gaussian_kde_silverman(data: np.ndarray) -> gaussian_kde:  # type: ignore[valid-type]
    """Gaussian KDE with Silverman bandwidth."""
    return gaussian_kde(data.astype(np.float64), bw_method="silverman")


def kde_mse(true: np.ndarray, gen: np.ndarray, num_pts: int = 512) -> float:
    """L2-MSE between KDE curves on a shared grid."""
    kde_t = gaussian_kde_silverman(true)
    kde_g = gaussian_kde_silverman(gen)
    lo, hi = min(true.min(), gen.min()), max(true.max(), gen.max())
    grid = np.linspace(lo, hi, num_pts)
    return float(np.mean((kde_t(grid) - kde_g(grid)) ** 2))


def compute_metrics(true: np.ndarray, gen: np.ndarray) -> Tuple[float, float, float]:
    ks, _ = ks_2samp(true, gen)
    w1 = wasserstein_distance(true, gen)
    mse = kde_mse(true, gen)
    return ks, w1, mse  # type: ignore[return-value]


def plot_axis(
    true: np.ndarray,
    gen: np.ndarray,
    axis: str,
    out: Path,
    bins: int = 100,
) -> None:
    """Save histogram + KDE overlay for a single coordinate."""
    out.parent.mkdir(parents=True, exist_ok=True)

    combined = np.concatenate([true, gen])
    _, edges = np.histogram(combined, bins=bins, density=True)
    edges_list: list[float] = edges.tolist()  # type: ignore[arg-type]

    plt.figure(figsize=(5, 4))
    plt.hist(
        true, bins=edges_list, density=True, alpha=0.4, label="True", color="tab:blue"
    )
    plt.hist(
        gen,
        bins=edges_list,
        density=True,
        alpha=0.4,
        label="Generated",
        color="tab:orange",
    )

    grid = np.linspace(edges[0], edges[-1], 512)
    plt.plot(grid, gaussian_kde_silverman(true)(grid), color="tab:blue", linewidth=1.5)
    plt.plot(grid, gaussian_kde_silverman(gen)(grid), color="tab:orange", linewidth=1.5)

    plt.title(f"{axis}-marginal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def subsample(
    arr: np.ndarray, max_pts: int | None, rng: np.random.Generator
) -> np.ndarray:
    if max_pts is None or arr.shape[0] <= max_pts:
        return arr
    idx = rng.choice(arr.shape[0], max_pts, replace=False)
    return arr[idx]


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate 1-D marginal fidelity.")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument(
        "--stats",
        type=Path,
        help="JSON with whitening stats to un-whiten before evaluation",
    )
    p.add_argument("--outdir", type=Path, default=Path("outputs/marginals"))
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_true", type=int)
    p.add_argument("--max_gen", type=int)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # load & (optionally) un-whiten
    true_xyz = unwhiten(np.load(args.data), args.stats)
    gen_xyz = unwhiten(np.load(args.samples)["samples"], args.stats)

    # optional subsample
    true_xyz = subsample(true_xyz, args.max_true, rng)
    gen_xyz = subsample(gen_xyz, args.max_gen, rng)

    metrics_rows: list[dict[str, Union[str, float]]] = []
    for idx, axis in enumerate(["x", "y", "z"]):
        ks, w1, mse = compute_metrics(true_xyz[:, idx], gen_xyz[:, idx])
        metrics_rows.append({"axis": axis, "KS": ks, "W1": w1, "KDE_MSE": mse})
        plot_axis(
            true_xyz[:, idx],
            gen_xyz[:, idx],
            axis,
            args.outdir / f"{axis}.png",
            bins=args.bins,
        )
        print(f"{axis}: KS={ks:.4f}  W1={w1:.4f}  KDE-MSE={mse:.2e}")

    # append to CSV
    csv_path = args.outdir / "metrics.csv"
    header = ["axis", "KS", "W1", "KDE_MSE"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerows(metrics_rows)


if __name__ == "__main__":
    main()
