"""
Compare target and generated samples in 3-D.

Example
-------
python -m scripts.plot_samples \
    --train   data/lorenz_train_norm.npy \
    --samples outputs/pc_lorenz_samples.npz \
    --stats   data/lorenz_stats.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objs as go

from utils.preprocess import unwhiten


def random_subset(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(arr.shape[0], n, replace=False)
    return arr[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--stats",   type=Path,
                        help="JSON whitening stats to un-whiten before plotting")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--outdir",  type=Path, default=Path("outputs/animations"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # load & un-whiten if requested
    train_all = unwhiten(np.load(args.train),               args.stats)
    samples   = unwhiten(np.load(args.samples)["samples"],  args.stats)

    train_plot = random_subset(train_all, 3_000, rng)
    gen_plot   = random_subset(samples,   1_000, rng)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=train_plot[:, 0], y=train_plot[:, 1], z=train_plot[:, 2],
        mode="markers", marker=dict(size=2, color="blue"), name="True"))
    fig.add_trace(go.Scatter3d(
        x=gen_plot[:, 0],  y=gen_plot[:, 1],  z=gen_plot[:, 2],
        mode="markers", marker=dict(size=3, color="orange"), name="Generated"))

    title = "True vs Generated (un-whitened)" if args.stats else \
            "True vs Generated (whitened)"
    fig.update_layout(
        scene=dict(aspectmode="auto"),
        title=title, margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    html_path = args.outdir / args.samples.with_suffix(".html").name
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"[plot_samples] wrote {html_path.resolve()}")


if __name__ == "__main__":
    main()