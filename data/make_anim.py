"""
data/make_anim.py
-----------------
Create an interactive 3-D animation of the forward VP-SDE acting on a
point-cloud dataset (Lorenz-63 or sphere).

Example
-------
python -m data.make_anim \
    --data  data/lorenz_train_norm.npy \
    --stats data/lorenz_stats.json \
    --out   outputs/animations/anim_raw.html \
    --frames 100 --sample 3000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch

from training.schedule import m_t, sigma
from utils.preprocess import unwhiten


# ----------------------------------------------------------------------
# VP-SDE forward step
# ----------------------------------------------------------------------
def forward_vp_step(
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x(t) = m(t)·x0 + σ(t)·ε  with ε ~ N(0, I)."""
    sig = sigma(t)[:, None]
    m = m_t(t)[:, None]
    eps = (
        torch.randn(x0.shape, generator=generator, device=x0.device)
        if generator
        else torch.randn_like(x0)
    )
    return m * x0 + sig * eps


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def build_dataframe(
    x0: np.ndarray,
    n_frames: int,
    eps: float = 1e-5,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Return long-format DataFrame with (x,y,z,frame)."""
    device = torch.device("cpu")
    g = torch.Generator(device=device)
    if rng_seed is not None:
        g.manual_seed(rng_seed)

    pts = torch.from_numpy(x0).to(device)
    times = torch.linspace(eps, 1.0, n_frames, device=device)

    frames = [
        forward_vp_step(pts, t.expand(pts.size(0)), generator=g).cpu().numpy()
        for t in times
    ]

    stacked = np.concatenate(frames, axis=0)
    frame_idx = np.repeat(np.arange(n_frames), pts.size(0))
    return pd.DataFrame(
        dict(x=stacked[:, 0], y=stacked[:, 1], z=stacked[:, 2], frame=frame_idx)
    )


def animate(
    df: pd.DataFrame,
    outfile: Path,
    range_x: tuple[float, float],
    range_y: tuple[float, float],
    range_z: tuple[float, float],
) -> None:
    """Save Plotly HTML animation with fixed axes that include all points."""
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        animation_frame="frame",
        range_x=list(range_x),
        range_y=list(range_y),
        range_z=list(range_z),
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        scene=dict(aspectmode="auto"),
        title="Forward VP-SDE Evolution",
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,
    )
    fig.write_html(outfile, include_plotlyjs="cdn")
    print(f"[make_anim] Saved animation → {outfile.resolve()}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Visualise forward VP-SDE.")
    p.add_argument(
        "--data", type=Path, required=True, help="Whitened *.npy point cloud"
    )
    p.add_argument(
        "--stats", type=Path, help="Whitening stats JSON to un-whiten before animating"
    )
    p.add_argument("--out", type=Path, default=Path("animation.html"))
    p.add_argument("--frames", type=int, default=100)
    p.add_argument(
        "--sample", type=int, default=3000, help="Sub-sample size (≤N, use N if <=0)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pad", type=float, default=0.05, help="Fractional padding on axis limits"
    )
    args = p.parse_args()

    # ----- load (and un-whiten) -----
    data = unwhiten(np.load(args.data), args.stats)

    if 0 < args.sample < data.shape[0]:
        idx = random.Random(args.seed).sample(range(data.shape[0]), k=args.sample)
        data = data[idx]

    # ----- axis ranges with padding -----
    pad = args.pad
    x_lo, x_hi = data[:, 0].min(), data[:, 0].max()
    y_lo, y_hi = data[:, 1].min(), data[:, 1].max()
    z_lo, z_hi = data[:, 2].min(), data[:, 2].max()

    range_x = (x_lo - pad * (x_hi - x_lo), x_hi + pad * (x_hi - x_lo))
    range_y = (y_lo - pad * (y_hi - y_lo), y_hi + pad * (y_hi - y_lo))
    range_z = (z_lo - pad * (z_hi - z_lo), z_hi + pad * (z_hi - z_lo))

    # ----- build frames & animate -----
    df = build_dataframe(data, n_frames=max(2, args.frames), rng_seed=args.seed)
    animate(df, args.out, range_x, range_y, range_z)


if __name__ == "__main__":
    main()
