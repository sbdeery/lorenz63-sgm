#!/usr/bin/env python3
"""
Sample points from a trained score model.

Examples
--------
ODE sampler, 1 000 points::

    python -m scripts.sample \
        --ckpt outputs/ckpts/e50.pt \
        --n 1000 \
        --sample_type ode
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Callable, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor

from models.mlp import ScoreMLP
from sampling.em import em_sampler
from sampling.ode import ode_sampler
from sampling.pc import pc_sampler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", required=True, type=pathlib.Path, help="Path to model checkpoint"
    )
    p.add_argument("--n", required=True, type=int, help="Number of points to draw")
    p.add_argument(
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Custom output *.npz (overrides auto-name)",
    )
    p.add_argument(
        "--sample_type",
        required=True,
        choices=("ode", "em", "pc"),
        help="Sampler to use: ode | em | pc",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScoreMLP().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # score function: input (B,3), (B,) → output (B,3)
    def score_fn(x: Tensor, t: Tensor) -> Tensor:
        # model(x,t) stub is untyped, so cast to Tensor
        return cast(Tensor, model(x, t))

    # Select sampler -------------------------------------------------------
    # all samplers have signature: (score_fn, ..., device) -> Tensor
    SAMPLERS: dict[str, Callable[..., Tensor]] = {
        "ode": ode_sampler,
        "em": em_sampler,
        "pc": pc_sampler,
    }
    sampler_fn: Callable[..., Tensor] = SAMPLERS[args.sample_type]

    with torch.no_grad():
        raw_out = sampler_fn(score_fn, batch_size=args.n, device=device)

    # Extract first element if a tuple is returned
    raw_samples: Union[Tensor, Tuple[Tensor, ...]] = raw_out
    if isinstance(raw_samples, tuple):
        first = raw_samples[0]
    else:
        first = raw_samples
    samples_tensor: Tensor = first

    samples = samples_tensor.cpu().numpy()

    # Save with descriptive name
    target = "lorenz"
    default_name = f"{args.sample_type}_{target}_samples.npz"
    outfile = pathlib.Path(args.outfile or default_name)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outfile, samples=samples)
    print("Saved", args.n, "samples to", outfile)


if __name__ == "__main__":
    main()
