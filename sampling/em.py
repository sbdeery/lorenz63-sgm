"""Euler–Maruyama sampler for the reverse VP SDE."""

from __future__ import annotations

from typing import Callable, Union

import torch
from torch import Tensor

from training.schedule import linear_beta_schedule


def em_sampler(
    score_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int = 20000,
    batch_size: int = 64,
    eps: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    """
    Euler–Maruyama sampler for the reverse VP SDE.
    """
    ts = torch.linspace(1.0, eps, steps, device=device)
    x = torch.randn(batch_size, 3, device=device)
    for i in range(steps - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]

        # expand to (batch_size,)
        t_batch = t.expand(batch_size)

        beta = linear_beta_schedule(t)
        drift = -0.5 * beta * x - beta * score_fn(x, t_batch)
        diffusion = torch.sqrt(beta * (-dt))
        x = x + drift * dt + diffusion * torch.randn_like(x)
    return x
