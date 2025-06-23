"""Probability flow ODE sampler."""

from __future__ import annotations

from typing import Union

import torch
from torchdiffeq import odeint

from training.schedule import linear_beta_schedule


def ode_sampler(
    score_fn,
    batch_size: int = 64,
    eps: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    atol=1e-5,
    rtol=1e-5,
):
    def drift(t, x):
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        beta = linear_beta_schedule(t_tensor)[:, None]
        return -0.5 * beta * x - 0.5 * beta * score_fn(x, t_tensor.squeeze())

    t_span = torch.tensor([1.0, eps], device=device)
    x0 = torch.randn(batch_size, 3, device=device)
    out = odeint(lambda t, y: drift(t, y), x0, t_span, atol=atol, rtol=rtol)
    return out[-1]
