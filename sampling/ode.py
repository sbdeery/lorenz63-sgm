"""Probability flow ODE sampler."""

from __future__ import annotations

from typing import Callable, Union, cast

import torch
from torch import Tensor
from torchdiffeq import odeint  # type: ignore[import-untyped]

from training.schedule import linear_beta_schedule


def ode_sampler(
    score_fn: Callable[[Tensor, Tensor], Tensor],
    batch_size: int = 64,
    eps: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Tensor:
    """
    Probability flow ODE sampler for the VP SDE.
    """

    def drift(t: float, x: Tensor) -> Tensor:
        """
        Drift function for the PF-ODE: dx/dt = -½ β(t) x - ½ β(t) s_θ(x, t).
        """
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        beta = linear_beta_schedule(t_tensor)[:, None]
        return -0.5 * beta * x - 0.5 * beta * score_fn(x, t_tensor.squeeze())

    t_span: Tensor = torch.tensor([1.0, eps], device=device)
    x0: Tensor = torch.randn(batch_size, 3, device=device)
    # Pass the typed drift directly to odeint; suppress its untyped signature
    raw_out = odeint(drift, x0, t_span, atol=atol, rtol=rtol)
    out = cast(Tensor, raw_out)
    return out[-1]
