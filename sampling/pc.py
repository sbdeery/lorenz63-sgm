"""Predictor-Corrector sampler for VP SDE."""

from __future__ import annotations

from typing import Union

import torch

from training.schedule import linear_beta_schedule


def langevin_step(x, t, score_fn, snr):
    noise = torch.randn_like(x)
    grad = score_fn(x, t)
    noise_norm = noise.norm(p=2, dim=1).mean()
    grad_norm = grad.norm(p=2, dim=1).mean()
    step = (snr * noise_norm / grad_norm) ** 2 * 2
    x = x + step * grad + torch.sqrt(2 * step) * noise
    return x


def pc_sampler(
    score_fn,
    steps: int = 5000,
    snr: float = 0.16,
    corrector_steps: int = 1,
    batch_size: int = 64,
    eps: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
):
    ts = torch.linspace(1.0, eps, steps, device=device)
    x = torch.randn(batch_size, 3, device=device)
    for i in range(steps - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]

        # expand to (batch_size,)
        t_batch = t.expand(batch_size)

        for _ in range(corrector_steps):
            x = langevin_step(x, t_batch, score_fn, snr)
        beta = linear_beta_schedule(t)
        drift = -0.5 * beta * x - beta * score_fn(x, t_batch)
        x = x + drift * dt + torch.sqrt(beta * (-dt)) * torch.randn_like(x)
    return x
