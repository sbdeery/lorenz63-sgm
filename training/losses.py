"""Continuous denoising score matching loss for VP SDE."""

from __future__ import annotations

import torch
from torch import nn

from training.schedule import m_t, sigma


def vp_score_matching(
    model: nn.Module, x0: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    B = x0.size(0)
    device = x0.device
    t = torch.rand(B, device=device) * (1.0 - eps) + eps
    sig = sigma(t)
    m = m_t(t)
    z = torch.randn_like(x0)
    xt = m[:, None] * x0 + sig[:, None] * z
    target = -(xt - m[:, None] * x0) / (sig[:, None] ** 2)
    score = model(xt, t)
    loss = ((score - target) ** 2).sum(dim=1) * (sig**2)
    return loss.mean()
