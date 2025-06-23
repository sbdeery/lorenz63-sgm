"""Residual MLP conditioned on time for score estimation."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Encode scalar time t with random Fourier features."""

    def __init__(self, embed_dim: int = 256, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(self.fc1(h))
        h = self.norm2(h)
        h = self.fc2(h)
        return cast(torch.Tensor, x + h)


class ScoreMLP(nn.Module):
    """Score network s_theta(x,t) -> ℝ³."""

    def __init__(self, hidden_dim: int = 128, num_blocks: int = 6) -> None:
        super().__init__()
        self.time_embed = GaussianFourierProjection(embed_dim=256)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc_in = nn.Linear(3, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.fc_out = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(self.time_embed(t))  # (B, H)
        h = self.fc_in(x) + emb
        for block in self.blocks:
            h = block(h + emb)  # FiLM-like conditioning
        return cast(torch.Tensor, self.fc_out(h))
