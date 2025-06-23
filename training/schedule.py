
"""Utilities for time-dependent coefficients in VP SDE."""
from __future__ import annotations
import torch


def linear_beta_schedule(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    return beta_min + t * (beta_max - beta_min)


def integral_beta(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2


def sigma(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    return torch.sqrt(1.0 - torch.exp(-integral_beta(t, beta_min, beta_max)))


def m_t(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    return torch.exp(-0.5 * integral_beta(t, beta_min, beta_max))
