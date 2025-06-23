"""Exponential Moving Average of parameters."""

from __future__ import annotations

import torch
from torch import nn


class EMA:
    """Maintain shadow copy of parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {
            k: p.detach().clone()
            for k, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        self.backup = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = {}
