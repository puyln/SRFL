from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TorchUSGRResult:
    loss: Tensor
    coupling: Tensor
    random_samples: Tensor
    cost_matrix: Tensor


def _feature_tokens(features: Tensor, max_features: Optional[int]) -> Tensor:
    if features.ndim == 4:

        tokens = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    elif features.ndim == 3:

        tokens = features.reshape(-1, features.shape[-1])
    elif features.ndim == 2:
        tokens = features
    else:
        raise ValueError(
            "features must have shape [N,C], [B,N,C], or [B,C,H,W]"
        )
    if max_features is not None and tokens.shape[0] > max_features:


        index = torch.linspace(
            0, tokens.shape[0] - 1, max_features, device=tokens.device
        ).long()
        tokens = tokens.index_select(0, index)
    return F.normalize(tokens, dim=-1)


def _spherical_gaussian(count: int, dim: int, reference: Tensor) -> Tensor:
    samples = torch.randn(
        count, dim, device=reference.device, dtype=reference.dtype
    )
    return F.normalize(samples, dim=-1)


def _squared_distance(x: Tensor, y: Tensor) -> Tensor:
    return (
        x.square().sum(dim=1, keepdim=True)
        + y.square().sum(dim=1).unsqueeze(0)
        - 2.0 * x @ y.t()
    ).clamp_min(0.0)


class USGRLoss(nn.Module):


    def __init__(
        self,
        random_sample_count: int = 128,
        tau_alpha: float = 10.0,
        tau_beta: float = 10.0,
        transport_iterations: int = 50,
        transport_learning_rate: Optional[float] = None,
        max_features: Optional[int] = 1024,
    ) -> None:
        super().__init__()
        if random_sample_count <= 0 or transport_iterations <= 0:
            raise ValueError("sample count and transport iterations must be positive")
        if not (0 < tau_alpha < float('inf') and 0 < tau_beta < float('inf')):
            raise ValueError("marginal penalties must be finite and positive")
        if max_features is not None and max_features <= 0:
            raise ValueError("max_features must be positive or None")
        if transport_learning_rate is not None and not 0 < transport_learning_rate < float('inf'):
            raise ValueError("transport_learning_rate must be finite and positive")
        self.random_sample_count = random_sample_count
        self.tau_alpha = tau_alpha
        self.tau_beta = tau_beta
        self.transport_iterations = transport_iterations
        self.transport_learning_rate = transport_learning_rate
        self.max_features = max_features

    def forward(self, features: Tensor) -> TorchUSGRResult:
        if features.numel() == 0 or not torch.isfinite(features).all():
            raise ValueError("features must be non-empty and finite")
        z = _feature_tokens(features, self.max_features)
        r = _spherical_gaussian(
            self.random_sample_count, z.shape[1], reference=z
        )
        cost = _squared_distance(z, r)
        n, m = cost.shape
        alpha = torch.full((n,), 1.0 / n, device=z.device, dtype=z.dtype)
        beta = torch.full((m,), 1.0 / m, device=z.device, dtype=z.dtype)


        with torch.no_grad():
            gamma = torch.outer(alpha, beta)
            if self.transport_learning_rate is None:
                step = 0.9 / (
                    self.tau_alpha * m + self.tau_beta * n
                )
            else:
                step = self.transport_learning_rate
            detached_cost = cost.detach()
            for _ in range(self.transport_iterations):
                row_residual = gamma.sum(dim=1) - alpha
                col_residual = gamma.sum(dim=0) - beta
                gradient = (
                    detached_cost
                    + self.tau_alpha * row_residual[:, None]
                    + self.tau_beta * col_residual[None, :]
                )
                gamma = (gamma - step * gradient).clamp_min(0.0)

        row_residual = gamma.sum(dim=1) - alpha
        col_residual = gamma.sum(dim=0) - beta
        loss = (
            (cost * gamma).sum()
            + 0.5 * self.tau_alpha * row_residual.square().sum()
            + 0.5 * self.tau_beta * col_residual.square().sum()
        )
        return TorchUSGRResult(
            loss=loss,
            coupling=gamma,
            random_samples=r,
            cost_matrix=cost,
        )
