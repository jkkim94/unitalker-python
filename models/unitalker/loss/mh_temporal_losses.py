"""Optional temporal / closure losses for MetaHuman controls.

NOT enabled by default. Pred velocity is typically BELOW GT (vel_ratio≈0.4–0.5),
so plain smoothing is discouraged. Use only after MH_REVISED_AUDIT diagnosis.

These helpers are additive; they do not modify existing checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def velocity_match_loss(pred: torch.Tensor, target: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
    """MSE on first differences. pred/target: (B,T,C)."""
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    if idx is not None:
        pred = pred.index_select(-1, idx)
        target = target.index_select(-1, idx)
    return nn.functional.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def acceleration_match_loss(pred: torch.Tensor, target: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_zeros(())
    if idx is not None:
        pred = pred.index_select(-1, idx)
        target = target.index_select(-1, idx)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return nn.functional.mse_loss(dp[:, 1:] - dp[:, :-1], dt[:, 1:] - dt[:, :-1])


def lip_closure_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    jaw_or_aperture_idx: int,
    close_thr: float = 0.05,
    margin: float = 0.02,
) -> torch.Tensor:
    """When GT channel is near-closed, penalize pred staying open.

    Uses a single control (e.g. jawOpen or lipsTogether) as aperture proxy —
    not geometric lip distance. Prefer RigLogic aperture when available.
    """
    g = target[..., jaw_or_aperture_idx]
    p = pred[..., jaw_or_aperture_idx]
    mask = (g <= close_thr).float()
    # hinge: pred should be <= close_thr + margin
    penalty = torch.relu(p - (close_thr + margin))
    denom = mask.sum().clamp_min(1.0)
    return (penalty * mask).sum() / denom
