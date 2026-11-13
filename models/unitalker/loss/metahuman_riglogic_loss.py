"""MetaHuman loss mirroring UniTalker Vanilla BlendShapeLoss (control + vertex rec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
# RigLogic linear basis for vertex rec_loss (controls @ blend_shape + mean_shape).
# Column count must match dataset annot_dim (D8=260).
DEFAULT_BASIS_PATH_D8_260 = ROOT / "data/shared_resources/metahuman_D8_control260_riglogic_basis.npz"
DEFAULT_BASIS_PATH_BY_ANNOT_DIM = {
    260: DEFAULT_BASIS_PATH_D8_260,
}
DEFAULT_BASIS_PATH = DEFAULT_BASIS_PATH_D8_260


def load_riglogic_basis(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"MetaHuman RigLogic basis not found: {p}")
    data = np.load(p, allow_pickle=False)
    vertex_count = int(data["vertex_count"][0]) if "vertex_count" in data else None
    if vertex_count == 11977:
        raise ValueError(
            f"RigLogic basis {p} is LOD1 (11977 verts). Re-bake LOD0: "
            "python util/bake_metahuman_riglogic_basis.py --lod 0 --meshes head"
        )
    mouth_idx = data["mouth_idx"]
    if mouth_idx.size == 0:
        mouth_idx = None
    return {
        "mean_shape": torch.from_numpy(data["mean_shape"].astype(np.float32)),
        "blend_shape": torch.from_numpy(data["blend_shape"].astype(np.float32)),
        "mouth_idx": torch.from_numpy(mouth_idx.astype(np.int64)) if mouth_idx is not None else None,
        "path": str(p),
        "vertex_count": int(data["vertex_count"][0]) if "vertex_count" in data else None,
    }


class MetahumanBlendShapeLoss(nn.Module):
    """Vanilla BlendShapeLoss와 동형 (UniTalker 논문 rec + λ·bs).

    rec_L2_loss  = MSE(RigLogic(pred), RigLogic(gt))  — vertex 공간, backward O
    blendshape_loss = MSE(pred_control, gt_control)   — control 공간
    total = rec + bs_beta * bs  (bs_beta = config blendshape_weight)

    blend_shape/mean_shape는 register_buffer → 학습 안 함, grad는 control 쪽으로만.
    rec 경로 grad: ∂L/∂control = (verts_err) @ blend_shape.T
    """

    def __init__(
        self,
        mean_shape: torch.Tensor,
        blend_shape: torch.Tensor,
        mouth_idx: torch.Tensor | None,
        bs_beta: float = 0.0,
        channel_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.bs_beta = bs_beta
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.register_buffer("mean_shape", mean_shape)
        self.register_buffer("blend_shape", blend_shape)
        if channel_mask is not None:
            self.register_buffer("channel_mask", channel_mask)
        else:
            self.channel_mask = None
        if mouth_idx is not None and mouth_idx.numel() > 0:
            self.register_buffer("mouth_idx", mouth_idx.long())
        else:
            self.mouth_idx = None

    def _control_mse(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """blendshape_loss (W&B). x,target: [B,T,C]. GT target는 grad 없음."""
        if self.channel_mask is None or x.shape[-1] != self.channel_mask.shape[0]:
            return self.mse(x, target)
        mask = self.channel_mask.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        diff_sq = (x - target) ** 2
        denom = mask.sum() * diff_sq.shape[0] * diff_sq.shape[1]
        if denom <= 0:
            return self.mse(x, target)
        return (diff_sq * mask).sum() / denom

    def controls_to_vertices(self, x: torch.Tensor) -> torch.Tensor:
        """Linear RigLogic decode (미분 가능). [B,T,C] @ [C,V*3] → [B,T,V*3]. D8 LOD0: C=200, V=24049."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        verts = flat @ self.blend_shape.to(flat.device) + self.mean_shape.to(flat.device)
        return verts.reshape(*shape[:-1], -1)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bs_loss = self._control_mse(x, target)
        rec_loss = self.mse(self.controls_to_vertices(x), self.controls_to_vertices(target))
        return rec_loss + self.bs_beta * bs_loss

    def forward_separate(self, x: torch.Tensor, target: torch.Tensor):
        """train_eval_loop에서 호출. returns (total, rec_L2, blendshape_loss) — 로깅용 분리."""
        bs_loss = self._control_mse(x, target)
        rec_loss = self.mse(self.controls_to_vertices(x), self.controls_to_vertices(target))
        total = rec_loss + self.bs_beta * bs_loss
        return total, rec_loss, bs_loss

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        """LVE: lip vertex별 L2 → 프레임별 max → mean. train: mouth_loss, val: LVE/LVD."""
        target_v = self.controls_to_vertices(target)
        x_v = self.controls_to_vertices(x)
        if self.mouth_idx is None:
            l2 = self.mse(x_v, target_v)
            l1 = self.l1(x_v, target_v)
            return l2, l1
        idx = self.mouth_idx.to(device=x.device)
        xv = x_v.reshape(*x_v.shape[:-1], -1, 3)
        tv = target_v.reshape(*target_v.shape[:-1], -1, 3)
        metric_l2 = (tv[..., idx, :] - xv[..., idx, :]) ** 2
        metric_l2 = metric_l2.sum(-1)
        metric_l2 = metric_l2.max(-1)[0]
        metric_l2norm = metric_l2**0.5
        return metric_l2.mean(), metric_l2norm.mean()

    def get_vertices(self, x: torch.Tensor):
        return self.controls_to_vertices(x).reshape(*x.shape[:-1], -1, 3)
