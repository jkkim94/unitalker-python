import numpy as np
import pickle
import torch
import os
from pathlib import Path
from dataclasses import dataclass
from smplx.lbs import lbs
from smplx.utils import Struct, to_np, to_tensor
from torch import nn

from .metahuman_lip_mask import (
    load_feature_names_for_metahuman,
    load_lip_control_indices,
    load_lip_vertex_ids_csv,
    resolve_lip_vertex_ids_for_lve,
    resolve_mask_path,
)
from .metahuman_riglogic_loss import (
    DEFAULT_BASIS_PATH_BY_ANNOT_DIM,
    MetahumanBlendShapeLoss,
    load_riglogic_basis,
)

# 프로젝트 루트 기준 resources 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESOURCES_PATH = PROJECT_ROOT / "data" / "shared_resources"


@dataclass
class FlameParams:
    betas: torch.Tensor
    jaw: torch.Tensor
    eyeballs: torch.Tensor
    neck: torch.Tensor


class FlameLayer(nn.Module):

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.dtype = torch.float32
        with open(model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(
            to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        self.register_buffer(
            'v_template',
            to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype),
        )
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs,
                              [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer(
            'lbs_weights',
            to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))
        default_neck_pose = torch.zeros([1, 3],
                                        dtype=self.dtype,
                                        requires_grad=False)
        self.register_parameter(
            'neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        default_eyeball_pose = torch.zeros([1, 6],
                                           dtype=self.dtype,
                                           requires_grad=False)
        self.register_parameter(
            'eyeballs',
            nn.Parameter(default_eyeball_pose, requires_grad=False))
        default_jaw = torch.zeros([1, 3],
                                  dtype=self.dtype,
                                  requires_grad=False)
        self.register_parameter('jaw',
                                nn.Parameter(default_jaw, requires_grad=False))

        self.FLAME_CONSTS = {
            'betas': 400,
            'rotation': 6,
            'jaw': 3,
            'eyeballs': 0,
            'neck': 0,
            'translation': 3,
            'scale': 1,
        }

    def flame_params_from_3dmm(self,
                               tensor_3dmm: torch.Tensor,
                               zero_expr: bool = False):
        constants = self.FLAME_CONSTS

        assert tensor_3dmm.ndim == 2

        cur_index: int = 0
        betas = tensor_3dmm[:, :constants['betas']]
        cur_index += constants['betas']
        jaw = tensor_3dmm[:, cur_index:cur_index + constants['jaw']]
        cur_index += constants['jaw']
        cur_index += constants['rotation']
        eyeballs = tensor_3dmm[:, cur_index:cur_index + constants['eyeballs']]
        cur_index += constants['eyeballs']
        neck = tensor_3dmm[:, cur_index:cur_index + constants['neck']]
        cur_index += constants['neck']
        cur_index += constants['translation']
        cur_index += constants['scale']
        return FlameParams(betas=betas, jaw=jaw, eyeballs=eyeballs, neck=neck)

    def forward(self, tensor_3dmm: torch.Tensor):
        bs = tensor_3dmm.shape[0]
        flame_params = self.flame_params_from_3dmm(tensor_3dmm)
        betas = flame_params.betas
        rotation = torch.zeros([bs, 3], device=tensor_3dmm.device)
        neck_pose = flame_params.neck if not (
            0 in flame_params.neck.shape) else self.neck_pose[[0]].expand(
                bs, -1)
        eyeballs = flame_params.eyeballs if not (
            0 in flame_params.eyeballs.shape) else self.eyeballs[[0]].expand(
                bs, -1)
        jaw = flame_params.jaw if not (
            0 in flame_params.jaw.shape) else self.jaw[[0]].expand(bs, -1)

        full_pose = torch.cat([rotation, neck_pose, jaw, eyeballs], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(bs, 1, 1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        return vertices


class FlameParamLossForDADHead(nn.Module):

    def __init__(self, mouth_indices_path: str, loss_mask_path: str,
                 flame_model_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        loss_indices = np.load(loss_mask_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        self.register_buffer('loss_indices', torch.from_numpy(loss_indices))
        self.mse = nn.MSELoss()
        self.flame_layer = FlameLayer(flame_model_path)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        x = x.reshape((ori_shape[0], ori_shape[1], -1, 3))
        target = target.reshape((ori_shape[0], ori_shape[1], -1, 3))
        return self.mse(x[:, :, self.loss_indices], target[:, :,
                                                           self.loss_indices])

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.flame_layer(x.reshape(-1, x.shape[-1]))
        return x[:, self.loss_indices]

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        x = self.flame_layer(x.reshape(-1, ori_shape[-1]))
        target = self.flame_layer(target.reshape(-1, ori_shape[-1]))
        metric_L2 = (target[:, self.mouth_idx] - x[:, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class VerticesLoss(nn.Module):

    def __init__(self, mouth_indices_path: str) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class BlendShapeLoss(nn.Module):

    def __init__(self,
                 mouth_indices_path: str,
                 blendshape_path: str,
                 bs_beta: float = 0.0,
                 components_num: int = None,
                 mouth_bs_weight: float = 1.0,
                 temporal_weight: float = 0.0) -> None:
        super().__init__()
        mouth_indices = np.load(mouth_indices_path)
        blendshape = np.load(blendshape_path)
        self.register_buffer('mouth_idx', torch.from_numpy(mouth_indices))
        mean_shape = blendshape['meanshape']
        mean_shape = mean_shape.reshape(-1).astype(np.float32)
        blend_shape = blendshape['blendshape']
        blend_shape = blend_shape.reshape(len(blend_shape),
                                          -1).astype(np.float32)
        self.register_buffer('mean_shape', torch.from_numpy(mean_shape))
        self.register_buffer('blend_shape', torch.from_numpy(blend_shape))
        self.bs_beta = bs_beta
        self.mse = nn.MSELoss()
        self.components_num = components_num
        self.mouth_bs_weight = mouth_bs_weight
        self.temporal_weight = temporal_weight

        if mouth_bs_weight != 1.0:
            self._build_mouth_channel_mask(blendshape, mouth_indices)

    def _build_mouth_channel_mask(self, blendshape_data, mouth_indices):
        """BS 기저 행렬에서 입 영역 에너지 비율이 높은 채널에 가중치 부여"""
        B_raw = blendshape_data['blendshape']
        n_bs = B_raw.shape[0]
        n_vertices = B_raw.shape[1]
        mouth_energy = np.array([np.sum(B_raw[i, mouth_indices]**2) for i in range(n_bs)])
        total_energy = np.array([np.sum(B_raw[i]**2) for i in range(n_bs)])
        mouth_ratio = mouth_energy / (total_energy + 1e-12)
        channel_weights = np.ones(n_bs, dtype=np.float32)
        channel_weights[mouth_ratio > 0.5] = self.mouth_bs_weight
        self.register_buffer('channel_weights', torch.from_numpy(channel_weights))

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        bs_loss = self._weighted_bs_loss(x, target)
        rec_loss = self.mse(self.bs2vertices(x), self.bs2vertices(target))
        total = self.bs_beta * bs_loss + rec_loss
        if self.temporal_weight > 0 and x.shape[1] > 1:
            total = total + self.temporal_weight * self._temporal_loss(x, target)
        return total

    def _weighted_bs_loss(self, x: torch.Tensor, target: torch.Tensor):
        if self.mouth_bs_weight == 1.0:
            return self.mse(x, target)
        diff_sq = (x - target) ** 2
        weights = self.channel_weights.unsqueeze(0).unsqueeze(0)
        return (diff_sq * weights).mean()

    def _temporal_loss(self, x: torch.Tensor, target: torch.Tensor):
        """속도 + 가속도 일관성 손실"""
        vel_pred = x[:, 1:] - x[:, :-1]
        vel_gt = target[:, 1:] - target[:, :-1]
        vel_loss = self.mse(vel_pred, vel_gt)
        if x.shape[1] > 2:
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
            acc_loss = self.mse(acc_pred, acc_gt)
            return vel_loss + 0.5 * acc_loss
        return vel_loss

    def forward_separate(self, x: torch.Tensor, target: torch.Tensor):
        """분리된 loss 반환: (total_loss, rec_loss, blendshape_loss)"""
        bs_loss = self._weighted_bs_loss(x, target)
        rec_loss = self.mse(self.bs2vertices(x), self.bs2vertices(target))
        total_loss = self.bs_beta * bs_loss + rec_loss
        if self.temporal_weight > 0 and x.shape[1] > 1:
            total_loss = total_loss + self.temporal_weight * self._temporal_loss(x, target)
        return total_loss, rec_loss, bs_loss

    def bs2vertices(self, x: torch.Tensor):
        return x[:, :self.components_num] @ \
            self.blend_shape[:self.components_num] + \
            self.mean_shape

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        x = self.bs2vertices(x)
        return x.reshape(-1, x.shape[-1] // 3, 3)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        target = self.bs2vertices(target)
        x = self.bs2vertices(x)
        ori_shape = x.shape
        target = target.reshape(*ori_shape[:-1], -1, 3)
        x = x.reshape(*ori_shape[:-1], -1, 3)
        metric_L2 = (target[:, :, self.mouth_idx] - x[:, :, self.mouth_idx])**2
        metric_L2 = metric_L2.sum(-1)
        metric_L2 = metric_L2.max(-1)[0]
        metric_L2norm = metric_L2**0.5
        return metric_L2.mean(), metric_L2norm.mean()


class ParamLoss(nn.Module):

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
        self.L1 = nn.L1Loss()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        return self.weight * self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target), self.L1(x, target)

    def get_vertices(
        self,
        x: torch.Tensor,
    ):
        return x.squeeze(0)


ANNOT_TYPES_WITHOUT_LVE = frozenset({"metahuman_control"})


def resolve_metahuman_riglogic_basis_path(args, annot_dim: int | None = None) -> Path | None:
    """RigLogic basis npz for vertex rec_loss. Must match metahuman annot_dim (260).

    D8 r2 (Unreal ONNX) used control MSE only. Set use_metahuman_riglogic_rec=False
    to disable vertex rec even if a default 260ch basis file exists.
    """
    if not bool(getattr(args, "use_metahuman_riglogic_rec", True)):
        return None
    explicit = getattr(args, "metahuman_riglogic_basis_path", None)
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.is_file() else None
    if annot_dim is None:
        from models.unitalker.dataset.dataset_config import dataset_config
        for dataset_name in getattr(args, "dataset", []):
            cfg = dataset_config.get(dataset_name, {})
            if cfg.get("annot_type") == "metahuman_control":
                annot_dim = cfg.get("annot_dim")
                break
    if annot_dim in DEFAULT_BASIS_PATH_BY_ANNOT_DIM:
        p = DEFAULT_BASIS_PATH_BY_ANNOT_DIM[annot_dim]
        if p.is_file():
            return p
    return None


def get_annot_types_without_lve(args) -> frozenset[str]:
    """Skip metahuman LVE unless lve_weight>0 and MH-LVE-LIP-v1 CSV mask is set."""
    blocked = set()
    lve_weight = float(getattr(args, "lve_weight", 0.0))
    if lve_weight <= 0:
        blocked.add("metahuman_control")
        return frozenset(blocked)
    mask_path = resolve_mask_path(getattr(args, "metahuman_lip_vertex_mask", None))
    if mask_path is None:
        blocked.add("metahuman_control")
    return frozenset(blocked)


METADATA_EXCLUDE_NAMES = frozenset({"DisableFaceOverride", "MHFDSVersion"})
DEFAULT_HEAD_POSE_EXCLUDE_FILE = "metahuman_drop_head_pose.txt"


def _load_name_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_metahuman_channel_mask(feature_names: list[str], exclude_names: list[str]) -> torch.Tensor:
    exclude = {n.lower() for n in exclude_names}
    exclude |= {n.lower() for n in METADATA_EXCLUDE_NAMES}
    mask = torch.ones(len(feature_names), dtype=torch.float32)
    for i, name in enumerate(feature_names):
        if name.lower() in exclude:
            mask[i] = 0.0
    return mask


def load_metahuman_channel_mask(args, annot_dim: int) -> torch.Tensor | None:
    exclude_file = getattr(args, "metahuman_loss_exclude_head_pose", DEFAULT_HEAD_POSE_EXCLUDE_FILE)
    exclude_path = Path(exclude_file)
    if not exclude_path.is_absolute():
        exclude_path = PROJECT_ROOT / exclude_path
    exclude_names = _load_name_list(exclude_path)
    feature_names = load_feature_names_for_metahuman(args, annot_dim)
    if feature_names is None or len(feature_names) != annot_dim:
        return None
    return build_metahuman_channel_mask(feature_names, exclude_names)


def load_dataset_channel_masks(args, annot_dim: int) -> dict[str, torch.Tensor]:
    """dataset_config의 loss_channel_mask (dataset 디렉토리 기준 npy) → {dataset 이름: mask}."""
    from models.unitalker.dataset.dataset_config import dataset_config
    masks = {}
    for dataset_name in getattr(args, "dataset", []):
        cfg = dataset_config.get(dataset_name, {})
        rel = cfg.get("loss_channel_mask")
        if not rel or cfg.get("annot_dim") != annot_dim:
            continue
        path = Path(args.data_root) / cfg["dirname"] / rel
        mask = torch.from_numpy(np.load(path).astype(np.float32))
        if mask.numel() != annot_dim:
            raise ValueError(f"{path}: mask size {mask.numel()} != annot_dim {annot_dim}")
        masks[dataset_name] = mask
    return masks


def init_metahuman_lve_from_args(args) -> dict:
    info: dict = {}
    mask_path = resolve_mask_path(getattr(args, "metahuman_lip_vertex_mask", None))
    if mask_path is None:
        return info
    vertex_ids, meta = load_lip_vertex_ids_csv(mask_path)
    info["lip_vertex_count"] = int(vertex_ids.shape[0])
    info["lip_vertex_mask_path"] = str(mask_path)
    info["lip_vertex_mask_meta"] = meta
    info["lip_vertex_id_min"] = int(vertex_ids.min())
    info["lip_vertex_id_max"] = int(vertex_ids.max())
    args.metahuman_lip_vertex_count = info["lip_vertex_count"]
    args.metahuman_lip_vertex_meta = meta
    return info


class MetahumanControlLoss(nn.Module):
    """MetaHuman control loss with optional head-pose mask and lipsync LVE channels.

    dataset_masks: dataset 이름별 추가 채널 마스크 (예: D10 변환 데이터는 매핑된 채널만 학습).
    """

    accepts_dataset_name = True

    def __init__(
        self,
        bs_beta: float = 0.0,
        channel_mask: torch.Tensor | None = None,
        lip_control_indices: torch.Tensor | None = None,
        dataset_masks: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.bs_beta = bs_beta
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        if channel_mask is not None:
            self.register_buffer("channel_mask", channel_mask)
        else:
            self.channel_mask = None
        if lip_control_indices is not None:
            self.register_buffer("lip_control_indices", lip_control_indices.long())
        else:
            self.lip_control_indices = None
        self.dataset_mask_names = []
        for name, mask in (dataset_masks or {}).items():
            if channel_mask is not None:
                mask = mask * channel_mask
            self.register_buffer(f"dataset_mask_{name}", mask.float())
            self.dataset_mask_names.append(name)

    def _resolve_mask(self, dataset_name: str | None) -> torch.Tensor | None:
        if dataset_name is not None and dataset_name in self.dataset_mask_names:
            return getattr(self, f"dataset_mask_{dataset_name}")
        return self.channel_mask

    def _masked_mse(self, x: torch.Tensor, target: torch.Tensor,
                    dataset_name: str | None = None) -> torch.Tensor:
        channel_mask = self._resolve_mask(dataset_name)
        if channel_mask is None or x.shape[-1] != channel_mask.shape[0]:
            return self.mse(x, target)
        mask = channel_mask.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        diff_sq = (x - target) ** 2
        denom = mask.sum() * diff_sq.shape[0] * diff_sq.shape[1]
        if denom <= 0:
            return self.mse(x, target)
        return (diff_sq * mask).sum() / denom

    def forward(self, x: torch.Tensor, target: torch.Tensor, dataset_name: str | None = None):
        return self._masked_mse(x, target, dataset_name)

    def forward_separate(self, x: torch.Tensor, target: torch.Tensor, dataset_name: str | None = None):
        rec = self._masked_mse(x, target, dataset_name)
        # Vanilla BlendShapeLoss: rec=vertex, bs=weight MSE. MetaHuman: bs=lip control subset.
        if self.lip_control_indices is not None:
            bs = self._lip_subset_mse(x, target)
        else:
            bs = rec
        total = rec + self.bs_beta * bs
        return total, rec, bs

    def _lip_subset_mse(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lip_control_indices is None:
            return self._masked_mse(x, target)
        idx = self.lip_control_indices.to(device=x.device)
        return self.mse(x.index_select(-1, idx), target.index_select(-1, idx))

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor):
        """Lip-control MSE and MAE — NOT vertex LVE.

        train_eval_loop logs these as mouth_control_mse / mouth_control_mae.
        Do not treat return values as academic Lip Vertex Error.
        """
        l2 = self._lip_subset_mse(x, target)
        if self.lip_control_indices is not None:
            idx = self.lip_control_indices.to(device=x.device)
            l1 = self.l1(x.index_select(-1, idx), target.index_select(-1, idx))
        else:
            l1 = self.l1(x, target)
        return l2, l1

    def get_vertices(self, x: torch.Tensor):
        return x.squeeze(0)


class UniTalkerLoss(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        loss_config = {
            'flame_params_from_dadhead': {
                'class': FlameParamLossForDADHead,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '02_flame_mouth_idx.npy'),
                    'loss_mask_path':
                    str(RESOURCES_PATH / 'binary_resources' / 'flame_humanface_index.npy'),
                    'flame_model_path': str(RESOURCES_PATH / 'binary_resources' / 'flame.pkl'),
                }
            },
            '3DETF_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '03_emo_talk_mouth_idx.npy'),
                    'blendshape_path':
                    str(RESOURCES_PATH / 'binary_resources' / 'EmoTalk.npz'),
                    'bs_beta': args.blendshape_weight,
                    'mouth_bs_weight': getattr(args, 'mouth_bs_weight', 1.0),
                    'temporal_weight': getattr(args, 'temporal_weight', 0.0),
                },
            },
            'inhouse_blendshape_weight': {
                'class': BlendShapeLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '05_inhouse_arkit_mouth_idx.npy'),
                    'blendshape_path':
                    str(RESOURCES_PATH / 'binary_resources' / 'inhouse_arkit.npz'),
                    'bs_beta': args.blendshape_weight,
                    'mouth_bs_weight': getattr(args, 'mouth_bs_weight', 1.0),
                    'temporal_weight': getattr(args, 'temporal_weight', 0.0),
                },
            },
            'FLAME_5023_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '02_flame_mouth_idx.npy'),
                },
            },
            'BIWI_23370_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '04_BIWI_mouth_idx.npy'),
                },
            },
            'FLAME_SUB_2055_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / 'mouth_idx_FLAME_5023_vertices_wo_eyes_wo_head.npy'),
                },
            },
            'meshtalk_6172_vertices': {
                'class': VerticesLoss,
                'args': {
                    'mouth_indices_path':
                    str(RESOURCES_PATH / 'binary_resources' / '06_mesh_talk_mouth_idx.npy'),
                },
            },
            '24_viseme': {
                'class': ParamLoss,
                'args': {
                    'weight': args.blendshape_weight
                },
            }
        }

        loss_module_dict = {}
        for k, v in loss_config.items():
            loss_module = v['class'](**v['args'])
            loss_module_dict[k] = loss_module

        bs_beta = float(getattr(args, "blendshape_weight", 0.0))
        metahuman_mask = None
        metahuman_lip_indices = None
        mh_dim = None
        from models.unitalker.dataset.dataset_config import dataset_config

        for dataset_name in getattr(args, "dataset", []):
            cfg = dataset_config.get(dataset_name, {})
            if cfg.get("annot_type") == "metahuman_control":
                mh_dim = cfg.get("annot_dim")
                break

        if mh_dim == 260 and getattr(args, "metahuman_loss_exclude_head_pose", True):
            metahuman_mask = load_metahuman_channel_mask(args, 260)

        self.metahuman_lve_info = init_metahuman_lve_from_args(args)
        if self.metahuman_lve_info and mh_dim is not None:
            feature_names = load_feature_names_for_metahuman(args, mh_dim)
            if feature_names:
                use_all = bool(getattr(args, "metahuman_lip_control_use_all_lipsync", True))
                metahuman_lip_indices = load_lip_control_indices(
                    feature_names, use_all_lipsync_channels=use_all)
                self.metahuman_lve_info["lip_control_channel_count"] = int(
                    metahuman_lip_indices.numel())

        dataset_masks = load_dataset_channel_masks(args, mh_dim) if mh_dim is not None else {}

        basis_path = resolve_metahuman_riglogic_basis_path(args, annot_dim=mh_dim)
        if basis_path is not None and dataset_masks:
            raise ValueError("loss_channel_mask datasets require use_metahuman_riglogic_rec: false")
        # RigLogic basis 있으면 Vanilla BlendShapeLoss 패턴 (rec=vertex, bs=control)
        # 없으면 control MSE만 (MetahumanControlLoss)
        if basis_path is not None and mh_dim is not None:
            basis = load_riglogic_basis(basis_path)
            n_basis_cols = basis["blend_shape"].shape[0]
            if n_basis_cols == mh_dim:
                vertex_count = basis["vertex_count"] or (
                    basis["mean_shape"].numel() // 3
                )
                lve_mouth_idx = None
                lip_ids = resolve_lip_vertex_ids_for_lve(args, vertex_count)
                if lip_ids is not None:
                    lve_mouth_idx = torch.from_numpy(lip_ids)
                    self.metahuman_lve_info["lip_vertex_count_on_rec_mesh"] = int(
                        lip_ids.shape[0])
                    self.metahuman_lve_info["rec_mesh_vertex_count"] = int(vertex_count)
                    args.metahuman_lip_vertex_count_on_rec = int(lip_ids.shape[0])
                    args.metahuman_riglogic_mouth_vertex_count = int(lip_ids.shape[0])
                loss_module_dict["metahuman_control"] = MetahumanBlendShapeLoss(
                    mean_shape=basis["mean_shape"],
                    blend_shape=basis["blend_shape"],
                    mouth_idx=lve_mouth_idx,
                    bs_beta=bs_beta,
                    channel_mask=metahuman_mask,
                )
                self.metahuman_lve_info["riglogic_basis_path"] = str(basis_path)
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "RigLogic basis %s has %d columns but metahuman annot_dim=%d; "
                    "using MetahumanControlLoss (no vertex rec). "
                    "Bake matching basis: util/bake_metahuman_riglogic_basis.py",
                    basis_path,
                    n_basis_cols,
                    mh_dim,
                )
                loss_module_dict["metahuman_control"] = MetahumanControlLoss(
                    bs_beta=bs_beta,
                    channel_mask=metahuman_mask,
                    lip_control_indices=metahuman_lip_indices,
                    dataset_masks=dataset_masks,
                )
        else:
            if mh_dim is not None:
                import logging
                _log = logging.getLogger(__name__)
                if not bool(getattr(args, "use_metahuman_riglogic_rec", True)):
                    _log.info(
                        "use_metahuman_riglogic_rec=False → MetahumanControlLoss "
                        "(control MSE only, D8 r2 / Unreal ONNX style)"
                    )
                else:
                    _log.warning(
                        "No RigLogic basis for metahuman annot_dim=%d; "
                        "using MetahumanControlLoss (control MSE only, no vertex rec). "
                        "Expected: %s",
                        mh_dim,
                        DEFAULT_BASIS_PATH_BY_ANNOT_DIM.get(mh_dim, "custom path via metahuman_riglogic_basis_path"),
                    )
            loss_module_dict["metahuman_control"] = MetahumanControlLoss(
                bs_beta=bs_beta,
                channel_mask=metahuman_mask,
                lip_control_indices=metahuman_lip_indices,
                dataset_masks=dataset_masks,
            )
        self.dataset_mask_info = {k: int(v.sum().item()) for k, v in dataset_masks.items()}

        self.loss_module_dict = nn.ModuleDict(loss_module_dict)
        self.mse = nn.MSELoss()
        return

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        annot_type: str,
    ):
        return self.loss_module_dict[annot_type](x, target)

    def pca_loss(self, x: torch.Tensor, target: torch.Tensor):
        return self.mse(x, target)

    def mouth_metric(self, x: torch.Tensor, target: torch.Tensor,
                     annot_type: str):
        return self.loss_module_dict[annot_type].mouth_metric(x, target)

    def get_vertices(self, x: torch.Tensor, annot_type: str):
        return self.loss_module_dict[annot_type].get_vertices(x)
