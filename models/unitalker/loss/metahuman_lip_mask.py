"""MetaHuman lip vertex mask (CSV) and lipsync control channel indices for LVE."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIP_VERTEX_MASK_CSV = ROOT / "metahuman_head_lod0_lip_mask.csv"

# Lipsync control channels used for differentiable training LVE (control space).
LIP_CONTROL_KEYWORDS = (
    "mouth",
    "jaw",
    "tongue",
    "teeth",
    "lip",
)


def resolve_mask_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.is_file() else None


def load_lip_vertex_ids_csv(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Load lip vertex indices from metahuman_head_lod0_lip_mask.csv."""
    p = resolve_mask_path(path)
    if p is None:
        raise FileNotFoundError(f"MetaHuman lip vertex mask CSV not found: {path}")
    ids: list[int] = []
    meta: dict[str, str] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["vertex_id"]))
            if not meta:
                meta = {
                    k: row[k]
                    for k in ("region", "protocol", "mesh_name", "lod")
                    if k in row and row[k]
                }
    arr = np.unique(np.asarray(ids, dtype=np.int64))
    arr.sort()
    return arr, meta


def filter_lip_vertex_ids(vertex_ids: np.ndarray, vertex_count: int) -> np.ndarray:
    """Keep lip indices valid for a RigLogic mesh (LOD0 head V=24049)."""
    ids = np.asarray(vertex_ids, dtype=np.int64)
    valid = ids[(ids >= 0) & (ids < vertex_count)]
    return np.unique(valid)


def resolve_lip_vertex_ids_for_lve(
    args,
    vertex_count: int,
    mask_path: str | Path | None = None,
) -> np.ndarray | None:
    """MH-LVE-LIP-v1 CSV lip IDs filtered to rec mesh vertex count."""
    path = resolve_mask_path(mask_path or getattr(args, "metahuman_lip_vertex_mask", None))
    if path is None:
        path = DEFAULT_LIP_VERTEX_MASK_CSV if DEFAULT_LIP_VERTEX_MASK_CSV.is_file() else None
    if path is None:
        return None
    ids, _ = load_lip_vertex_ids_csv(path)
    filtered = filter_lip_vertex_ids(ids, vertex_count)
    if filtered.size == 0:
        return None
    return filtered


def load_lip_control_indices(
    feature_names: list[str],
    use_all_lipsync_channels: bool = True,
) -> torch.Tensor:
    """Indices into metahuman_control vector for lip-focused LVE (training).

    For D8 lipsync-only (200ch), all kept channels are lipsync-related -> use all.
    """
    if use_all_lipsync_channels or not feature_names:
        return torch.arange(len(feature_names), dtype=torch.long)
    idx = [
        i
        for i, name in enumerate(feature_names)
        if any(k in name.lower() for k in LIP_CONTROL_KEYWORDS)
    ]
    if not idx:
        return torch.arange(len(feature_names), dtype=torch.long)
    return torch.tensor(idx, dtype=torch.long)


def load_feature_names_for_metahuman(args, annot_dim: int) -> list[str] | None:
    from models.unitalker.dataset.dataset_config import dataset_config

    data_root = getattr(args, "data_root", None)
    if not data_root:
        return None
    for dataset_name in getattr(args, "dataset", []):
        cfg = dataset_config.get(dataset_name, {})
        if cfg.get("annot_type") != "metahuman_control":
            continue
        if cfg.get("annot_dim") != annot_dim:
            continue
        fn_path = Path(data_root) / cfg["dirname"] / "feature_names.txt"
        if fn_path.is_file():
            return fn_path.read_text(encoding="utf-8").strip().split("\n")
    return None
