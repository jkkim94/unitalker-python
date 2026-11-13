"""Vertex and control metrics (MH-LVE-LIP-v1 CSV lip mask for LVE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_lip_mask_csv(
    lip_csv: str | Path,
    vertex_count: int,
) -> np.ndarray:
    from models.unitalker.loss.metahuman_lip_mask import (
        filter_lip_vertex_ids,
        load_lip_vertex_ids_csv,
    )

    ids, _ = load_lip_vertex_ids_csv(lip_csv)
    return filter_lip_vertex_ids(ids, vertex_count)


def load_lip_mask(lip_json: str | Path, meshes: str = "face") -> np.ndarray:
    """Legacy MH-LVR JSON mask (eval only; training LVE uses CSV)."""
    with open(lip_json, encoding="utf-8-sig") as f:
        lip = json.load(f)
    mask = np.asarray(lip["lip"], dtype=np.int64)
    if meshes == "head":
        return mask
    head_count = int(lip.get("headVertexCount", 11977))
    if meshes == "face":
        return mask
    return mask


def compute_control_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    column_names: list[str] | None = None,
) -> dict[str, Any]:
    diff = np.abs(gt - pred)
    per_frame_mae = diff.mean(axis=1)
    per_column_mae = diff.mean(axis=0)
    out: dict[str, Any] = {
        "control_mae_mean": float(per_frame_mae.mean()),
        "control_mae_std": float(per_frame_mae.std()),
        "control_mae_max_frame": float(per_frame_mae.max()),
        "frame_count": int(gt.shape[0]),
        "column_count": int(gt.shape[1]),
    }
    if column_names is not None and len(column_names) == gt.shape[1]:
        out["per_column_mae"] = {
            column_names[i]: float(per_column_mae[i]) for i in range(len(column_names))
        }
    return out


def compute_vertex_metrics(
    gt_verts: np.ndarray,
    pred_verts: np.ndarray,
    lip_mask: np.ndarray,
    protocol: str = "MH-LVE-LIP-v1",
) -> dict[str, Any]:
    """gt_verts, pred_verts: (F, V, 3) in cm."""
    diff = gt_verts - pred_verts
    per_vert_l2 = np.linalg.norm(diff, axis=-1)
    per_frame_mean_l2 = per_vert_l2.mean(axis=1)
    per_frame_max_l2 = per_vert_l2.max(axis=1)

    lip_l2 = per_vert_l2[:, lip_mask]
    per_frame_lip_lve = lip_l2.mean(axis=1)

    return {
        "protocol": protocol,
        "vertex_mean_l2": float(per_frame_mean_l2.mean()),
        "vertex_mean_l2_std": float(per_frame_mean_l2.std()),
        "vertex_max_l2_mean": float(per_frame_max_l2.mean()),
        "vertex_max_l2_max": float(per_frame_max_l2.max()),
        "lip_lve_mean": float(per_frame_lip_lve.mean()),
        "lip_lve_std": float(per_frame_lip_lve.std()),
        "lip_lve_max_frame": float(per_frame_lip_lve.max()),
        "lip_vertex_count": int(lip_mask.shape[0]),
        "vertex_count": int(gt_verts.shape[1]),
        "frame_count": int(gt_verts.shape[0]),
        "per_frame_vertex_mean_l2": per_frame_mean_l2.tolist(),
        "per_frame_lip_lve": per_frame_lip_lve.tolist(),
    }


def summarize_batch(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"clip_count": 0}
    keys = [
        "control_mae_mean",
        "vertex_mean_l2",
        "lip_lve_mean",
    ]
    summary: dict[str, Any] = {"clip_count": len(reports), "clips": reports}
    for key in keys:
        vals = [r[key] for r in reports if key in r]
        if vals:
            summary[f"{key}_avg"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    return summary
