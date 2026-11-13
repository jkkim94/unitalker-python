#!/usr/bin/env python3
"""Bake linear RigLogic Head Decoder basis for differentiable vertex rec loss."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "riglogic"))
sys.path.insert(0, str(ROOT))


def bake_basis(
    dna_path: Path,
    feature_names_path: Path,
    output_path: Path,
    lod: int = 0,
    meshes: str = "head",
    eps: float = 0.01,
    lip_csv: Path | None = None,
    vertex_count_for_lip: int | None = None,
) -> dict:
    from face_model import FaceModel

    feature_names = feature_names_path.read_text(encoding="utf-8").strip().split("\n")
    fm = FaceModel(str(dna_path), lod=lod, meshes=meshes)
    fm.set_columns(feature_names)
    n_cols = len(feature_names)
    v0 = fm.eval_columns(np.zeros(n_cols, np.float32)).astype(np.float64)
    v_dim = v0.size
    mean_shape = v0.reshape(-1).astype(np.float32)

    blend_shape = np.zeros((n_cols, v_dim), dtype=np.float32)
    row = np.zeros(n_cols, np.float32)
    for j in range(n_cols):
        row[:] = 0.0
        row[j] = eps
        vj = fm.eval_columns(row).astype(np.float64).reshape(-1)
        blend_shape[j] = ((vj - mean_shape.astype(np.float64)) / eps).astype(np.float32)
        if j % 20 == 0:
            print(f"  baked {j}/{n_cols}", flush=True)

    mouth_idx = None
    v_count = vertex_count_for_lip or (v_dim // 3)
    if lip_csv and lip_csv.is_file():
        ids: list[int] = []
        with lip_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.append(int(row["vertex_id"]))
        if ids:
            arr = np.unique(np.asarray(ids, dtype=np.int64))
            mouth_idx = arr[(arr >= 0) & (arr < v_count)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean_shape=mean_shape,
        blend_shape=blend_shape,
        feature_names=np.array(feature_names),
        mouth_idx=mouth_idx if mouth_idx is not None else np.array([], dtype=np.int64),
        vertex_count=np.array([v_dim // 3], dtype=np.int64),
        lod=np.array([lod], dtype=np.int64),
        eps=np.array([eps], dtype=np.float32),
    )
    return {
        "columns": n_cols,
        "vertices": v_dim // 3,
        "mouth_idx": int(mouth_idx.shape[0]) if mouth_idx is not None else 0,
        "output": str(output_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dna",
        default=str(ROOT / "MIGRATE_DNA/0421_V33/head.dna"),
    )
    ap.add_argument(
        "--feature-names",
        default=str(ROOT / "data/unitalker_data/D8_metahuman_control/feature_names.txt"),
        help="260ch D8 full control layout (default).",
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "data/shared_resources/metahuman_D8_control260_riglogic_basis.npz"),
        help="Output npz; column count must match feature_names (260 for D8).",
    )
    ap.add_argument("--lod", type=int, default=0)
    ap.add_argument("--meshes", default="head")
    ap.add_argument(
        "--lip-csv",
        default=str(ROOT / "metahuman_head_lod0_lip_mask.csv"),
        help="MH-LVE-LIP-v1 lip mask CSV (optional; filtered to baked mesh vertex count)",
    )
    args = ap.parse_args()
    lip_csv = Path(args.lip_csv) if args.lip_csv else None
    info = bake_basis(
        Path(args.dna),
        Path(args.feature_names),
        Path(args.output),
        lod=args.lod,
        meshes=args.meshes,
        lip_csv=lip_csv if lip_csv and lip_csv.is_file() else None,
    )
    print(info)


if __name__ == "__main__":
    main()
