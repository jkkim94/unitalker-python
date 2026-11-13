"""Export pred controls to RigLogic vertex cache (.npy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.unitalker.riglogic_eval.compare_clip import compare_clip
from models.unitalker.riglogic_eval.config import load_riglogic_config
from models.unitalker.riglogic_eval.controls import (
    expand_to_full_columns,
    load_controls,
    resolve_full_columns,
)
from models.unitalker.riglogic_eval.face_model_loader import eval_controls, get_face_model


def export_vertices(
    pred_path: Path,
    out_path: Path,
    columns: Path | None = None,
    full_columns: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    import numpy as np

    cfg = load_riglogic_config(config_path)
    pred_data, pred_cols, _ = load_controls(
        pred_path, columns=columns, columns_dir=cfg["metahuman_columns_dir"]
    )

    if full_columns is None and columns is not None:
        full_columns = columns
    if full_columns is not None:
        full_list = __import__(
            "models.unitalker.riglogic_eval.controls",
            fromlist=["read_columns_file"],
        ).read_columns_file(full_columns)
        _, need_expand = resolve_full_columns(full_columns, pred_cols)
        if need_expand:
            pred_data = expand_to_full_columns(pred_data, pred_cols, full_list)
            pred_cols = full_list
    elif columns is not None:
        pred_cols = __import__(
            "models.unitalker.riglogic_eval.controls",
            fromlist=["read_columns_file"],
        ).read_columns_file(columns)

    fm = get_face_model(str(cfg["dna"]), lod=cfg["lod"], meshes=cfg["meshes"])
    verts = eval_controls(fm, pred_data, pred_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, verts.astype(np.float64))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Pred controls -> vertices.npy via RigLogic")
    ap.add_argument("--pred", required=True, help="Pred .npy or .csv")
    ap.add_argument("--out", required=True, help="Output .npy path (F, V, 3)")
    ap.add_argument("--columns", help="Column layout .columns.txt")
    ap.add_argument("--full-columns", help="Full 260-col layout for partial pred")
    ap.add_argument("--config", dest="config_path", default=None)
    args = ap.parse_args()

    out = export_vertices(
        Path(args.pred),
        Path(args.out),
        columns=Path(args.columns) if args.columns else None,
        full_columns=Path(args.full_columns) if args.full_columns else None,
        config_path=Path(args.config_path) if args.config_path else None,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
