"""GT vs Pred RigLogic vertex metrics for one clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.unitalker.riglogic_eval.config import load_riglogic_config
from models.unitalker.riglogic_eval.controls import (
    align_frame_count,
    expand_to_full_columns,
    infer_columns_path,
    load_controls,
    read_columns_file,
    resolve_full_columns,
)
from models.unitalker.riglogic_eval.face_model_loader import eval_controls, get_face_model
from models.unitalker.riglogic_eval.metrics import (
    compute_control_metrics,
    compute_vertex_metrics,
    load_lip_mask_csv,
)


def compare_clip(
    gt_npy: str | Path | None = None,
    pred_npy: str | Path | None = None,
    gt_path: str | Path | None = None,
    pred_path: str | Path | None = None,
    columns: str | Path | None = None,
    full_columns: str | Path | None = None,
    dna: str | Path | None = None,
    lip_json: str | Path | None = None,
    lod: int | None = None,
    meshes: str | None = None,
    align: str = "min",
    skip_vertices: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare GT and pred controls through RigLogic FaceModel.

    Accepts NPY or CSV. For D8 lipsync-only pred (200 cols), pass full GT
    columns via ``columns`` / ``full_columns``; dropped channels are zero-filled.
    """
    cfg = load_riglogic_config(config_path)
    gt_path = Path(gt_path or gt_npy)
    pred_path = Path(pred_path or pred_npy)
    dna = str(dna or cfg["dna"])
    lip_mask_csv = cfg.get("lip_vertex_mask")
    lod = int(lod if lod is not None else cfg["lod"])
    meshes = meshes or cfg["meshes"]

    gt_data, gt_cols, gt_cols_path = load_controls(
        gt_path,
        columns=columns,
        columns_dir=cfg["metahuman_columns_dir"],
    )
    pred_data, pred_cols, _ = load_controls(
        pred_path,
        columns=None,
        columns_dir=cfg["metahuman_columns_dir"],
    )

    if gt_cols_path is None and columns is not None:
        gt_cols_path = Path(columns)
    if gt_cols_path is None:
        gt_cols_path = infer_columns_path(gt_path, columns, cfg["metahuman_columns_dir"])

    full_cols_path = Path(full_columns) if full_columns else gt_cols_path
    full_columns_list, need_expand = resolve_full_columns(
        gt_cols_path, pred_cols, full_columns_path=full_cols_path
    )

    if need_expand:
        pred_data = expand_to_full_columns(pred_data, pred_cols, full_columns_list)
        pred_cols = full_columns_list
        gt_data_expanded = gt_data
        if gt_cols != full_columns_list:
            gt_data_expanded = expand_to_full_columns(gt_data, gt_cols, full_columns_list)
            gt_cols = full_columns_list
        else:
            gt_data_expanded = gt_data
    else:
        gt_data_expanded = gt_data
        if gt_cols != pred_cols:
            raise ValueError("GT and pred column names differ; use full_columns or matching layout")

    gt_data_expanded, pred_data, n_frames = align_frame_count(
        gt_data_expanded, pred_data, mode=align
    )

    report: dict[str, Any] = {
        "clip": pred_path.stem,
        "gt_path": str(gt_path.resolve()),
        "pred_path": str(pred_path.resolve()),
        "columns_path": str(full_cols_path.resolve()),
        "frame_count": n_frames,
        "column_count": len(full_columns_list),
        "pred_expanded_to_full": need_expand,
        "dna": dna,
        "meshes": meshes,
        "lod": lod,
    }
    report.update(
        compute_control_metrics(gt_data_expanded, pred_data, full_columns_list)
    )

    if skip_vertices:
        return report

    fm = get_face_model(dna, lod=lod, meshes=meshes)
    gt_verts = eval_controls(fm, gt_data_expanded, full_columns_list)
    pred_verts = eval_controls(fm, pred_data, full_columns_list)

    v_count = int(gt_verts.shape[1])
    if lip_mask_csv and Path(lip_mask_csv).is_file():
        lip_mask = load_lip_mask_csv(lip_mask_csv, v_count)
        report["lip_mask_source"] = str(lip_mask_csv)
    else:
        raise FileNotFoundError(
            f"MH-LVE-LIP-v1 lip mask CSV not found: {lip_mask_csv}"
        )
    report.update(compute_vertex_metrics(gt_verts, pred_verts, lip_mask))

    zero_dev = float(
        np.abs(
            eval_controls(fm, np.zeros_like(gt_data_expanded[:1]), full_columns_list)[0]
            - fm.neutral_vertices
        ).max()
    )
    report["zero_control_deviation_cm"] = zero_dev
    return report


def _self_test_report(
    gt_path: Path,
    columns: Path | None,
    cfg_path: Path | None,
    dna: str | None,
) -> dict[str, Any]:
    return compare_clip(
        gt_path=gt_path,
        pred_path=gt_path,
        columns=columns,
        dna=dna,
        config_path=cfg_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RigLogic GT vs Pred vertex metrics (one clip)")
    ap.add_argument("--gt-npy", "--gt", dest="gt_path", help="GT controls (.npy or .csv)")
    ap.add_argument("--pred-npy", "--pred", dest="pred_path", help="Pred controls (.npy or .csv)")
    ap.add_argument("--columns", help="Full MetaHuman .columns.txt (default: sibling of GT)")
    ap.add_argument("--full-columns", help="Full layout for partial pred expansion")
    ap.add_argument("--dna", help="head.dna path (overrides config / RIGLOGIC_DNA)")
    ap.add_argument("--lip-json", help="lip_vertices.json path")
    ap.add_argument("--lod", type=int, default=None)
    ap.add_argument("--meshes", default=None, choices=["head", "face", "all"])
    ap.add_argument("--align", default="min", choices=["min", "gt", "pred"])
    ap.add_argument("--config", dest="config_path", default=None, help="riglogic.yaml")
    ap.add_argument("--out", help="Write JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="GT vs GT (error should be ~0)")
    ap.add_argument("--control-only", action="store_true", help="Skip RigLogic vertex eval")
    args = ap.parse_args()

    if args.self_test:
        if not args.gt_path:
            ap.error("--self-test requires --gt-npy")
        report = _self_test_report(
            Path(args.gt_path),
            Path(args.columns) if args.columns else None,
            Path(args.config_path) if args.config_path else None,
            args.dna,
        )
    else:
        if not args.gt_path or not args.pred_path:
            ap.error("--gt-npy and --pred-npy are required")
        report = compare_clip(
            gt_path=args.gt_path,
            pred_path=args.pred_path,
            columns=args.columns,
            full_columns=args.full_columns,
            dna=args.dna,
            lip_json=args.lip_json,
            lod=args.lod,
            meshes=args.meshes,
            align=args.align,
            skip_vertices=args.control_only,
            config_path=args.config_path,
        )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
