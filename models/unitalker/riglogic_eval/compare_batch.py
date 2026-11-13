"""Batch RigLogic eval over validation clips or inference output directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.unitalker.riglogic_eval.compare_clip import compare_clip
from models.unitalker.riglogic_eval.config import load_riglogic_config
from models.unitalker.riglogic_eval.metrics import summarize_batch


def _pair_from_manifest(manifest: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = Path(row["gt_path"])
            pred = Path(row["pred_path"])
            pairs.append((gt, pred))
    return pairs


def _pair_from_dirs(gt_dir: Path, pred_dir: Path, suffixes: tuple[str, ...]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for gt_path in sorted(gt_dir.iterdir()):
        if not gt_path.is_file():
            continue
        if gt_path.suffix.lower() not in suffixes and not gt_path.name.endswith(".npy"):
            continue
        stem = gt_path.stem
        for suf in suffixes:
            pred_path = pred_dir / f"{stem}{suf}"
            if pred_path.is_file():
                pairs.append((gt_path, pred_path))
                break
            pred_path = pred_dir / f"pred_{stem}{suf}"
            if pred_path.is_file():
                pairs.append((gt_path, pred_path))
                break
    return pairs


def compare_batch(
    pairs: list[tuple[Path, Path]],
    columns: Path | None = None,
    full_columns: Path | None = None,
    config_path: Path | None = None,
    skip_vertices: bool = False,
    **compare_kwargs,
) -> dict:
    reports = []
    errors = []
    for gt_path, pred_path in pairs:
        try:
            report = compare_clip(
                gt_path=gt_path,
                pred_path=pred_path,
                columns=columns,
                full_columns=full_columns,
                config_path=config_path,
                skip_vertices=skip_vertices,
                **compare_kwargs,
            )
            reports.append(report)
        except Exception as exc:
            errors.append({"gt": str(gt_path), "pred": str(pred_path), "error": str(exc)})

    summary = summarize_batch(reports)
    summary["errors"] = errors
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch RigLogic GT vs Pred eval")
    ap.add_argument("--manifest", help="CSV with gt_path,pred_path columns")
    ap.add_argument("--gt-dir", help="Directory of GT .npy/.csv files")
    ap.add_argument("--pred-dir", help="Directory of pred .npy/.csv files")
    ap.add_argument("--columns", help="Shared full .columns.txt (optional per-clip sibling)")
    ap.add_argument("--full-columns", help="Full layout for lipsync partial pred")
    ap.add_argument("--config", dest="config_path", default=None)
    ap.add_argument("--out", required=True, help="Output JSON summary path")
    ap.add_argument("--control-only", action="store_true")
    args = ap.parse_args()

    cfg = load_riglogic_config(args.config_path)
    pairs: list[tuple[Path, Path]] = []
    if args.manifest:
        pairs = _pair_from_manifest(Path(args.manifest))
    elif args.gt_dir and args.pred_dir:
        pairs = _pair_from_dirs(
            Path(args.gt_dir),
            Path(args.pred_dir),
            (".npy", ".csv"),
        )
    else:
        ap.error("Provide --manifest or both --gt-dir and --pred-dir")

    if not pairs:
        ap.error("No GT/pred pairs found")

    summary = compare_batch(
        pairs,
        columns=Path(args.columns) if args.columns else None,
        full_columns=Path(args.full_columns) if args.full_columns else None,
        config_path=Path(args.config_path) if args.config_path else None,
        skip_vertices=args.control_only,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Evaluated {summary.get('clip_count', 0)} clips, "
        f"{len(summary.get('errors', []))} errors -> {out_path}"
    )
    if summary.get("lip_lve_mean_avg") is not None:
        print(f"  lip_lve_mean_avg: {summary['lip_lve_mean_avg']:.6f}")
    if summary.get("vertex_mean_l2_avg") is not None:
        print(f"  vertex_mean_l2_avg: {summary['vertex_mean_l2_avg']:.6f}")


if __name__ == "__main__":
    main()
