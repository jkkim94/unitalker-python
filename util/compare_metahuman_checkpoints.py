#!/usr/bin/env python3
"""Compare MetaHuman checkpoints against a registered baseline manifest.

Baselines:
  - configs/baseline_D8_metahuman_260ch_mse.json (260ch control MSE)
  - configs/metahuman_baselines.json (index)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE = ROOT / "configs" / "baseline_D8_metahuman_260ch_mse.json"


def _load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _load_split(data_root: Path, dirname: str, split: str) -> list[dict]:
    payload = json.loads((data_root / dirname / f"{split}.json").read_text(encoding="utf-8"))
    return payload["data"]


def _control_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    diff = pred - gt
    return {
        "control_mse": float(np.mean(diff ** 2)),
        "control_mae": float(np.mean(np.abs(diff))),
        "frames": int(n),
    }


def _infer_clip(
    item: dict,
    data_root: Path,
    dirname: str,
    checkpoint: Path,
    dataset_key: str,
) -> np.ndarray:
    from models.unitalker.infer_audio_to_csv import infer_audio_to_blendshape_train_aligned

    gt_path = data_root / dirname / item["annot_path"]
    audio_path = data_root / dirname / item["audio_path"]
    out = infer_audio_to_blendshape_train_aligned(
        audio_path=str(audio_path),
        checkpoint_path=str(checkpoint),
        data_root=str(data_root),
        dataset_for_blendshape=dataset_key,
        identity_idx=int(item.get("id", 0)),
        gt_npy_path=str(gt_path),
    )
    return out["blendshape"]


def _riglogic_metrics(
    gt_path: Path,
    pred: np.ndarray,
    pred_path: Path,
    config_path: Path | None,
    skip_vertices: bool,
) -> dict[str, Any]:
    np.save(pred_path, pred.astype(np.float32))
    from models.unitalker.riglogic_eval.compare_clip import compare_clip

    try:
        report = compare_clip(
            gt_path=gt_path,
            pred_path=pred_path,
            config_path=config_path,
            skip_vertices=skip_vertices,
        )
        return {
            k: report[k]
            for k in (
                "control_mse",
                "control_mae",
                "lip_lve_mean",
                "vertex_mean_l2",
            )
            if k in report
        }
    except Exception as exc:
        return {"riglogic_error": str(exc)}


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path,
    dirname: str,
    dataset_key: str,
    split: str,
    max_clips: int,
    riglogic_config: Path | None,
    skip_riglogic: bool,
    work_dir: Path,
) -> dict[str, Any]:
    items = _load_split(data_root, dirname, split)[:max_clips]
    per_clip: list[dict[str, Any]] = []
    agg = {"control_mse": [], "control_mae": []}
    rl_agg: dict[str, list[float]] = {"lip_lve_mean": [], "vertex_mean_l2": []}

    work_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(items):
        stem = Path(item["annot_path"]).stem
        gt_path = data_root / dirname / item["annot_path"]
        gt = np.load(gt_path).astype(np.float32)
        pred = _infer_clip(item, data_root, dirname, checkpoint, dataset_key)
        metrics = _control_metrics(gt, pred)
        clip_report: dict[str, Any] = {
            "clip": stem,
            "annot_path": item["annot_path"],
            **metrics,
        }

        if not skip_riglogic:
            pred_tmp = work_dir / f"{stem}_pred.npy"
            rl = _riglogic_metrics(
                gt_path, pred, pred_tmp, riglogic_config, skip_vertices=False
            )
            clip_report["riglogic"] = rl
            if "lip_lve_mean" in rl:
                rl_agg["lip_lve_mean"].append(float(rl["lip_lve_mean"]))
            if "vertex_mean_l2" in rl:
                rl_agg["vertex_mean_l2"].append(float(rl["vertex_mean_l2"]))

        per_clip.append(clip_report)
        agg["control_mse"].append(metrics["control_mse"])
        agg["control_mae"].append(metrics["control_mae"])

    summary = {
        "checkpoint": str(checkpoint),
        "clips_evaluated": len(per_clip),
        "mean_control_mse": float(np.mean(agg["control_mse"])) if agg["control_mse"] else None,
        "mean_control_mae": float(np.mean(agg["control_mae"])) if agg["control_mae"] else None,
    }
    if rl_agg["lip_lve_mean"]:
        summary["mean_lip_lve_mean"] = float(np.mean(rl_agg["lip_lve_mean"]))
    if rl_agg["vertex_mean_l2"]:
        summary["mean_vertex_mean_l2"] = float(np.mean(rl_agg["vertex_mean_l2"]))

    return {"summary": summary, "per_clip": per_clip}


def _delta(candidate: dict, baseline: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("mean_control_mse", "mean_control_mae", "mean_lip_lve_mean", "mean_vertex_mean_l2"):
        c = candidate.get(key)
        b = baseline.get(key)
        if c is not None and b is not None:
            out[f"delta_{key}"] = float(c - b)
            out[f"rel_{key}"] = float((c - b) / b) if b != 0 else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MetaHuman checkpoints vs 260ch MSE baseline")
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="Print configs/metahuman_baselines.json and exit",
    )
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Override baseline checkpoint (default: from manifest)",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="LVE-enabled or new training checkpoint to compare",
    )
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all clips in split")
    parser.add_argument("--skip-riglogic", action="store_true", help="control MSE/MAE only")
    parser.add_argument("--riglogic-config", type=Path, default=ROOT / "configs" / "riglogic.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "reports" / "metahuman_checkpoint_compare.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp_debug" / "metahuman_compare")
    args = parser.parse_args()

    if args.list_baselines:
        index = json.loads((ROOT / "configs" / "metahuman_baselines.json").read_text(encoding="utf-8"))
        for b in index["baselines"]:
            print(f"[{b['id']}] {b['label']}")
            print(f"  manifest: {b['manifest']}")
            print(f"  checkpoint: {b['checkpoint']}")
            if b.get("onnx"):
                print(f"  onnx: {b['onnx']}")
            print(f"  use_when: {b['use_when']}")
            print()
        return

    if args.candidate_checkpoint is None:
        parser.error("--candidate-checkpoint is required unless --list-baselines is set")

    manifest = _load_baseline(args.baseline_manifest)
    data_root = _resolve(manifest["data_root"])
    from models.unitalker.dataset.dataset_config import dataset_config

    dataset_key = manifest["dataset"]
    dirname = dataset_config[dataset_key]["dirname"]
    baseline_ckpt = _resolve(args.baseline_checkpoint or manifest["checkpoint"])
    candidate_ckpt = _resolve(args.candidate_checkpoint)

    max_clips = args.max_clips or len(_load_split(data_root, dirname, args.split))

    print(f"Baseline: {baseline_ckpt}")
    print(f"Candidate: {candidate_ckpt}")
    print(f"Evaluating {max_clips} clips from {args.split} ...")

    baseline_result = evaluate_checkpoint(
        baseline_ckpt,
        data_root,
        dirname,
        dataset_key,
        args.split,
        max_clips,
        args.riglogic_config if not args.skip_riglogic else None,
        args.skip_riglogic,
        args.work_dir / "baseline",
    )
    candidate_result = evaluate_checkpoint(
        candidate_ckpt,
        data_root,
        dirname,
        dataset_key,
        args.split,
        max_clips,
        args.riglogic_config if not args.skip_riglogic else None,
        args.skip_riglogic,
        args.work_dir / "candidate",
    )

    report = {
        "baseline_manifest": str(args.baseline_manifest),
        "baseline_name": manifest.get("name"),
        "baseline_annot_dim": manifest.get("annot_dim"),
        "baseline_onnx": manifest.get("onnx"),
        "baseline_training": manifest.get("training", {}),
        "baseline_val_at_train_time": manifest.get("val_metrics_at_best", {}),
        "split": args.split,
        "max_clips": max_clips,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "delta_candidate_minus_baseline": _delta(
            candidate_result["summary"], baseline_result["summary"]
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Summary ===")
    for label, result in (("baseline", baseline_result), ("candidate", candidate_result)):
        s = result["summary"]
        print(f"{label}: control_mse={s.get('mean_control_mse')} control_mae={s.get('mean_control_mae')}", end="")
        if "mean_lip_lve_mean" in s:
            print(f" lip_lve_mean={s['mean_lip_lve_mean']}", end="")
        print()
    print(f"delta: {report['delta_candidate_minus_baseline']}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
