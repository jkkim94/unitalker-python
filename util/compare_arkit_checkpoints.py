#!/usr/bin/env python3
"""D3 ARKit: scratch 학습 ckpt vs UniTalker pretrained val/test blendshape MSE 비교."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE = ROOT / "configs" / "baseline_D3_arkit_pretrained.json"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _load_split(data_root: Path, dirname: str, split: str) -> list[dict]:
    payload = json.loads((data_root / dirname / f"{split}.json").read_text(encoding="utf-8"))
    return payload["data"]


def _infer_clip(
    item: dict,
    data_root: Path,
    dirname: str,
    checkpoint: Path,
    infer_dataset: str,
) -> np.ndarray:
    from models.unitalker.infer_audio_to_csv import infer_audio_to_blendshape_train_aligned

    gt_path = data_root / dirname / item["annot_path"]
    audio_path = data_root / dirname / item["audio_path"]
    out = infer_audio_to_blendshape_train_aligned(
        audio_path=str(audio_path),
        checkpoint_path=str(checkpoint),
        data_root=str(data_root),
        dataset_for_blendshape=infer_dataset,
        identity_idx=int(item.get("id", 0)),
        gt_npy_path=str(gt_path),
    )
    return out["blendshape"]


def _bs_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    diff = pred - gt
    return {
        "blendshape_mse": float(np.mean(diff ** 2)),
        "blendshape_mae": float(np.mean(np.abs(diff))),
        "frames": int(n),
    }


def evaluate(
    checkpoint: Path,
    infer_dataset: str,
    data_root: Path,
    dirname: str,
    split: str,
    max_clips: int,
) -> dict[str, Any]:
    items = _load_split(data_root, dirname, split)[:max_clips]
    per_clip = []
    mse_list, mae_list = [], []
    for item in items:
        stem = Path(item["annot_path"]).stem
        gt = np.load(data_root / dirname / item["annot_path"]).astype(np.float32)
        pred = _infer_clip(item, data_root, dirname, checkpoint, infer_dataset)
        m = _bs_metrics(gt, pred)
        per_clip.append({"clip": stem, **m})
        mse_list.append(m["blendshape_mse"])
        mae_list.append(m["blendshape_mae"])
    return {
        "summary": {
            "checkpoint": str(checkpoint),
            "infer_dataset": infer_dataset,
            "clips_evaluated": len(per_clip),
            "mean_blendshape_mse": float(np.mean(mse_list)) if mse_list else None,
            "mean_blendshape_mae": float(np.mean(mae_list)) if mae_list else None,
        },
        "per_clip": per_clip,
    }


def _parse_train_log_val_best(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    best = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Saved best checkpoint" not in line:
            continue
        # epoch N, val_loss: X
        import re
        m = re.search(r"epoch (\d+), val_loss: ([0-9.eE+-]+)", line)
        if m:
            loss = float(m.group(2))
            if best is None or loss < best["val_loss"]:
                best = {"epoch": int(m.group(1)), "val_loss": loss}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare D3 ARKit scratch vs pretrained")
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate-infer-dataset",
        default="D3",
        help="infer head for scratch ckpt (default D3)",
    )
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--candidate-train-log",
        type=Path,
        default=ROOT / "models/unitalker/results_D3_arkit_scratch/train.log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/reports/d3_arkit_scratch_vs_pretrained.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    from models.unitalker.dataset.dataset_config import dataset_config

    dataset_key = manifest["dataset"]
    dirname = dataset_config[dataset_key]["dirname"]
    data_root = _resolve(manifest["data_root"])
    baseline_ckpt = _resolve(manifest["checkpoint"])
    baseline_infer = manifest.get("infer_dataset", "D3_3DETF")
    candidate_ckpt = _resolve(args.candidate_checkpoint)
    max_clips = args.max_clips or len(_load_split(data_root, dirname, args.split))

    print(f"Baseline (pretrained): {baseline_ckpt} infer={baseline_infer}")
    print(f"Candidate (scratch):   {candidate_ckpt} infer={args.candidate_infer_dataset}")
    print(f"Evaluating {max_clips} clips from {args.split} ...")

    baseline = evaluate(baseline_ckpt, baseline_infer, data_root, dirname, args.split, max_clips)
    candidate = evaluate(
        candidate_ckpt, args.candidate_infer_dataset, data_root, dirname, args.split, max_clips
    )

    bs = baseline["summary"]
    cs = candidate["summary"]
    delta = {}
    for k in ("mean_blendshape_mse", "mean_blendshape_mae"):
        b, c = bs.get(k), cs.get(k)
        if b is not None and c is not None:
            delta[f"delta_{k}"] = float(c - b)
            delta[f"rel_{k}"] = float((c - b) / b) if b else None

    report = {
        "baseline_manifest": str(args.baseline_manifest),
        "split": args.split,
        "max_clips": max_clips,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_pretrained": delta,
        "candidate_train_val_best": _parse_train_log_val_best(args.candidate_train_log),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary (infer val blendshape vs GT) ===")
    print(f"pretrained MSE={bs.get('mean_blendshape_mse')} MAE={bs.get('mean_blendshape_mae')}")
    print(f"scratch    MSE={cs.get('mean_blendshape_mse')} MAE={cs.get('mean_blendshape_mae')}")
    print(f"delta MSE={delta.get('delta_mean_blendshape_mse')} rel={delta.get('rel_mean_blendshape_mse')}")
    if report["candidate_train_val_best"]:
        print(f"scratch train.log val best: {report['candidate_train_val_best']}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
