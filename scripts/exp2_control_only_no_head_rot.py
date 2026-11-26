#!/usr/bin/env python3
"""Experiment 2: fit one raw (T, 260) control tensor. No network.

HeadPitch, HeadRoll, and HeadYaw stay in the vector. They are omitted
from the loss. The loss is ParamLoss: blendshape_weight * mean squared
error, with the mean taken only over the remaining channels.

Stops when jawOpen MAE < 0.01 and jawOpen peak ratio >= 0.9, or at 500
Adam steps. Does not read or write the 20-epoch overfit directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/motion/AI/trainFramework")
DATA = ROOT / "data/unitalker_data/overfit_marusam_mh260_on_30"
OUT = ROOT / "models/UniTalker_Vanilla/results_exp2_control_only_no_head_rot"
OVERFIT = ROOT / "models/UniTalker_Vanilla/results_overfit_marusam_mh260_on_30"
PROTECTED = (
    OVERFIT,
    OUT,
)

WEIGHT = 0.0001
LR = 0.001
MAX_STEPS = 500
JAW_MAE_STOP = 0.01
PEAK_RATIO_STOP = 0.9
EXCLUDED = ("HeadPitch", "HeadRoll", "HeadYaw")
LIP_CLOSE = (
    "CTRL_expressions_mouthLipsTogetherDL",
    "CTRL_expressions_mouthLipsTogetherDR",
    "CTRL_expressions_mouthLipsTogetherUL",
    "CTRL_expressions_mouthLipsTogetherUR",
)


def feature_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names = [line.strip() for line in text.splitlines()]
    names = [name for name in names if name]
    if len(names) != 260 or len(names) != len(set(names)):
        raise SystemExit(f"feature_names.txt has {len(names)} names")
    return names


def index_of(names: list[str], wanted: str) -> int:
    hits = [i for i, name in enumerate(names) if name == wanted or name.endswith(wanted)]
    exact = [i for i, name in enumerate(names) if name == wanted]
    if exact:
        return exact[0]
    if len(hits) != 1:
        raise SystemExit(f"channel {wanted} matched {hits}")
    return hits[0]


def select_clip(names: list[str]) -> tuple[Path, list[dict]]:
    jaw = index_of(names, "CTRL_expressions_jawOpen")
    rows = []
    for path in sorted((DATA / "train").glob("*.npy")):
        jaw_open = np.load(path)[:, jaw]
        rows.append({
            "file": path.name,
            "frames": int(jaw_open.shape[0]),
            "jaw_std": float(jaw_open.std()),
            "jaw_max": float(jaw_open.max()),
            "jaw_mae_from_zero": float(np.abs(jaw_open).mean()),
        })
    chosen = max(rows, key=lambda row: row["jaw_std"])
    return DATA / "train" / chosen["file"], rows


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        return {
            "wav": str(path),
            "sample_rate": rate,
            "samples": frames,
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "seconds": frames / rate,
        }


def metrics(pred: torch.Tensor, gt: torch.Tensor, groups: dict[str, torch.Tensor]) -> dict:
    err = (pred - gt).abs()
    jaw = err[:, groups["jaw"]]
    gt_jaw = gt[:, groups["jaw"]]
    pred_jaw = pred[:, groups["jaw"]]
    gt_peak = float(gt_jaw.max())
    peak_at = int(torch.argmax(gt_jaw))
    return {
        "jaw_mae": float(jaw.mean()),
        "lip_close_mae": float(err[:, groups["lip"]].mean()),
        "active_mae": float(err[:, groups["active"]].mean()),
        "active_including_head_rot_mae": float(err[:, groups["active_with_head"]].mean()),
        "included_mae": float(err[:, groups["include"]].mean()),
        "head_rot_mae": float(err[:, groups["head"]].mean()),
        "peak_ratio": float(pred_jaw.max() / gt_jaw.max()),
        "pred_at_gt_peak": float(pred_jaw[peak_at]),
        "gt_peak": gt_peak,
        "gt_peak_frame": peak_at,
        "pred_peak": float(pred_jaw.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=WEIGHT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    weight = args.weight
    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    for protected in PROTECTED:
        if out.resolve() == protected.resolve():
            raise SystemExit(f"refusing to write {protected}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out}")

    names = feature_names(DATA / "feature_names.txt")
    clip_path, catalog = select_clip(names)
    gt_np = np.load(clip_path)
    if gt_np.dtype != np.float32 or gt_np.ndim != 2 or gt_np.shape[1] != 260:
        raise SystemExit(f"unexpected clip array {gt_np.dtype} {gt_np.shape}")

    jaw_i = index_of(names, "CTRL_expressions_jawOpen")
    head_i = [index_of(names, name) for name in EXCLUDED]
    lip_i = [index_of(names, name) for name in LIP_CLOSE]
    include = np.ones(260, dtype=bool)
    include[head_i] = False
    nonzero = np.max(np.abs(gt_np), axis=0) > 0
    active = nonzero & include
    active_with_head = nonzero.copy()

    torch.manual_seed(42)
    gt = torch.from_numpy(gt_np.copy())
    ctrl = torch.nn.Parameter(torch.zeros_like(gt))
    include_t = torch.from_numpy(include)
    groups = {
        "jaw": jaw_i,
        "lip": lip_i,
        "head": head_i,
        "include": torch.from_numpy(include),
        "active": torch.from_numpy(active),
        "active_with_head": torch.from_numpy(active_with_head),
    }
    opt = torch.optim.Adam([ctrl], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    out.mkdir(parents=True)
    rows = []
    stopped = "max_steps"
    stop_step = MAX_STEPS

    def record(step: int, loss_value: float | None, grad_l1: float | None) -> dict:
        with torch.no_grad():
            current = metrics(ctrl, gt, groups)
        current.update({
            "step": step,
            "loss": loss_value,
            "jaw_grad_l1": grad_l1,
        })
        rows.append(current)
        return current

    record(0, None, None)
    for step in range(1, MAX_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss = weight * F.mse_loss(ctrl[:, include_t], gt[:, include_t])
        loss.backward()
        grad_l1 = float(ctrl.grad[:, jaw_i].abs().mean())
        opt.step()
        current = record(step, float(loss.detach()), grad_l1)
        if current["jaw_mae"] < JAW_MAE_STOP and current["peak_ratio"] >= PEAK_RATIO_STOP:
            stopped = "criteria"
            stop_step = step
            break

    with torch.no_grad():
        held = ctrl[:, head_i]
        if float(held.abs().max()) != 0.0:
            raise SystemExit("excluded head rotation channels moved")
        fitted = ctrl.detach().cpu().numpy().astype(np.float32)

    final = rows[-1]
    err = np.abs(fitted - gt_np)
    included_mae = err[:, include].mean(axis=0)
    included_index = np.flatnonzero(include)
    top = np.argsort(-included_mae)[:12]
    top_rows = [{
        "index": int(included_index[k]),
        "name": names[int(included_index[k])],
        "mae": float(included_mae[k]),
        "gt_mae_from_zero": float(np.abs(gt_np[:, included_index[k]]).mean()),
        "gt_max_abs": float(np.max(np.abs(gt_np[:, included_index[k]]))),
    } for k in top]

    sse = gt_np.astype(np.float64) ** 2
    summary = {
        "experiment": out.name,
        "network": None,
        "clip": str(clip_path),
        "clip_sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
        "frames": int(gt_np.shape[0]),
        "channels": 260,
        "audio": wav_info(clip_path.with_suffix(".wav")),
        "init": "zeros",
        "device": "cpu",
        "optimizer": "Adam",
        "lr": LR,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "loss": f"{weight:g} * mean((pred-gt)^2) over channels other than HeadPitch, HeadRoll, HeadYaw",
        "loss_weight": weight,
        "loss_matches": "ParamLoss.forward in models/UniTalker_Vanilla/UniTalker-main/loss/loss.py",
        "excluded_from_loss": [
            {"index": int(i), "name": names[i]} for i in head_i
        ],
        "excluded_stay_in_vector": True,
        "jaw_index": jaw_i,
        "jaw_name": names[jaw_i],
        "lip_close": [
            {"index": int(i), "name": names[i]} for i in lip_i
        ],
        "active_definition": "GT max-abs > 0 on this clip, and the channel is inside the loss",
        "active_count": int(active.sum()),
        "active_including_head_rot_count": int(active_with_head.sum()),
        "zero_channel_count": int((~nonzero).sum()),
        "clip_catalog_by_jaw_std": catalog,
        "sse_share_at_zero_pred": {
            "head_rot": float(sse[:, head_i].sum() / sse.sum()),
            "jaw": float(sse[:, jaw_i].sum() / sse.sum()),
            "jaw_within_included": float(sse[:, jaw_i].sum() / sse[:, include].sum()),
            "lip_close": float(sse[:, lip_i].sum() / sse.sum()),
        },
        "first_jaw_grad_l1": rows[1]["jaw_grad_l1"],
        "adam_eps": 1e-8,
        "stop": stopped,
        "steps": stop_step,
        "max_steps": MAX_STEPS,
        "criteria": {
            "jaw_mae_below": JAW_MAE_STOP,
            "peak_ratio_at_least": PEAK_RATIO_STOP,
        },
        "criteria_met": stopped == "criteria",
        "final": {
            "jaw_mae": final["jaw_mae"],
            "lip_close_mae": final["lip_close_mae"],
            "active_mae": final["active_mae"],
            "active_including_head_rot_mae": final["active_including_head_rot_mae"],
            "included_mae": final["included_mae"],
            "head_rot_mae": final["head_rot_mae"],
            "peak_ratio": final["peak_ratio"],
            "pred_peak": final["pred_peak"],
            "gt_peak": final["gt_peak"],
            "gt_peak_frame": final["gt_peak_frame"],
            "pred_at_gt_peak": final["pred_at_gt_peak"],
            "loss": final["loss"],
        },
        "start": {
            "jaw_mae": rows[0]["jaw_mae"],
            "lip_close_mae": rows[0]["lip_close_mae"],
            "active_mae": rows[0]["active_mae"],
            "peak_ratio": rows[0]["peak_ratio"],
        },
        "largest_included_mae": top_rows,
        "overfit_dir_not_written": str(OVERFIT),
    }

    np.save(out / "optimized_controls.npy", fitted)
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "step", "loss", "jaw_mae", "lip_close_mae", "active_mae",
            "peak_ratio", "jaw_grad_l1", "pred_peak", "pred_at_gt_peak",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "jaw_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "gt_jawOpen", "pred_jawOpen"])
        for frame, (gt_value, pred_value) in enumerate(zip(gt_np[:, jaw_i], fitted[:, jaw_i])):
            writer.writerow([frame, f"{gt_value:.8f}", f"{pred_value:.8f}"])

    print(json.dumps({
        "stop": stopped,
        "steps": stop_step,
        "final": summary["final"],
        "start": summary["start"],
        "first_jaw_grad_l1": summary["first_jaw_grad_l1"],
        "sse_share_at_zero_pred": summary["sse_share_at_zero_pred"],
        "active_count": summary["active_count"],
        "clip": clip_path.name,
        "frames": summary["frames"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
