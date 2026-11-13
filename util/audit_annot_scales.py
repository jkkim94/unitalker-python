#!/usr/bin/env python3
"""Audit annotation value scales across UniTalker datasets.

Reports per-dataset min/max, [0,1] compliance, scale-after-config values,
and MetaHuman channel-group variance share (head_pose / lipsync / eye / neck).
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

from models.unitalker.dataset.dataset_config import dataset_config  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "unitalker_data"
DEFAULT_REPORT = ROOT / "docs" / "reports" / "annot_scale_audit.json"
DROP_NON_LIPSYNC = ROOT / "configs" / "metahuman_drop_non_lipsync.txt"
HEAD_POSE_NAMES = {
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "HeadTranslationX",
    "HeadTranslationY",
    "HeadTranslationZ",
    "HeadControlSwitch",
}
METADATA_NAMES = {"DisableFaceOverride", "MHFDSVersion"}

ANNOT_EXPECTED_IN_01 = (
    "metahuman_control",
    "3DETF_blendshape_weight",
    "inhouse_blendshape_weight",
    "arkit_blendshape_52",
)


def _load_drop_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _resolve_datasets(names: list[str] | None) -> list[str]:
    if not names:
        return sorted(dataset_config.keys())
    out = []
    for name in names:
        if name in dataset_config:
            out.append(name)
            continue
        matches = [k for k in dataset_config if k == name or k.startswith(name + "_") or name in k]
        if not matches:
            raise KeyError(f"Unknown dataset key: {name}")
        out.extend(matches)
    return sorted(dict.fromkeys(out))


def _sample_clips(data_root: Path, dirname: str, max_clips: int) -> list[dict]:
    train_json = data_root / dirname / "train.json"
    if not train_json.exists():
        return []
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    return payload.get("data", [])[:max_clips]


def _array_stats(arr: np.ndarray) -> dict[str, float]:
    flat = arr.astype(np.float64).reshape(-1)
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "abs_mean": float(np.abs(flat).mean()),
        "pct_in_01": float(((flat >= 0) & (flat <= 1)).mean() * 100),
        "pct_neg": float((flat < 0).mean() * 100),
        "pct_gt1": float((flat > 1).mean() * 100),
        "variance": float(flat.var()),
    }


def _classify_metahuman_groups(feature_names: list[str], drop_names: set[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "head_pose": [],
        "eye_neck_drop": [],
        "lipsync_keep": [],
        "metadata": [],
    }
    for i, name in enumerate(feature_names):
        if name in HEAD_POSE_NAMES:
            groups["head_pose"].append(i)
        elif name in METADATA_NAMES:
            groups["metadata"].append(i)
        elif name in drop_names:
            groups["eye_neck_drop"].append(i)
        else:
            groups["lipsync_keep"].append(i)
    return groups


def _group_variance_share(clips: list[np.ndarray], groups: dict[str, list[int]]) -> dict[str, Any]:
    if not clips:
        return {}
    concat = np.concatenate(clips, axis=0)
    var_by_group: dict[str, float] = {}
    for gname, indices in groups.items():
        if not indices:
            continue
        var_by_group[gname] = float(concat[:, indices].var())
    total = sum(var_by_group.values()) or 1.0
    return {
        g: {"variance": v, "share_pct": 100.0 * v / total}
        for g, v in sorted(var_by_group.items(), key=lambda x: -x[1])
    }


def audit_dataset(
    dataset_key: str,
    data_root: Path,
    max_clips: int,
    drop_names: set[str],
) -> dict[str, Any]:
    cfg = dataset_config[dataset_key]
    dirname = cfg["dirname"]
    scale = float(cfg["scale"])
    annot_type = cfg["annot_type"]
    ds_root = data_root / dirname

    result: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dirname": dirname,
        "annot_type": annot_type,
        "scale": scale,
        "annot_dim": cfg["annot_dim"],
        "status": "ok",
    }

    clips_meta = _sample_clips(data_root, dirname, max_clips)
    if not clips_meta:
        result["status"] = "missing_train_json"
        return result

    raw_arrays: list[np.ndarray] = []
    scaled_arrays: list[np.ndarray] = []
    for item in clips_meta:
        ap = ds_root / item["annot_path"]
        if not ap.exists():
            continue
        raw = np.load(ap)
        raw_arrays.append(raw)
        scaled_arrays.append(raw * scale)

    if not raw_arrays:
        result["status"] = "no_annot_files"
        result["clips_requested"] = len(clips_meta)
        return result

    raw_stats = _array_stats(np.concatenate(raw_arrays, axis=0))
    scaled_stats = _array_stats(np.concatenate(scaled_arrays, axis=0))
    result["clips_sampled"] = len(raw_arrays)
    result["raw"] = raw_stats
    result["after_scale"] = scaled_stats

    if annot_type in ANNOT_EXPECTED_IN_01:
        result["in_01_check"] = "WARN" if raw_stats["pct_in_01"] < 95.0 else "PASS"
    else:
        result["in_01_check"] = "INFO"

    tmpl_path = ds_root / "id_template.npy"
    if tmpl_path.exists():
        tmpl = np.load(tmpl_path) * scale
        result["id_template"] = _array_stats(tmpl)

    fn_path = ds_root / "feature_names.txt"
    if annot_type == "metahuman_control" and fn_path.exists():
        feature_names = fn_path.read_text(encoding="utf-8").strip().split("\n")
        groups = _classify_metahuman_groups(feature_names, drop_names)
        result["metahuman_groups"] = _group_variance_share(raw_arrays, groups)
        top_var = []
        concat = np.concatenate(raw_arrays, axis=0)
        for i, name in enumerate(feature_names):
            ch = concat[:, i]
            top_var.append((name, float(ch.var()), float(np.abs(ch).mean())))
        top_var.sort(key=lambda x: -x[1])
        result["top_variance_channels"] = [
            {"name": n, "variance": v, "abs_mean": m, "dropped_in_lipsync_only": n in drop_names}
            for n, v, m in top_var[:15]
        ]

    return result


def format_table(report: dict[str, Any]) -> str:
    lines = [
        f"{'dataset':<28} {'type':<22} {'scale':>5} {'raw_min':>10} {'raw_max':>10} "
        f"{'in[0,1]%':>8} {'check':>5}",
        "-" * 95,
    ]
    for ds in report["datasets"]:
        if ds.get("status") != "ok":
            lines.append(f"{ds['dataset_key']:<28} {ds.get('status', 'error')}")
            continue
        raw = ds["raw"]
        lines.append(
            f"{ds['dataset_key']:<28} {ds['annot_type']:<22} {ds['scale']:>5.3f} "
            f"{raw['min']:>10.4f} {raw['max']:>10.4f} {raw['pct_in_01']:>7.1f}% "
            f"{ds.get('in_01_check', ''):>5}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UniTalker annotation value scales")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset_config keys (default: all registered)",
    )
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--drop-names-file",
        type=Path,
        default=DROP_NON_LIPSYNC,
        help="MetaHuman non-lipsync drop list for group variance",
    )
    args = parser.parse_args()

    dataset_keys = _resolve_datasets(
        [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    )
    drop_names = _load_drop_names(args.drop_names_file)

    datasets_report = [
        audit_dataset(k, args.data_root, args.max_clips, drop_names) for k in dataset_keys
    ]
    report = {
        "data_root": str(args.data_root),
        "max_clips_per_dataset": args.max_clips,
        "datasets": datasets_report,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(format_table(report))
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
