#!/usr/bin/env python3
"""Write or backfill models/unitalker/{run}/service_manifest.json.

Examples:
  python scripts/write_service_manifest.py --run results_D8_control_pretrained --dataset D8
  python scripts/write_service_manifest.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.run_config import get_run_metadata, load_run_config_csv  # noqa: E402
from server.service_manifest import (  # noqa: E402
    build_service_manifest,
    default_dataset_guess_from_run_id,
    write_service_manifest,
)


def _candidate_runs() -> list[str]:
    base = ROOT / "models" / "unitalker"
    runs = []
    if (base / "pretrained").is_dir():
        runs.append("pretrained")
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "pretrained":
            continue
        if p.name.startswith("results_") or p.name.startswith("results"):
            has = (
                any(p.glob("*.pt"))
                or any(p.glob("*.onnx"))
                or (p / "train.log").is_file()
                or (p / "train_console.log").is_file()
            )
            if has:
                runs.append(p.name)
    return runs


def write_one(run_id: str, dataset: str | None, template_idx: int | None) -> Path:
    csv_meta = get_run_metadata(run_id)
    ds = (
        dataset
        or csv_meta.get("dataset_for_blendshape")
        or default_dataset_guess_from_run_id(run_id)
    )
    if not ds:
        raise SystemExit(f"dataset unknown for run={run_id}; pass --dataset")
    ti = template_idx if template_idx is not None else int(csv_meta.get("template_idx") or 0)
    mf = build_service_manifest(
        run_id,
        dataset_for_blendshape=ds,
        template_idx=ti,
        label=csv_meta.get("label"),
        strategy=csv_meta.get("strategy"),
        output_kind=csv_meta.get("output_kind"),
        root=ROOT,
    )
    path = write_service_manifest(run_id, mf, root=ROOT, merge=True)
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write service_manifest.json for a run")
    parser.add_argument("--run", help="run_id under models/unitalker/")
    parser.add_argument("--dataset", help="dataset_for_blendshape (D8, D9, D3_3DETF, ...)")
    parser.add_argument("--template-idx", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="backfill all discovered runs + CSV ids")
    args = parser.parse_args()

    if args.all:
        by_csv, _ = load_run_config_csv()
        ids = sorted(set(_candidate_runs()) | set(by_csv.keys()))
        for rid in ids:
            try:
                write_one(rid, None, None)
            except SystemExit as e:
                print(f"skip {rid}: {e}")
        return

    if not args.run:
        raise SystemExit("--run or --all required")
    write_one(args.run, args.dataset, args.template_idx)


if __name__ == "__main__":
    main()
