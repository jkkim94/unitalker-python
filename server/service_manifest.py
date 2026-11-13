"""service_manifest.json — FaceTool/서버 run 메타 단일 소스.

경로: models/unitalker/{run_id}/service_manifest.json
FaceTool은 이 파일을 직접 읽지 않고 GET /api/runs 로만 소비한다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# dataset_for_blendshape → FaceTool 로컬 DNA id
_DNA_BY_DATASET = {
    "D8": "0421_V33",
    "D8_sanity": "0421_V33",
    "D8_Metahuman_Control": "0421_V33",
    "D10": "0421_V33",
    "D9": "D9_Nuridam",
    "D9_Nuridam": "D9_Nuridam",
}

_MH_DATASETS = frozenset(_DNA_BY_DATASET.keys())


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_dir(run_id: str, root: Optional[Path] = None) -> Path:
    base = root or project_root()
    if run_id == "pretrained":
        return base / "models" / "unitalker" / "pretrained"
    return base / "models" / "unitalker" / run_id


def manifest_path(run_id: str, root: Optional[Path] = None) -> Path:
    return run_dir(run_id, root) / "service_manifest.json"


def dna_id_for_dataset(dataset: str) -> Optional[str]:
    return _DNA_BY_DATASET.get(dataset)


def output_kind_for_dataset(dataset: str, channels: Optional[int] = None) -> str:
    if dataset in _MH_DATASETS or channels == 260:
        return "metahuman_260"
    return "arkit_51"


def channels_for_kind(output_kind: str, dataset: str = "") -> int:
    if output_kind == "metahuman_260":
        return 260
    if dataset in _MH_DATASETS:
        return 260
    return 51


def feature_names_relpath(dataset: str) -> Optional[str]:
    try:
        from models.unitalker.dataset.dataset_config import dataset_config
    except Exception:
        dataset_config = {}
    cfg = dataset_config.get(dataset) or {}
    dirname = cfg.get("dirname")
    if not dirname:
        if dataset in _MH_DATASETS:
            dirname = "D8_metahuman_control" if dataset.startswith("D8") or dataset == "D10" else "D9_Nuridam"
        else:
            return None
    return f"data/unitalker_data/{dirname}/feature_names.txt"


def feature_names_hash(root: Path, relpath: Optional[str]) -> Optional[str]:
    if not relpath:
        return None
    p = root / relpath
    if not p.is_file():
        return None
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return h[:16]


def discover_engines(run_folder: Path, run_id: str = "") -> Dict[str, Any]:
    engines: Dict[str, Any] = {}
    if run_id == "pretrained" or run_folder.name == "pretrained":
        pt = run_folder / "UniTalker-B-D0-D7.pt"
        if pt.is_file():
            engines["pytorch"] = {"checkpoint": pt.name}
        onnx = run_folder / "unitalker_arkit51_pretrained.onnx"
        if onnx.is_file():
            engines["onnx"] = {"file": onnx.name}
        return engines

    best = run_folder / "best.pt"
    if best.is_file():
        engines["pytorch"] = {"checkpoint": "best.pt"}
    else:
        pts = sorted(run_folder.glob("*.pt"))
        if pts:
            engines["pytorch"] = {"checkpoint": pts[0].name}

    onnxs = sorted(run_folder.glob("*.onnx"))
    if onnxs:
        engines["onnx"] = {"file": onnxs[0].name}
    return engines


def load_service_manifest(run_id: str, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = manifest_path(run_id, root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def build_service_manifest(
    run_id: str,
    *,
    dataset_for_blendshape: str,
    template_idx: int = 0,
    label: Optional[str] = None,
    strategy: Optional[str] = None,
    output_kind: Optional[str] = None,
    root: Optional[Path] = None,
    fps: float = 30.0,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    root = root or project_root()
    folder = run_dir(run_id, root)
    kind = output_kind or output_kind_for_dataset(dataset_for_blendshape)
    if kind not in ("arkit_51", "metahuman_260"):
        kind = output_kind_for_dataset(dataset_for_blendshape)
    channels = channels_for_kind(kind, dataset_for_blendshape)
    fn_rel = feature_names_relpath(dataset_for_blendshape)
    engines = discover_engines(folder, run_id=run_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "output_kind": kind,
        "channels": channels,
        "dataset_for_blendshape": dataset_for_blendshape,
        "template_idx": int(template_idx),
        "feature_names_file": fn_rel,
        "fps": float(fps),
        "sample_rate": int(sample_rate),
        "engines": engines,
        "dna_id": dna_id_for_dataset(dataset_for_blendshape),
        "label": label or run_id,
        "strategy": strategy,
    }


def write_service_manifest(
    run_id: str,
    manifest: Dict[str, Any],
    *,
    root: Optional[Path] = None,
    merge: bool = True,
) -> Path:
    root = root or project_root()
    path = manifest_path(run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(manifest)
    if merge and path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                merged = dict(prev)
                merged.update(out)
                # engines: union by key, prefer new file names when present
                prev_eng = prev.get("engines") if isinstance(prev.get("engines"), dict) else {}
                new_eng = out.get("engines") if isinstance(out.get("engines"), dict) else {}
                eng = dict(prev_eng)
                eng.update(new_eng)
                merged["engines"] = eng
                out = merged
        except Exception:
            pass
    out["schema_version"] = SCHEMA_VERSION
    out["run_id"] = run_id
    # refresh engines from disk so stale onnx/pt names are fixed
    disk_eng = discover_engines(path.parent, run_id=run_id)
    if disk_eng:
        out["engines"] = disk_eng
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def resolve_run_service_meta(
    run_id: str,
    csv_meta: Optional[Dict[str, Any]] = None,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """manifest → CSV → 추정 순으로 FaceTool/runs API용 메타 합성."""
    root = root or project_root()
    folder = run_dir(run_id, root)
    csv_meta = csv_meta or {}
    mf = load_service_manifest(run_id, root) or {}

    dataset = (
        mf.get("dataset_for_blendshape")
        or csv_meta.get("dataset_for_blendshape")
        or ("D3_3DETF" if run_id == "pretrained" else None)
    )
    output_kind = (
        mf.get("output_kind")
        or csv_meta.get("output_kind")
        or (output_kind_for_dataset(dataset) if dataset else None)
        or "arkit_51"
    )
    if output_kind not in ("arkit_51", "metahuman_260"):
        output_kind = output_kind_for_dataset(dataset) if dataset else "arkit_51"
    if dataset and output_kind == "arkit_51" and dataset in _MH_DATASETS:
        output_kind = "metahuman_260"

    channels = mf.get("channels")
    if channels is None:
        channels = channels_for_kind(output_kind, dataset or "")

    engines_raw = mf.get("engines") if isinstance(mf.get("engines"), dict) else {}
    if not engines_raw:
        engines_raw = discover_engines(folder, run_id=run_id)

    engines_flags = {
        "pytorch": bool(engines_raw.get("pytorch")),
        "onnx": bool(engines_raw.get("onnx")),
    }

    pt_name = None
    if isinstance(engines_raw.get("pytorch"), dict):
        pt_name = engines_raw["pytorch"].get("checkpoint")
    onnx_name = None
    if isinstance(engines_raw.get("onnx"), dict):
        onnx_name = engines_raw["onnx"].get("file")

    has_checkpoint = False
    if pt_name and (folder / pt_name).is_file():
        has_checkpoint = True
    elif engines_flags["pytorch"]:
        has_checkpoint = any(folder.glob("*.pt"))

    label = mf.get("label") or csv_meta.get("label") or run_id
    strategy = mf.get("strategy") if mf.get("strategy") is not None else csv_meta.get("strategy")
    dna_id = mf.get("dna_id")
    if dna_id is None and dataset:
        dna_id = dna_id_for_dataset(dataset)

    template_idx = int(mf.get("template_idx", csv_meta.get("template_idx", 0)) or 0)
    fn_rel = mf.get("feature_names_file") or feature_names_relpath(dataset or "")
    is_default = bool(csv_meta.get("is_default"))

    return {
        "dataset_for_blendshape": dataset,
        "dataset": dataset,
        "output_kind": output_kind,
        "channels": int(channels),
        "template_idx": template_idx,
        "label": label,
        "strategy": strategy,
        "dna_id": dna_id,
        "engines": engines_flags,
        "engines_detail": engines_raw,
        "has_checkpoint": has_checkpoint,
        "feature_names_file": fn_rel,
        "feature_names_hash": feature_names_hash(root, fn_rel),
        "is_default": is_default,
        "pytorch_checkpoint": pt_name,
        "onnx_file": onnx_name,
    }


def default_dataset_guess_from_run_id(run_id: str) -> Optional[str]:
    rid = run_id.lower()
    if "d9" in rid or "nuridam" in rid:
        return "D9"
    if "d8" in rid or "metahuman" in rid or "arkit2mh" in rid:
        return "D8"
    if "d3" in rid or "arkit" in rid:
        return "D3"
    if run_id == "pretrained":
        return "D3_3DETF"
    return None
