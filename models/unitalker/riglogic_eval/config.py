"""Load riglogic.yaml and resolve paths relative to trainFramework root."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "riglogic.yaml"


def project_root() -> Path:
    return _PROJECT_ROOT


def _expand_env(value: str) -> str:
    pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(key)
        if env_val:
            return env_val
        if default is not None:
            return default
        return match.group(0)

    return pattern.sub(repl, value)


def _resolve_path(raw: str, base: Path) -> Path:
    expanded = _expand_env(str(raw).strip())
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_riglogic_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"riglogic config not found: {config_path}")

    data: dict[str, Any] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        data[key.strip()] = value

    root = _PROJECT_ROOT
    cfg = {
        "dna": _resolve_path(data.get("dna", ""), root),
        "lod": int(data.get("lod", 0)),
        "meshes": data.get("meshes", "face"),
        "lip_vertex_mask": _resolve_path(
            data.get("lip_vertex_mask", "metahuman_head_lod0_lip_mask.csv"), root
        ),
        "lip_vertices": _resolve_path(
            data.get("lip_vertices", "third_party/riglogic/lip_vertices.json"), root
        ),
        "control_mapping": _resolve_path(
            data.get("control_mapping", "third_party/riglogic/control_mapping.csv"), root
        ),
        "lipsync_drop_columns": _resolve_path(
            data.get("lipsync_drop_columns", "metahuman_drop_non_lipsync.txt"),
            root,
        ),
        "metahuman_columns_dir": _resolve_path(
            data.get("metahuman_columns_dir", "metahuman"), root
        ),
        "config_path": config_path.resolve(),
        "project_root": root,
    }
    return cfg
