"""Lazy FaceModel loader with riglogic extension path setup."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RIGLOGIC_DIR = _PROJECT_ROOT / "third_party" / "riglogic"


def _ensure_riglogic_path() -> None:
    riglogic_dir = str(_RIGLOGIC_DIR)
    if riglogic_dir not in sys.path:
        sys.path.insert(0, riglogic_dir)


@lru_cache(maxsize=4)
def get_face_model(dna: str, lod: int = 0, meshes: str = "head"):
    _ensure_riglogic_path()
    try:
        import riglogic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "riglogic extension not built. Run:\n"
            f"  cd {_RIGLOGIC_DIR} && pip install -r requirements.txt && "
            "python setup.py build_ext --inplace"
        ) from exc
    from face_model import FaceModel

    dna_path = Path(dna)
    if not dna_path.is_file():
        raise FileNotFoundError(
            f"DNA not found: {dna_path}. Set RIGLOGIC_DNA or edit riglogic.yaml"
        )
    return FaceModel(str(dna_path), lod=lod, meshes=meshes)


def eval_controls(
    fm,
    controls: Any,
    columns: list[str],
) -> Any:
    import numpy as np

    fm.set_columns(columns)
    return fm.eval_columns_batch(np.asarray(controls, dtype=np.float32))
