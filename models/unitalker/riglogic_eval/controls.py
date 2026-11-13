"""Load GT/pred controls from NPY or CSV; expand partial (200ch) to full columns."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_columns_file(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_columns_dir(columns_dir: str | Path) -> dict[str, Path]:
    columns_dir = Path(columns_dir)
    out: dict[str, Path] = {}
    for path in columns_dir.glob("*.columns.txt"):
        out[path.stem] = path
    return out


def infer_columns_path(
    controls_path: str | Path,
    columns: str | Path | None = None,
    columns_dir: str | Path | None = None,
) -> Path:
    controls_path = Path(controls_path)
    if columns is not None:
        return Path(columns)
    sibling = controls_path.with_suffix(".columns.txt")
    if sibling.is_file():
        return sibling
    if columns_dir is not None:
        stem = controls_path.stem
        candidate = Path(columns_dir) / f"{stem}.columns.txt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"columns file not found for {controls_path}. "
        "Pass --columns or place {stem}.columns.txt beside the controls file."
    )


def load_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return np.asarray(arr, dtype=np.float32)


def load_csv_controls(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        skip = {"frame", "time_sec"}
        col_names = [c for c in reader.fieldnames if c not in skip]
        rows = []
        for row in reader:
            rows.append([float(row[c]) for c in col_names])
    if not rows:
        raise ValueError(f"no data rows in CSV: {path}")
    return np.asarray(rows, dtype=np.float32), col_names


def load_controls(
    path: str | Path,
    columns: str | Path | None = None,
    columns_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[str], Path | None]:
    """Return (F, C), column names, full-columns path if resolved."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = load_npy(path)
        cols_path = infer_columns_path(path, columns, columns_dir)
        col_names = read_columns_file(cols_path)
        if data.shape[1] != len(col_names):
            raise ValueError(
                f"{path}: shape {data.shape[1]} != columns {len(col_names)}"
            )
        return data, col_names, cols_path
    if path.suffix.lower() == ".csv":
        data, col_names = load_csv_controls(path)
        cols_path = None
        if columns is not None:
            cols_path = Path(columns)
        return data, col_names, cols_path
    raise ValueError(f"unsupported controls format: {path}")


def align_frame_count(
    gt: np.ndarray,
    pred: np.ndarray,
    mode: str = "min",
) -> tuple[np.ndarray, np.ndarray, int]:
    if gt.shape[0] == pred.shape[0]:
        return gt, pred, gt.shape[0]
    if mode == "min":
        n = min(gt.shape[0], pred.shape[0])
        return gt[:n], pred[:n], n
    if mode == "gt":
        n = gt.shape[0]
        if pred.shape[0] < n:
            raise ValueError(f"pred shorter than gt: {pred.shape[0]} < {n}")
        return gt, pred[:n], n
    if mode == "pred":
        n = pred.shape[0]
        if gt.shape[0] < n:
            raise ValueError(f"gt shorter than pred: {gt.shape[0]} < {n}")
        return gt[:n], pred, n
    raise ValueError(f"unknown align mode: {mode}")


def expand_to_full_columns(
    partial: np.ndarray,
    partial_columns: list[str],
    full_columns: list[str],
) -> np.ndarray:
    """Map partial pred (e.g. D8 lipsync 200ch) into full MetaHuman column layout."""
    full_idx = {name: i for i, name in enumerate(full_columns)}
    out = np.zeros((partial.shape[0], len(full_columns)), dtype=np.float32)
    mapped = 0
    for pi, name in enumerate(partial_columns):
        fi = full_idx.get(name)
        if fi is None:
            continue
        out[:, fi] = partial[:, pi]
        mapped += 1
    if mapped != len(partial_columns):
        missing = set(partial_columns) - set(full_columns)
        if missing:
            raise ValueError(
                f"{len(missing)} pred columns not in full layout, e.g. {next(iter(missing))}"
            )
    return out


def resolve_full_columns(
    gt_columns_path: Path,
    pred_columns: list[str],
    full_columns_path: str | Path | None = None,
) -> tuple[list[str], bool]:
    """Return full column list and whether pred was expanded."""
    full_cols = read_columns_file(gt_columns_path)
    if pred_columns == full_cols:
        return full_cols, False
    if full_columns_path is not None:
        full_cols = read_columns_file(full_columns_path)
    if set(pred_columns).issubset(set(full_cols)) and len(pred_columns) < len(full_cols):
        return full_cols, True
    if pred_columns == full_cols:
        return full_cols, False
    raise ValueError(
        f"column mismatch: pred {len(pred_columns)} cols, full {len(full_cols)}. "
        "Pred must match GT layout or be a subset (lipsync-only)."
    )
