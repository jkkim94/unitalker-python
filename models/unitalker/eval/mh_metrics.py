"""MetaHuman evaluation metrics with explicit units and aggregation.

LVE (FaceFormer / academic):
  per frame t: max over lip verts v of ||pred_v - gt_v||_2  (Euclidean, mesh units)
  clip LVE = mean_t of that quantity.

Legacy note (2026-10 audit):
  old `linear_lip_lve` was mean of max *squared* distance — incorrect name.
  old `linear_lip_lvd` equaled the academic LVE. Names are corrected here.

Closure:
  upper/lower lip inner gap from linear-proxy verts (not jawOpen alone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_feature_names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def mouth_control_indices(feature_names: list[str], include_tongue: bool = False) -> np.ndarray:
    keys = ("mouth", "jaw", "lip", "teeth")
    if include_tongue:
        keys = keys + ("tongue",)
    idx = [i for i, n in enumerate(feature_names) if any(k in n.lower() for k in keys)]
    return np.asarray(idx, dtype=np.int64)


def tongue_control_indices(feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [i for i, n in enumerate(feature_names) if "tongue" in n.lower()],
        dtype=np.int64,
    )


def head_control_indices(feature_names: list[str]) -> np.ndarray:
    keys = ("head", "neck")
    return np.asarray(
        [i for i, n in enumerate(feature_names) if any(k in n.lower() for k in keys)],
        dtype=np.int64,
    )


def find_control_index(feature_names: list[str], substr: str) -> int | None:
    s = substr.lower()
    for i, n in enumerate(feature_names):
        if s in n.lower():
            return i
    return None


class LinearRigProxy:
    """Decode MetaHuman controls → vertices via baked linear basis (PROXY)."""

    def __init__(self, basis_path: Path):
        data = np.load(basis_path, allow_pickle=False)
        self.mean = data["mean_shape"].astype(np.float64)  # (3V,)
        self.basis = data["blend_shape"].astype(np.float64)  # (C, 3V)
        self.mouth_idx = data["mouth_idx"].astype(np.int64)
        self.vertex_count = int(data["vertex_count"][0]) if "vertex_count" in data else self.mean.size // 3
        self.feature_names = [str(x) for x in data["feature_names"].tolist()] if "feature_names" in data else []
        # Precompute lip-only basis slice: (C, L*3) for fast decode
        # mouth_idx are vertex ids; xyz interleaved in mean/basis as [x0,y0,z0,x1,...]
        lip_flat = np.empty(self.mouth_idx.size * 3, dtype=np.int64)
        lip_flat[0::3] = self.mouth_idx * 3
        lip_flat[1::3] = self.mouth_idx * 3 + 1
        lip_flat[2::3] = self.mouth_idx * 3 + 2
        self._lip_flat = lip_flat
        self._lip_mean = self.mean[lip_flat]
        self._lip_basis = self.basis[:, lip_flat]  # (C, 3L)
        # upper / lower lip partition from neutral Y (MetaHuman: +Y up)
        neut = self.controls_to_lip_vertices(np.zeros((1, self.basis.shape[0]), dtype=np.float64))[0]
        y = neut[:, 1]
        med = float(np.median(y))
        self.upper_local = np.where(y >= med)[0]
        self.lower_local = np.where(y < med)[0]
        self._x_bins = self._build_x_bins(neut)

    def _build_x_bins(self, lip_xyz: np.ndarray, n_bins: int = 24) -> list[tuple[np.ndarray, np.ndarray]]:
        x = lip_xyz[:, 0]
        edges = np.linspace(x.min(), x.max() + 1e-8, n_bins + 1)
        bins: list[tuple[np.ndarray, np.ndarray]] = []
        for b in range(n_bins):
            m = (x >= edges[b]) & (x < edges[b + 1])
            up = np.intersect1d(np.where(m)[0], self.upper_local)
            lo = np.intersect1d(np.where(m)[0], self.lower_local)
            if up.size and lo.size:
                bins.append((up, lo))
        return bins

    def controls_to_vertices(self, controls: np.ndarray) -> np.ndarray:
        """controls (T,C) -> verts (T,V,3). Full mesh (slow). Prefer lip-only helpers."""
        c = np.asarray(controls, dtype=np.float64)
        if c.ndim == 1:
            c = c[None]
        flat = c @ self.basis + self.mean[None, :]
        return flat.reshape(c.shape[0], self.vertex_count, 3)

    def controls_to_lip_vertices(self, controls: np.ndarray) -> np.ndarray:
        """controls (T,C) -> lip verts (T,L,3) via pre-sliced basis."""
        c = np.asarray(controls, dtype=np.float64)
        if c.ndim == 1:
            c = c[None]
        flat = c @ self._lip_basis + self._lip_mean[None, :]
        return flat.reshape(c.shape[0], self.mouth_idx.size, 3)

    def lip_vertex_errors(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """(T, L) Euclidean L2 per lip vertex."""
        n = min(len(gt), len(pred))
        gv = self.controls_to_lip_vertices(gt[:n])
        pv = self.controls_to_lip_vertices(pred[:n])
        return np.linalg.norm(gv - pv, axis=-1)

    def lip_lve_metrics(self, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        """Academic LVE + diagnostics.

        linear_lip_lve: mean_t max_v ||e||_2   (FaceFormer definition, mesh units)
        linear_lip_lve_sq: mean_t max_v ||e||^2 (legacy misnamed value)
        linear_lip_mean_l2: mean_{t,v} ||e||_2
        """
        err = self.lip_vertex_errors(gt, pred)  # (T,L)
        if err.size == 0:
            return {
                "linear_lip_lve": float("nan"),
                "linear_lip_lve_sq": float("nan"),
                "linear_lip_mean_l2": float("nan"),
                "linear_lip_lvd_legacy": float("nan"),
            }
        max_l2 = err.max(axis=1)
        max_sq = (err ** 2).max(axis=1)
        return {
            "linear_lip_lve": float(max_l2.mean()),
            "linear_lip_lve_sq": float(max_sq.mean()),
            "linear_lip_mean_l2": float(err.mean()),
            # legacy name that previously stored academic LVE
            "linear_lip_lvd_legacy": float(max_l2.mean()),
        }

    def lip_lve_lvd(self, gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        """Backward-compatible: returns (academic LVE, academic LVE).

        Previously returned (mean max sq, mean max L2). Callers should migrate
        to lip_lve_metrics().
        """
        m = self.lip_lve_metrics(gt, pred)
        return m["linear_lip_lve"], m["linear_lip_lve"]

    def lip_velocity(self, controls: np.ndarray) -> np.ndarray:
        v = self.controls_to_lip_vertices(controls)
        if len(v) < 2:
            return np.zeros((0,), dtype=np.float64)
        return np.linalg.norm(np.diff(v, axis=0), axis=-1).mean(axis=1)

    def lip_inner_gap(self, controls: np.ndarray) -> np.ndarray:
        """Upper–lower inner gap (mesh units) via X-binned Y separation.

        gap_bin = min(upper_Y) - max(lower_Y); clip gap = mean over bins.
        Larger = more open. Closing → gap ↓.
        """
        lip = self.controls_to_lip_vertices(controls)  # (T,L,3)
        T = lip.shape[0]
        gaps = np.zeros(T, dtype=np.float64)
        # vectorize per-bin
        for up_i, lo_i in self._x_bins:
            g = lip[:, up_i, 1].min(axis=1) - lip[:, lo_i, 1].max(axis=1)
            gaps += g
        if self._x_bins:
            gaps /= len(self._x_bins)
        return gaps


def control_mse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    n = min(len(gt), len(pred))
    g, p = gt[:n].astype(np.float64), pred[:n].astype(np.float64)
    diff2 = (g - p) ** 2
    if mask is None:
        return float(diff2.mean())
    m = mask.astype(np.float64).reshape(1, -1)
    denom = m.sum() * n
    if denom <= 0:
        return float(diff2.mean())
    return float((diff2 * m).sum() / denom)


def mouth_control_mse(gt: np.ndarray, pred: np.ndarray, mouth_idx: np.ndarray) -> float:
    if mouth_idx.size == 0:
        return control_mse(gt, pred)
    n = min(len(gt), len(pred))
    g = gt[:n, mouth_idx].astype(np.float64)
    p = pred[:n, mouth_idx].astype(np.float64)
    return float(((g - p) ** 2).mean())


def lip_velocity_error(proxy: LinearRigProxy, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    n = min(len(gt), len(pred))
    vg = proxy.lip_velocity(gt[:n])
    vp = proxy.lip_velocity(pred[:n])
    m = min(len(vg), len(vp))
    if m == 0:
        return {
            "lip_velocity_error": float("nan"),
            "lip_vel_ratio": float("nan"),
            "vel_gt_mean": float("nan"),
            "vel_pred_mean": float("nan"),
        }
    vg, vp = vg[:m], vp[:m]
    return {
        "lip_velocity_error": float(np.mean(np.abs(vp - vg))),
        "lip_vel_ratio": float(vp.mean() / (vg.mean() + 1e-8)),
        "vel_gt_mean": float(vg.mean()),
        "vel_pred_mean": float(vp.mean()),
    }


def _closure_events(closed: np.ndarray) -> np.ndarray:
    """Frame indices where closed becomes True (onset)."""
    if closed.size == 0:
        return np.zeros(0, dtype=np.int64)
    prev = np.concatenate([[False], closed[:-1]])
    return np.where(closed & ~prev)[0].astype(np.int64)


def lip_closure_metrics(
    proxy: LinearRigProxy,
    gt: np.ndarray,
    pred: np.ndarray,
    fps: float = 30.0,
    gt_closed_quantile: float = 0.12,
    min_closed_frames: int = 2,
) -> dict[str, Any]:
    """Upper/lower gap closure detection with TP/FP/FN/TN and timing.

    GT closed: gap <= quantile(gt_gap, gt_closed_quantile) AND gap < median(gt_gap).
    If GT has fewer than min_closed_frames closed frames → not applicable
    (do NOT report accuracy/F1 as 1.0).
    """
    n = min(len(gt), len(pred))
    gap_g = proxy.lip_inner_gap(gt[:n])
    gap_p = proxy.lip_inner_gap(pred[:n])
    thr = float(np.quantile(gap_g, gt_closed_quantile))
    med = float(np.median(gap_g))
    # require thr strictly below median so 'always closed' clips still need contrast
    if thr >= med:
        thr = float(min(thr, med - 1e-6))
    closed_gt = gap_g <= thr
    closed_pr = gap_p <= thr
    n_closed_gt = int(closed_gt.sum())
    n_open_gt = int((~closed_gt).sum())
    n_closed_pr = int(closed_pr.sum())

    out: dict[str, Any] = {
        "gap_thr": thr,
        "gap_gt_mean": float(gap_g.mean()),
        "gap_pred_mean": float(gap_p.mean()),
        "gap_mae": float(np.mean(np.abs(gap_p - gap_g))),
        "n_closed_gt": n_closed_gt,
        "n_open_gt": n_open_gt,
        "n_closed_pred": n_closed_pr,
        "closure_eval_applicable": bool(n_closed_gt >= min_closed_frames and n_open_gt >= 1),
        "lip_closure_agree": "",  # deprecated; do not use
    }

    if not out["closure_eval_applicable"]:
        out.update(
            {
                "closure_tp": "",
                "closure_fp": "",
                "closure_fn": "",
                "closure_tn": "",
                "closure_precision": "",
                "closure_recall": "",
                "closure_f1": "",
                "closure_accuracy": "N/A_no_gt_closure",
                "closure_onset_err_ms_mean": "",
                "closure_onset_err_ms_median": "",
                "n_gt_closure_onsets": 0,
            }
        )
        return out

    tp = int((closed_gt & closed_pr).sum())
    fp = int((~closed_gt & closed_pr).sum())
    fn = int((closed_gt & ~closed_pr).sum())
    tn = int((~closed_gt & ~closed_pr).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(n, 1)

    on_g = _closure_events(closed_gt)
    on_p = _closure_events(closed_pr)
    onset_errs_ms: list[float] = []
    for og in on_g:
        if on_p.size == 0:
            onset_errs_ms.append(float("nan"))
            continue
        nearest = on_p[np.argmin(np.abs(on_p - og))]
        onset_errs_ms.append(float(abs(int(nearest) - int(og)) * 1000.0 / fps))
    onset_arr = np.asarray([x for x in onset_errs_ms if x == x], dtype=np.float64)

    out.update(
        {
            "closure_tp": tp,
            "closure_fp": fp,
            "closure_fn": fn,
            "closure_tn": tn,
            "closure_precision": float(prec),
            "closure_recall": float(rec),
            "closure_f1": float(f1),
            "closure_accuracy": float(acc),
            "closure_onset_err_ms_mean": float(onset_arr.mean()) if onset_arr.size else "",
            "closure_onset_err_ms_median": float(np.median(onset_arr)) if onset_arr.size else "",
            "n_gt_closure_onsets": int(on_g.size),
            "gap_gt_closed_mean": float(gap_g[closed_gt].mean()),
            "gap_pred_at_gt_closed_mean": float(gap_p[closed_gt].mean()),
        }
    )
    return out


# keep old name as wrapper that redirects to new metrics (for any external imports)
def lip_closure_error(proxy: LinearRigProxy, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = lip_closure_metrics(proxy, gt, pred)
    return {
        "lip_closure_error": float(m["gap_mae"]),
        "lip_closure_agree": float("nan"),  # intentionally broken legacy
        "lip_closure_miss_open": float(m.get("gap_pred_at_gt_closed_mean", float("nan")) or float("nan")),
        "aperture_gt_mean": float(m["gap_gt_mean"]),
        "aperture_pred_mean": float(m["gap_pred_mean"]),
        "aperture_thr": float(m["gap_thr"]),
    }


def apply_channel_ablation(
    controls: np.ndarray,
    feature_names: list[str],
    mode: str,
    neutral: np.ndarray,
) -> np.ndarray:
    """Keep 260-dim interface; set non-target channels to neutral."""
    out = controls.copy()
    neu = neutral.reshape(-1).astype(np.float32)
    if neu.size != out.shape[-1]:
        neu = np.zeros((out.shape[-1],), dtype=np.float32)
    names = [n.lower() for n in feature_names]
    if mode == "full":
        return out
    if mode == "no_tongue":
        for i, n in enumerate(names):
            if "tongue" in n:
                out[:, i] = neu[i]
        return out
    if mode == "mouth_jaw":
        keep = ("mouth", "jaw", "lip", "teeth")
    elif mode == "mouth_jaw_face":
        keep = ("mouth", "jaw", "lip", "teeth", "cheek", "nose", "eye", "brow")
    else:
        raise ValueError(f"unknown ablation mode: {mode}")
    for i, n in enumerate(names):
        if not any(k in n for k in keep):
            out[:, i] = neu[i]
    return out


def try_native_riglogic_lve(
    gt_path: Path,
    pred: np.ndarray,
    work_pred_path: Path,
    config_path: Path,
    dna_path: Path | None = None,
    columns_path: Path | None = None,
) -> dict[str, Any]:
    """Attempt native DNA RigLogic LVE. Returns status + metrics or error."""
    import os
    import sys

    # ensure third_party/riglogic is importable
    rl_dir = str(ROOT / "third_party" / "riglogic")
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)
    if dna_path is not None and dna_path.is_file():
        os.environ["RIGLOGIC_DNA"] = str(dna_path)
    try:
        import riglogic  # noqa: F401
    except Exception as e:
        return {"native_riglogic_available": False, "native_riglogic_error": f"import riglogic: {e}"}
    try:
        from models.unitalker.riglogic_eval.compare_clip import compare_clip
    except Exception as e:
        return {"native_riglogic_available": False, "native_riglogic_error": f"import compare_clip: {e}"}
    try:
        work_pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(work_pred_path, pred.astype(np.float32))
        kwargs = dict(
            gt_path=gt_path,
            pred_path=work_pred_path,
            config_path=config_path,
            skip_vertices=False,
        )
        if columns_path is not None:
            kwargs["columns"] = columns_path
        report = compare_clip(**kwargs)
        out: dict[str, Any] = {"native_riglogic_available": True}
        # compare_clip lip_lve_mean — verify if Euclidean mean or max; pass through with note
        for k in ("lip_lve_mean", "vertex_mean_l2", "control_mse", "control_mae"):
            if k in report:
                out[k if k != "lip_lve_mean" else "native_riglogic_lve"] = report[k]
        return out
    except Exception as e:
        return {"native_riglogic_available": False, "native_riglogic_error": str(e)}
