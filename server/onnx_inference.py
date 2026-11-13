"""ONNX Runtime 경로 — POST /api/inference engine=onnx."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _split_raw_audio(audio: np.ndarray, sr: int = 16000, a: float = 25.0, b: float = 5.0):
    total_length = len(audio) / sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    chunks = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for _ in range(reps):
        chunks.append(audio[start:end].astype(np.float32))
        start += step
        end += step
    return chunks


def infer_onnx_blendshape(
    audio_path: str,
    onnx_path: str,
    data_root: str,
    dataset_for_blendshape: str,
    template_idx: int = 0,
    style_idx: int = 0,
    fps: float = 30.0,
    logger=None,
) -> np.ndarray:
    """Return float32 [frames, channels] from ONNX (raw 16kHz mono)."""
    import librosa
    import onnxruntime as ort

    from models.unitalker.dataset.dataset_config import dataset_config
    from models.unitalker.infer_audio_to_csv import merge_out_list

    if logger:
        logger.section("ONNX INFERENCE")
        logger.kv("onnx_path", onnx_path)
        logger.kv("dataset", dataset_for_blendshape)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    cfg = dataset_config[dataset_for_blendshape]
    scale = float(cfg["scale"])
    tpl_path = Path(data_root) / cfg["dirname"] / "id_template.npy"
    tpl_all = np.load(tpl_path).astype(np.float32)
    ti = max(0, min(int(template_idx), len(tpl_all) - 1))
    template = (scale * tpl_all[ti]).astype(np.float32)[None, :]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    chunks = _split_raw_audio(audio)
    out_list = []
    for i, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        num_frames = np.array(
            [int(round(chunk.shape[0] / 16000.0 * fps))], dtype=np.int64
        )
        style = np.array([int(style_idx)], dtype=np.int64)
        feeds = {
            "audio": chunk[None, :],
            "template": template,
            "style_idx": style,
            "num_frames": num_frames,
        }
        out = sess.run(None, feeds)[0]
        # out: [1, F, C] or [F, C]
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        out_list.append(arr)
        if logger:
            logger.kv(f"chunk_{i}_frames", arr.shape[0])

    if not out_list:
        raise RuntimeError("ONNX inference produced no frames")

    merged = merge_out_list(out_list, int(fps)).astype(np.float32)
    if logger:
        logger.kv("merged_shape", str(merged.shape))
    return merged


def resolve_onnx_path(run_folder: Path, onnx_file: Optional[str] = None) -> Path:
    if onnx_file:
        p = run_folder / onnx_file
        if p.is_file():
            return p
        raise FileNotFoundError(p)
    onnxs = sorted(run_folder.glob("*.onnx"))
    if not onnxs:
        raise FileNotFoundError(f"No .onnx in {run_folder}")
    return onnxs[0]
