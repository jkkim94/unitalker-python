#!/usr/bin/env python3
"""
UniTalker 체크포인트 -> ONNX (MetaHuman D8 또는 Vanilla blendshape head).

기본: results_D8_control260_pretrained/best.pt
입력: audio (B, T) 16kHz float32, template (B, C), style_idx (B,) int64, num_frames
출력: controls/blendshape (B, F, C) — F = round(T/16000*fps)

전처리(split_long_audio, Wav2Vec2FeatureExtractor)와 병합(merge_out_list)은
런타임에서 동일하게 처리하거나 별도 구현이 필요합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.unitalker.dataset.dataset_config import dataset_config
from models.unitalker.infer_audio_to_csv import _build_args_from_ckpt
from models.unitalker.model.unitalker import UniTalker

# server/main.py 와 동일: 3DETF 52ch 알파벳순에서 tongueOut 인덱스
_ARKIT_51_BS_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
_TONGUE_OUT_IDX_52 = sorted(_ARKIT_51_BS_NAMES + ["tongueOut"]).index("tongueOut")


class UniTalkerOnnxWrapper(nn.Module):
    """ONNX용: 단일 annot_type head (metahuman_control 또는 blendshape)."""

    def __init__(
        self,
        model: UniTalker,
        annot_dim: int,
        fps: float = 30.0,
        annot_type: str = "metahuman_control",
        drop_tongue_out: bool = False,
    ):
        super().__init__()
        self.model = model
        self.fps = float(fps)
        self.annot_dim = int(annot_dim)
        self.annot_type = str(annot_type)
        self.drop_tongue_out = bool(drop_tongue_out)
        if self.drop_tongue_out:
            if self.annot_dim != 52:
                raise ValueError("drop_tongue_out requires annot_dim=52 (3DETF head)")
            keep = [i for i in range(self.annot_dim) if i != _TONGUE_OUT_IDX_52]
            self.register_buffer(
                "_keep_idx",
                torch.tensor(keep, dtype=torch.int64),
                persistent=False,
            )
        self.output_dim = self.annot_dim - 1 if self.drop_tongue_out else self.annot_dim

    def forward(
        self,
        audio: torch.Tensor,
        template: torch.Tensor,
        style_idx: torch.Tensor,
        num_frames: torch.Tensor,
    ) -> torch.Tensor:
        nf = num_frames.reshape(-1)[0]
        b = audio.shape[0]
        face_motion = torch.zeros(
            b, 1, self.annot_dim, device=audio.device, dtype=audio.dtype
        ).repeat(1, nf, 1)
        out, _, _, _ = self.model(
            audio,
            template,
            face_motion,
            style_idx=style_idx,
            annot_type=self.annot_type,
            fps=self.fps,
        )
        if self.drop_tongue_out:
            out = out.index_select(-1, self._keep_idx)
        return out


def _load_model(checkpoint: Path, data_root: Path, dataset_name: str) -> tuple[UniTalker, object]:
    args, model_state = _build_args_from_ckpt(str(checkpoint), str(data_root), dataset_name)
    model = UniTalker(args)
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in model_state.items():
        nk = k
        if "pos_conv_embed.conv.weight_g" in k:
            nk = k.replace(
                "pos_conv_embed.conv.weight_g",
                "pos_conv_embed.conv.parametrizations.weight.original0",
            )
        elif "pos_conv_embed.conv.weight_v" in k:
            nk = k.replace(
                "pos_conv_embed.conv.weight_v",
                "pos_conv_embed.conv.parametrizations.weight.original1",
            )
        if nk in model_keys:
            remapped[nk] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()
    return model, args


def _default_template(data_root: Path, dataset_name: str, template_idx: int) -> np.ndarray:
    scale = float(dataset_config[dataset_name]["scale"])
    tpl_path = data_root / dataset_config[dataset_name]["dirname"] / "id_template.npy"
    tpl_all = np.load(tpl_path).astype(np.float32)
    ti = max(0, min(int(template_idx), len(tpl_all) - 1))
    return (scale * tpl_all[ti]).astype(np.float32)


def export_onnx(
    checkpoint: Path,
    output_path: Path,
    data_root: Path,
    dataset_name: str = "D9",
    template_idx: int = 0,
    opset: int = 17,
    fps: float = 30.0,
    audio_sec: float = 10.0,
    arkit_51: bool = False,
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model, args = _load_model(checkpoint, data_root, dataset_name)
    annot_dim = int(dataset_config[dataset_name]["annot_dim"])
    annot_type = str(dataset_config[dataset_name]["annot_type"])
    wrapper = UniTalkerOnnxWrapper(
        model,
        annot_dim=annot_dim,
        fps=fps,
        annot_type=annot_type,
        drop_tongue_out=arkit_51,
    ).eval()
    out_dim = wrapper.output_dim

    n_samples = int(round(audio_sec * 16000))
    n_frames = int(round(n_samples / 16000.0 * fps))
    dummy_audio = torch.randn(1, n_samples, dtype=torch.float32)
    template_np = _default_template(data_root, dataset_name, template_idx)
    dummy_template = torch.from_numpy(template_np).unsqueeze(0)
    dummy_style = torch.zeros(1, dtype=torch.int64)
    dummy_frames = torch.tensor([n_frames], dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "audio": {0: "batch", 1: "audio_samples"},
        "template": {0: "batch"},
        "style_idx": {0: "batch"},
        "blendshape": {0: "batch", 1: "num_frames"},
    }

    export_opset = int(opset)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_template, dummy_style, dummy_frames),
            str(output_path),
            input_names=["audio", "template", "style_idx", "num_frames"],
            output_names=["blendshape"],
            dynamic_axes=dynamic_axes,
            opset_version=export_opset,
            do_constant_folding=True,
        )

    meta = {
        "checkpoint": str(checkpoint.resolve()),
        "dataset": dataset_name,
        "annot_type": annot_type,
        "output_kind": "arkit_51" if arkit_51 else annot_type,
        "output_dim": out_dim,
        "fps": fps,
        "sample_rate": 16000,
        "template_idx": template_idx,
        "identity_num": int(args.identity_num),
        "audio_encoder_repo": args.audio_encoder_repo,
        "decoder_type": args.decoder_type,
        "interpolate_pos": args.interpolate_pos,
        "opset": export_opset,
        "inputs": {
            "audio": "float32 [batch, samples] mono 16kHz",
            "template": f"float32 [batch, {annot_dim}]",
            "style_idx": "int64 [batch] speaker/identity index",
            "num_frames": "int64 scalar (or [1]); round(samples/16000*fps)",
        },
        "outputs": {
            "blendshape": f"float32 [batch, frames, {out_dim}]",
        },
        "notes": [
            "긴 오디오는 infer_audio_to_csv.split_long_audio + merge_out_list를 ONNX 외부에서 적용.",
            "채널 의미는 dataset의 feature_names.txt를 참고.",
        ],
    }
    if arkit_51:
        meta["blendshape_names"] = list(_ARKIT_51_BS_NAMES)
        meta["notes"].append(
            "3DETF 52ch head 추론 후 tongueOut 제거. 서버 /api/inference arkit_51 경로와 동일."
        )
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # FaceTool/서버 run 메타: service_manifest.json merge
    try:
        from server.service_manifest import (
            build_service_manifest,
            output_kind_for_dataset,
            write_service_manifest,
        )

        run_id = checkpoint.parent.name
        kind = "arkit_51" if arkit_51 else output_kind_for_dataset(dataset_name, out_dim)
        mf = build_service_manifest(
            run_id,
            dataset_for_blendshape=dataset_name if not arkit_51 else "D3_3DETF",
            template_idx=template_idx,
            output_kind=kind,
            root=_ROOT,
            fps=fps,
        )
        write_service_manifest(run_id, mf, root=_ROOT, merge=True)
    except Exception as e:
        print(f"[export_onnx] service_manifest write skipped: {e}", flush=True)

    return output_path


def verify_onnx(onnx_path: Path, wrapper: UniTalkerOnnxWrapper, fps: float = 30.0, atol: float = 1e-3) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    n_samples = int(16000 * 5.0)
    n_frames = int(round(n_samples / 16000.0 * fps))
    audio = torch.randn(1, n_samples)
    template = torch.randn(1, wrapper.annot_dim)
    style = torch.zeros(1, dtype=torch.int64)
    num_frames = torch.tensor([n_frames], dtype=torch.int64)

    with torch.no_grad():
        pt_out = wrapper(audio, template, style, num_frames).numpy()

    ort_out = sess.run(
        None,
        {
            "audio": audio.numpy(),
            "template": template.numpy(),
            "style_idx": style.numpy(),
            "num_frames": num_frames.numpy(),
        },
    )[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    if max_diff > atol:
        raise RuntimeError(f"ONNX verify failed: max_diff={max_diff} > {atol}")
    return max_diff


def main():
    default_ckpt = _ROOT / "models/unitalker/pretrained/UniTalker-B-D0-D7.pt"
    default_out = _ROOT / "models/unitalker/pretrained/unitalker_arkit51_pretrained.onnx"
    parser = argparse.ArgumentParser(description="Export UniTalker checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_out,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_ROOT / "data/unitalker_data",
    )
    parser.add_argument("--dataset", default="D3")
    parser.add_argument("--template-idx", type=int, default=0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--verify", action="store_true", help="onnxruntime로 PyTorch 출력과 비교")
    parser.add_argument(
        "--arkit-51",
        action="store_true",
        help="3DETF 52ch head에서 tongueOut 제거 후 51ch ARKit export (dataset D3)",
    )
    args = parser.parse_args()

    out = export_onnx(
        checkpoint=args.checkpoint,
        output_path=args.output,
        data_root=args.data_root,
        dataset_name=args.dataset,
        template_idx=args.template_idx,
        opset=args.opset,
        fps=args.fps,
        arkit_51=args.arkit_51,
    )
    print(f"Exported: {out}")
    print(f"Metadata: {out.with_suffix('.json')}")

    if args.verify:
        model, _ = _load_model(args.checkpoint, args.data_root, args.dataset)
        adim = int(dataset_config[args.dataset]["annot_dim"])
        atype = str(dataset_config[args.dataset]["annot_type"])
        wrapper = UniTalkerOnnxWrapper(
            model,
            annot_dim=adim,
            fps=args.fps,
            annot_type=atype,
            drop_tongue_out=args.arkit_51,
        ).eval()
        diff = verify_onnx(out, wrapper, fps=args.fps)
        print(f"Verify OK (max_diff={diff:.2e})")


if __name__ == "__main__":
    main()
