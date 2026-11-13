import os
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from models.unitalker.dataset.dataset_config import dataset_config
from models.unitalker.model.unitalker import UniTalker
from core.utils.utils import get_audio_encoder_dim


def split_long_audio(audio: np.ndarray, processor: Wav2Vec2FeatureExtractor):
    a, b = 25, 5
    sr = 16000
    total_length = len(audio) / sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    in_audio_split_list = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for _ in range(reps):
        audio_split = audio[start:end]
        audio_split = np.squeeze(processor(audio_split, sampling_rate=sr).input_values)
        in_audio_split_list.append(audio_split)
        start += step
        end += step
    return in_audio_split_list


def merge_out_list(out_list: list, fps: int):
    if len(out_list) == 1:
        return out_list[0]
    a, b = 25, 5
    left_weight = np.linspace(1, 0, b * fps)[:, np.newaxis]
    right_weight = 1 - left_weight
    a = a * fps
    b = b * fps
    offset = a - b
    out_length = len(out_list[-1]) + offset * (len(out_list) - 1)
    merged_out = np.empty((out_length, out_list[-1].shape[-1]), dtype=out_list[-1].dtype)
    merged_out[:a] = out_list[0]
    for out_piece in out_list[1:]:
        merged_out[a - b : a] = left_weight * merged_out[a - b : a] + right_weight * out_piece[:b]
        merged_out[a : a + offset] = out_piece[b:]
        a += offset
    return merged_out


def _build_args_from_ckpt(checkpoint_path: str, data_root: str, dataset_name: str):
    from types import SimpleNamespace

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model_state_dict", ckpt)

    args = SimpleNamespace()
    args.audio_encoder_repo = "microsoft/wavlm-base-plus"
    if "audio_encoder.masked_spec_embed" in model_state:
        dim = model_state["audio_encoder.masked_spec_embed"].shape[0]
        args.audio_encoder_repo = "microsoft/wavlm-large" if dim == 1024 else "microsoft/wavlm-base-plus"
    args.freeze_wav2vec = False
    args.interpolate_pos = 1
    args.decoder_dimension = 256
    args.decoder_type = "conv"
    args.period = 30
    args.headlayer = 1
    args.use_pca = False
    args.pca_dim = 512
    args.data_root = data_root
    args.rec_weight = 1.0
    args.pca_weight = 0.0
    args.blendshape_weight = 0.0001
    args.dataset = [dataset_name]
    args.audio_encoder_feature_dim = get_audio_encoder_dim(args.audio_encoder_repo)
    id_key = "decoder.learnable_style_emb.weight"
    args.identity_num = len(model_state[id_key]) if id_key in model_state else 1
    return args, model_state


def _load_model_for_infer(
    checkpoint_path: str,
    data_root: str,
    dataset_for_blendshape: str,
    device: torch.device,
):
    dataset_name = dataset_for_blendshape.replace("_3DETF", "")
    if dataset_name not in dataset_config:
        if "metahuman" in dataset_for_blendshape.lower():
            dataset_name = "D8"
        else:
            dataset_name = "D7"

    args, model_state = _build_args_from_ckpt(checkpoint_path, data_root, dataset_name)
    model = UniTalker(args)
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in model_state.items():
        nk = k
        if "pos_conv_embed.conv.weight_g" in k:
            nk = k.replace("pos_conv_embed.conv.weight_g", "pos_conv_embed.conv.parametrizations.weight.original0")
        elif "pos_conv_embed.conv.weight_v" in k:
            nk = k.replace("pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
        if nk in model_keys:
            remapped[nk] = v
    model.load_state_dict(remapped, strict=False)
    model.to(device).eval()

    scale = float(dataset_config[dataset_name]["scale"])
    annot_type = dataset_config[dataset_name]["annot_type"]
    annot_dim = int(dataset_config[dataset_name]["annot_dim"])
    return model, args, dataset_name, scale, annot_type, annot_dim


def _truncate_audio_annot_like_training(
    audio_array: np.ndarray,
    sr: int,
    annot_data: np.ndarray | None,
    fps: int,
    max_sec: float = 20.0,
):
    """DataItem.truncate와 동일: min(audio, annot, max_sec)."""
    audio_duration = len(audio_array) / sr
    if annot_data is not None:
        annot_duration = len(annot_data) / fps
        duration = min(audio_duration, annot_duration, max_sec)
    else:
        duration = min(audio_duration, max_sec)
    audio_length = round(duration * sr)
    annot_length = round(duration * fps)
    audio_trunc = audio_array[:audio_length]
    if annot_data is not None:
        annot_trunc = annot_data[:annot_length]
    else:
        annot_trunc = None
    return audio_trunc, annot_trunc, duration, annot_length


def infer_audio_to_blendshape_train_aligned(
    audio_path: str,
    checkpoint_path: str,
    data_root: str,
    dataset_for_blendshape: str,
    identity_idx: int = 0,
    template_idx: int = 0,
    fps: int = 30,
    max_sec: float = 20.0,
    gt_npy_path: str | None = None,
    output_dir: str | None = None,
):
    """
    학습(DataItem)과 동일 경로: 앞 max_sec(기본 20s)만 사용, split/merge 없음.
    frame_num은 GT 길이(학습 시 face_motion.shape[1])와 동일하게 맞춤.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args, dataset_name, scale, annot_type, annot_dim = _load_model_for_infer(
        checkpoint_path, data_root, dataset_for_blendshape, device
    )
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_encoder_repo)

    sr = 16000
    if audio_path.endswith(".npy"):
        audio_raw = np.load(audio_path).astype(np.float32)
    else:
        audio_raw, _ = librosa.load(audio_path, sr=sr)

    gt = None
    if gt_npy_path is not None and os.path.exists(gt_npy_path):
        gt = np.load(gt_npy_path).astype(np.float32)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

    audio_trunc, gt_trunc, duration, annot_length = _truncate_audio_annot_like_training(
        audio_raw, sr, gt, fps, max_sec=max_sec
    )
    audio_feat = np.squeeze(processor(audio_trunc, sampling_rate=sr).input_values).astype(np.float32)

    try:
        tpl_path = os.path.join(data_root, dataset_config[dataset_name]["dirname"], "id_template.npy")
        tpl_all = np.load(tpl_path).astype(np.float32)
        ti = max(0, min(int(template_idx), len(tpl_all) - 1))
        template = scale * tpl_all[ti]
    except Exception:
        template = np.zeros((annot_dim,), dtype=np.float32)
    template = torch.tensor(template.reshape(1, -1), dtype=torch.float32, device=device)
    style_idx = torch.tensor([identity_idx % max(args.identity_num, 1)], dtype=torch.long, device=device)

    # 학습과 동일하게 face_motion 길이로 frame_num 고정
    frame_motion = torch.zeros((1, annot_length, annot_dim), dtype=torch.float32, device=device)
    audio_tensor = torch.tensor(audio_feat[None], dtype=torch.float32, device=device)

    with torch.no_grad():
        out, _, _, _ = model(
            audio_tensor,
            template,
            frame_motion,
            style_idx=style_idx,
            annot_type=annot_type,
            fps=fps,
        )
    blendshape = out.cpu().numpy().squeeze(0).astype(np.float32)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(Path(output_dir) / "latest_pred_train_aligned.npy", blendshape)

    return {
        "blendshape": blendshape,
        "duration_sec": duration,
        "num_frames": annot_length,
        "mode": "train_aligned",
    }


def infer_audio_to_blendshape(
    audio_path: str,
    checkpoint_path: str,
    output_dir: str,
    data_root: str,
    dataset_for_blendshape: str,
    identity_idx: int = 0,
    template_idx: int = 0,
    fps: int = 30,
    logger=None,
):
    dataset_name = dataset_for_blendshape.replace("_3DETF", "")
    if dataset_name not in dataset_config:
        if "metahuman" in dataset_for_blendshape.lower():
            dataset_name = "D8"
        elif dataset_for_blendshape in dataset_config:
            dataset_name = dataset_for_blendshape
        else:
            dataset_name = "D7"

    args, model_state = _build_args_from_ckpt(checkpoint_path, data_root, dataset_name)
    model = UniTalker(args)
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in model_state.items():
        nk = k
        if "pos_conv_embed.conv.weight_g" in k:
            nk = k.replace("pos_conv_embed.conv.weight_g", "pos_conv_embed.conv.parametrizations.weight.original0")
        elif "pos_conv_embed.conv.weight_v" in k:
            nk = k.replace("pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
        if nk in model_keys:
            remapped[nk] = v
    model.load_state_dict(remapped, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_encoder_repo)
    audio_data, sr = librosa.load(audio_path, sr=16000)
    audio_splits = split_long_audio(audio_data, processor)

    scale = float(dataset_config[dataset_name]["scale"])
    try:
        tpl_path = os.path.join(
            data_root, dataset_config[dataset_name]["dirname"], "id_template.npy"
        )
        tpl_all = np.load(tpl_path).astype(np.float32)
        ti = max(0, min(int(template_idx), len(tpl_all) - 1))
        template = scale * tpl_all[ti]
    except Exception:
        annot_dim = dataset_config[dataset_name]["annot_dim"]
        template = np.zeros((annot_dim,), dtype=np.float32)
    template = torch.tensor(template.reshape(1, -1), dtype=torch.float32, device=device)
    style_idx = identity_idx % max(args.identity_num, 1)
    style_idx = torch.tensor([style_idx], dtype=torch.long, device=device)
    annot_type = dataset_config[dataset_name]["annot_type"]

    out_list = []
    with torch.no_grad():
        for seg in audio_splits:
            seg = torch.tensor(seg[None], dtype=torch.float32, device=device)
            out, _, _, _ = model(
                seg,
                template,
                face_motion=None,
                style_idx=style_idx,
                annot_type=annot_type,
                fps=fps,
            )
            out_list.append(out.cpu().numpy().squeeze(0))
    blendshape = merge_out_list(out_list, fps).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(Path(output_dir) / "latest_pred.npy", blendshape)
    return {"blendshape": blendshape}
