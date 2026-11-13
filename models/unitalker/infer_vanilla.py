#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vanilla UniTalker (원본 demo.py) 방식으로 추론하는 래퍼.
동일 오디오에 대해 Ours vs Vanilla 결과를 비교하기 위한 모듈.

Vanilla demo.py와의 동일점/차이점을 InferenceLogger에 상세히 기록하여,
Ours 추론 로그와 나란히 비교할 수 있게 한다.

주요 차이점 (Vanilla vs Ours):
  1. 오디오 전처리: Vanilla는 전체 오디오를 processor로 1차 처리한 뒤,
     split_long_audio 내부에서 각 구간을 또 processor로 2차 처리 (double processing).
     Ours는 raw 오디오를 분할 후 각 구간을 1회만 처리 (single processing).
  2. dataset_config: Vanilla D3는 3DETF_blendshape_weight(52-dim),
     Ours D3는 inhouse_blendshape_weight(51-dim).
     Ours는 pretrained 감지 시 D3_3DETF로 전환하여 3DETF head를 사용.
  3. condition_id: Vanilla는 데이터셋별 하드코딩된 기본 인덱스 사용,
     Ours는 사용자 지정 speaker + 자동 offset 매핑.
  4. template: Vanilla는 scale 적용 (D3=1.0이라 무차이).
"""
import os
import sys
import time as _time
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unitalker.model.unitalker import UniTalker
from core.utils.utils import get_audio_encoder_dim

# ── Vanilla dataset_config (원본 그대로) ──
# models/UniTalker_Vanilla/UniTalker-main/dataset/dataset_config.py 원본 복사
_VANILLA_DATASET_CONFIG = {
    'D0': {
        'dirname': 'D0_BIWI',
        'annot_type': 'BIWI_23370_vertices',
        'scale': 0.2,
        'annot_dim': 23370 * 3,
        'subjects': 6,
        'pca': True,
    },
    'D1': {
        'dirname': 'D1_vocaset',
        'annot_type': 'FLAME_5023_vertices',
        'scale': 1.0,
        'annot_dim': 5023 * 3,
        'subjects': 12,
        'pca': True,
    },
    'D2': {
        'dirname': 'D2_meshtalk',
        'annot_type': 'meshtalk_6172_vertices',
        'scale': 0.001,
        'annot_dim': 6172 * 3,
        'subjects': 13,
        'pca': True,
    },
    'D3': {
        'dirname': 'D3D4_3DETF/D3_HDTF',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 141,
        'pca': False,
    },
    'D4': {
        'dirname': 'D3D4_3DETF/D4_RAVDESS',
        'annot_type': '3DETF_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 52,
        'subjects': 24,
        'pca': False,
    },
    'D5': {
        'dirname': 'D5_unitalker_faceforensics++',
        'annot_type': 'flame_params_from_dadhead',
        'scale': 1.0,
        'annot_dim': 413,
        'subjects': 719,
        'pca': False,
    },
    'D6': {
        'dirname': 'D6_unitalker_Chinese_speech',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 8,
        'pca': False,
    },
    'D7': {
        'dirname': 'D7_unitalker_song',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 11,
        'pca': False,
    },
}

# Vanilla demo.py의 하드코딩된 condition_id / template_id
_VANILLA_CONDITION_ID_CONFIG = {
    'D0': 3, 'D1': 3, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 4, 'D7': 0,
}
_VANILLA_TEMPLATE_ID_CONFIG = {
    'D0': 3, 'D1': 3, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 4, 'D7': 0,
}

# Vanilla의 training dataset 순서 (demo.py에서 offset 누적에 사용)
_VANILLA_DATASET_ORDER = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']


def _vanilla_split_long_audio(audio, processor):
    """Vanilla demo.py의 split_long_audio 원본 그대로.
    주의: audio가 이미 processor로 1차 처리된 상태에서 또 processor를 적용한다 (double processing).
    """
    sr = 16000
    a, b = 25, 5
    total_length = len(audio) / sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    in_audio_split_list = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for i in range(reps):
        audio_split = audio[start:end]
        audio_split = np.squeeze(
            processor(audio_split, sampling_rate=sr).input_values)
        in_audio_split_list.append(audio_split)
        start += step
        end += step
    return in_audio_split_list


def _vanilla_merge_out_list(out_list, fps):
    """Vanilla demo.py의 merge_out_list 원본 그대로."""
    if len(out_list) == 1:
        return out_list[0]
    a, b = 25, 5
    left_weight = np.linspace(1, 0, b * fps)[:, np.newaxis]
    right_weight = 1 - left_weight
    a = a * fps
    b = b * fps
    offset = a - b
    out_length = len(out_list[-1]) + offset * (len(out_list) - 1)
    merged_out = np.empty((out_length, out_list[-1].shape[-1]),
                          dtype=out_list[-1].dtype)
    merged_out[:a] = out_list[0]
    for out_piece in out_list[1:]:
        merged_out[a - b:a] = (left_weight * merged_out[a - b:a]
                               + right_weight * out_piece[:b])
        merged_out[a:a + offset] = out_piece[b:]
        a += offset
    return merged_out


def _compute_vanilla_condition_id(target_dataset, identity_idx=None):
    """Vanilla demo.py 방식으로 condition_id 계산.
    condition_id='each' 모드 기준 (demo.py 기본값).

    Args:
        target_dataset: 'D3', 'D4', 'D6', 'D7' 등
        identity_idx: None이면 vanilla 기본값, 아니면 해당 인덱스를 local로 사용

    Returns:
        (global_condition_id, local_condition_id, offset)
    """
    start_idx = 0
    for ds in _VANILLA_DATASET_ORDER:
        id_num = _VANILLA_DATASET_CONFIG[ds]['subjects']
        end_idx = start_idx + id_num
        if ds == target_dataset:
            if identity_idx is not None:
                local_idx = identity_idx % id_num
            else:
                local_idx = _VANILLA_CONDITION_ID_CONFIG[ds]
            global_id = start_idx + local_idx
            return global_id, local_idx, start_idx
        start_idx = end_idx
    raise ValueError(f"Unknown dataset: {target_dataset}")


def infer_vanilla_blendshape(
    audio_path,
    checkpoint_path,
    output_dir,
    data_root='./data',
    dataset_name='D3',
    identity_idx=None,
    fps=30,
    logger=None,
):
    """
    Vanilla UniTalker demo.py 방식으로 Blendshape 추론.

    Args:
        audio_path: 입력 오디오 파일 경로
        checkpoint_path: 모델 체크포인트 경로
        output_dir: 출력 디렉토리
        data_root: 데이터 루트 디렉토리
        dataset_name: 데이터셋 이름 (D3, D4, D6, D7 등). _3DETF 접미사가 있으면 제거.
        identity_idx: 화자 인덱스. None이면 vanilla 기본값 사용.
        fps: 출력 FPS
        logger: InferenceLogger 인스턴스
    """
    import gc
    os.makedirs(output_dir, exist_ok=True)

    # _3DETF 접미사 제거 (vanilla에는 D3_3DETF 같은 키가 없음)
    ds_name = dataset_name.replace('_3DETF', '') if dataset_name.endswith('_3DETF') else dataset_name

    if ds_name not in _VANILLA_DATASET_CONFIG:
        if logger:
            logger.warning(f"Vanilla에 '{ds_name}' 데이터셋 없음, 비교 스킵")
        return None

    if logger:
        logger.section("VANILLA - CHECKPOINT LOAD")
        logger.kv("checkpoint_path", checkpoint_path)
        logger.kv("checkpoint_size", f"{os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")

    t0 = _time.time()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    t_ckpt = _time.time() - t0

    # Vanilla demo.py: checkpoint는 raw state_dict (model_state_dict 키 없음)
    # 하지만 ours 형식의 checkpoint도 지원
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        ckpt_type = "model_state_dict (ours format)"
    else:
        model_state = checkpoint
        ckpt_type = "raw state_dict (vanilla format)"

    if logger:
        logger.kv("load_time", f"{t_ckpt:.3f}s")
        logger.kv("ckpt_type", ckpt_type)

    # ── Vanilla: identity_num ──
    identity_key = 'decoder.learnable_style_emb.weight'
    identity_num = len(model_state[identity_key]) if identity_key in model_state else 100

    if logger:
        logger.kv("identity_num", identity_num)

    # ── Vanilla: audio encoder 감지 ──
    if logger:
        logger.section("VANILLA - MODEL DETECTION")

    from types import SimpleNamespace
    args = SimpleNamespace()

    if 'audio_encoder.masked_spec_embed' in model_state:
        encoder_dim = model_state['audio_encoder.masked_spec_embed'].shape[0]
        if encoder_dim == 1024:
            args.audio_encoder_repo = 'microsoft/wavlm-large'
        else:
            args.audio_encoder_repo = 'microsoft/wavlm-base-plus'
    else:
        args.audio_encoder_repo = 'microsoft/wavlm-base-plus'
        encoder_dim = 768

    if logger:
        logger.kv("audio_encoder_dim", encoder_dim)
        logger.kv("audio_encoder_repo", args.audio_encoder_repo)
        logger.log("[NOTE] Vanilla yaml 기본값은 wavlm-base-plus이지만, "
                   "checkpoint에 맞춰 자동 감지 적용")

    # ── Vanilla: 모델 config ──
    args.freeze_wav2vec = False
    args.interpolate_pos = 1
    args.decoder_dimension = 256
    args.decoder_type = 'conv'
    args.period = 30
    args.headlayer = 1
    args.use_pca = True
    args.pca_dim = 512
    args.data_root = data_root
    args.rec_weight = 1.0
    args.pca_weight = 0.01
    args.blendshape_weight = 0.0001
    args.identity_num = identity_num

    # ── Vanilla: 학습 데이터셋 전체를 model init에 사용 (demo.py 방식) ──
    # 실제 forward에 사용할 head만 필요하지만, vanilla는 전체 head를 빌드
    # 여기서는 target 데이터셋만 빌드 (strict=False로 나머지 무시)
    vanilla_ds_cfg = _VANILLA_DATASET_CONFIG[ds_name]
    annot_type = vanilla_ds_cfg['annot_type']
    scale = vanilla_ds_cfg['scale']

    args.dataset = [ds_name]
    args.audio_encoder_feature_dim = get_audio_encoder_dim(args.audio_encoder_repo)

    if logger:
        logger.kv("target_dataset", ds_name)
        logger.kv("annot_type", annot_type)
        logger.kv("annot_dim", vanilla_ds_cfg['annot_dim'])
        logger.kv("scale", scale)

    # ── Vanilla: condition_id 계산 ──
    if logger:
        logger.section("VANILLA - CONDITION_ID MAPPING")

    global_cid, local_cid, offset = _compute_vanilla_condition_id(ds_name, identity_idx)
    vanilla_default_cid_global, vanilla_default_cid_local, _ = _compute_vanilla_condition_id(ds_name, None)

    if logger:
        logger.kv("vanilla default local_id", vanilla_default_cid_local)
        logger.kv("vanilla default global_id", vanilla_default_cid_global)
        if identity_idx is not None:
            logger.kv("requested identity_idx", identity_idx)
            logger.kv("mapped local_id", local_cid)
            logger.kv("mapped global_id", global_cid)
        else:
            logger.log("identity_idx=None -> vanilla 기본값 사용")
        logger.kv("dataset offset", offset)
        logger.kv("dataset subjects", vanilla_ds_cfg['subjects'])

    condition_id = torch.tensor([global_cid])

    # ── Vanilla: template 로드 (scale 적용) ──
    template_idx_v = _VANILLA_TEMPLATE_ID_CONFIG.get(ds_name, 0)
    dirname = vanilla_ds_cfg['dirname']
    tpl_path = os.path.join(data_root, dirname, 'id_template.npy')
    tpl_all = np.load(tpl_path)
    template = torch.from_numpy(tpl_all[template_idx_v]).float()
    template = scale * template  # Vanilla는 scale 적용
    template = template.reshape(1, -1).cuda()

    if logger:
        logger.kv("template_path", tpl_path)
        logger.kv("template_idx (vanilla default)", template_idx_v)
        logger.kv("template_shape", template.shape)
        logger.kv("scale applied", scale)

    # ── Vanilla: 모델 빌드 ──
    if logger:
        logger.section("VANILLA - MODEL BUILD & LOAD")

    gc.collect()
    torch.cuda.empty_cache()

    t0 = _time.time()
    # Vanilla 모델에는 vanilla dataset_config를 사용해야 함
    # 임시로 dataset_config를 패치
    from models.unitalker.dataset import dataset_config as ds_cfg_module
    original_config = ds_cfg_module.dataset_config.copy()
    ds_cfg_module.dataset_config.update(_VANILLA_DATASET_CONFIG)
    try:
        model = UniTalker(args)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        model.cuda()
    finally:
        # 원래 config 복원
        ds_cfg_module.dataset_config.clear()
        ds_cfg_module.dataset_config.update(original_config)

    t_model = _time.time() - t0

    if logger:
        logger.kv("model_build_time", f"{t_model:.3f}s")
        total_params = sum(p.numel() for p in model.parameters())
        logger.kv("total_params", f"{total_params:,} ({total_params/1e6:.1f}M)")

    # ── Vanilla: 오디오 로드 + DOUBLE PROCESSING ──
    if logger:
        logger.section("VANILLA - AUDIO LOAD (DOUBLE PROCESSING)")
        logger.log("[KEY DIFF] Vanilla는 오디오를 2번 processor 처리함:")
        logger.log("  1) 전체 오디오 -> processor -> normalized")
        logger.log("  2) split -> 각 구간 -> processor -> normalized (2차)")

    t0 = _time.time()
    audio_data, sr = librosa.load(audio_path, sr=16000)
    t_load = _time.time() - t0

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_encoder_repo)

    if logger:
        logger.kv("audio_load_time", f"{t_load:.3f}s")
        logger.kv("raw audio samples", len(audio_data))
        logger.kv("raw audio duration", f"{len(audio_data)/sr:.3f}s")
        logger.kv("raw audio range", f"[{audio_data.min():.6f}, {audio_data.max():.6f}]")

    # 1차 처리 (vanilla demo.py line 153)
    audio_processed = np.squeeze(
        processor(audio_data, sampling_rate=sr).input_values)

    if logger:
        logger.kv("1st processing shape", audio_processed.shape)
        logger.kv("1st processing range", f"[{audio_processed.min():.6f}, {audio_processed.max():.6f}]")

    # 2차 처리 포함된 split (vanilla demo.py line 156 + split_long_audio)
    audio_splits = _vanilla_split_long_audio(audio_processed, processor)

    if logger:
        logger.kv("audio_splits", len(audio_splits))
        for i, split in enumerate(audio_splits):
            logger.kv(f"  split[{i}] shape", split.shape)
            logger.kv(f"  split[{i}] range", f"[{split.min():.6f}, {split.max():.6f}]")

    # ── Vanilla: 추론 ──
    if logger:
        logger.section("VANILLA - INFERENCE (forward pass)")
        logger.kv("annot_type", annot_type)
        logger.kv("condition_id", condition_id.item())
        logger.kv("fps", fps)

    t_infer_start = _time.time()
    out_list = []
    with torch.no_grad():
        for i, audio_split in enumerate(audio_splits):
            t_split = _time.time()
            audio_tensor = torch.Tensor(audio_split[None]).cuda()
            out, _, _, _ = model(
                audio_tensor,
                template,
                face_motion=None,
                style_idx=condition_id.cuda(),
                annot_type=annot_type,
                fps=fps,
            )
            out_np = out.cpu().numpy().squeeze(0)
            out_list.append(out_np)
            t_split_elapsed = _time.time() - t_split
            if logger:
                logger.log(f"split[{i}] forward: {t_split_elapsed:.3f}s, "
                           f"input={audio_tensor.shape}, output={out_np.shape}")

    t_infer_total = _time.time() - t_infer_start
    if logger:
        logger.separator()
        logger.kv("total forward time", f"{t_infer_total:.3f}s")

    # ── Vanilla: 병합 ──
    if logger:
        logger.section("VANILLA - POST-PROCESSING")

    if len(out_list) > 1:
        result = _vanilla_merge_out_list(out_list, fps)
        if logger:
            logger.log(f"{len(out_list)}개 구간 병합 완료")
    else:
        result = out_list[0]
        if logger:
            logger.log("단일 구간, 병합 불필요")

    blendshape_np = np.array(result) if not isinstance(result, np.ndarray) else result

    if logger:
        logger.kv("output shape", blendshape_np.shape)
        logger.kv("output frames", blendshape_np.shape[0])
        logger.kv("output channels", blendshape_np.shape[1])
        logger.kv("output duration", f"{blendshape_np.shape[0] / fps:.3f}s @ {fps}fps")
        logger.kv("value range", f"[{blendshape_np.min():.6f}, {blendshape_np.max():.6f}]")
        logger.kv("value mean", f"{blendshape_np.mean():.6f}")
        logger.kv("value std", f"{blendshape_np.std():.6f}")

    # NPY 저장
    audio_name = Path(audio_path).stem
    tag = audio_name.replace("_input", "")
    vanilla_npy_path = os.path.join(output_dir, f"{tag}_vanilla_pred.npy")
    np.save(vanilla_npy_path, blendshape_np)

    if logger:
        logger.kv("vanilla_npy_saved", vanilla_npy_path)

    # GPU 메모리 해제
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if logger:
        logger.log("GPU 메모리 해제 완료")

    return blendshape_np


def compare_ours_vs_vanilla(ours_np, vanilla_np, logger):
    """Ours vs Vanilla 결과를 정량적으로 비교하여 로그에 기록.

    Args:
        ours_np: Ours 추론 결과 (T, C)
        vanilla_np: Vanilla 추론 결과 (T, C')
        logger: InferenceLogger
    """
    logger.section("COMPARISON: OURS vs VANILLA")

    # Shape 비교
    logger.kv("ours shape", ours_np.shape)
    logger.kv("vanilla shape", vanilla_np.shape)

    ours_T, ours_C = ours_np.shape
    van_T, van_C = vanilla_np.shape

    if ours_T != van_T:
        logger.warning(f"프레임 수 불일치: ours={ours_T}, vanilla={van_T}")
        min_T = min(ours_T, van_T)
        logger.log(f"앞쪽 {min_T} 프레임만 비교")
        ours_cmp = ours_np[:min_T]
        van_cmp = vanilla_np[:min_T]
    else:
        min_T = ours_T
        ours_cmp = ours_np
        van_cmp = vanilla_np

    # 채널 수 맞추기 (vanilla 52-dim vs ours 51-dim 인 경우)
    min_C = min(ours_C, van_C)
    if ours_C != van_C:
        logger.warning(f"채널 수 불일치: ours={ours_C}, vanilla={van_C} -> 앞쪽 {min_C}채널 비교")
        ours_cmp = ours_cmp[:, :min_C]
        van_cmp = van_cmp[:, :min_C]

    # 전체 통계
    diff = ours_cmp - van_cmp
    abs_diff = np.abs(diff)

    logger.separator()
    logger.log("[전체 통계]")
    logger.kv("비교 프레임 수", min_T)
    logger.kv("비교 채널 수", min_C)
    logger.kv("MSE", f"{np.mean(diff**2):.8f}")
    logger.kv("MAE", f"{np.mean(abs_diff):.8f}")
    logger.kv("Max Abs Diff", f"{abs_diff.max():.8f}")
    logger.kv("Max Diff 위치", f"frame={np.unravel_index(abs_diff.argmax(), abs_diff.shape)[0]}, "
                               f"ch={np.unravel_index(abs_diff.argmax(), abs_diff.shape)[1]}")

    # 값 범위 비교
    logger.separator()
    logger.log("[값 범위 비교]")
    logger.kv("ours  range", f"[{ours_cmp.min():.6f}, {ours_cmp.max():.6f}]")
    logger.kv("vanilla range", f"[{van_cmp.min():.6f}, {van_cmp.max():.6f}]")
    logger.kv("ours  mean/std", f"{ours_cmp.mean():.6f} / {ours_cmp.std():.6f}")
    logger.kv("vanilla mean/std", f"{van_cmp.mean():.6f} / {van_cmp.std():.6f}")

    # 동일 여부
    is_identical = np.allclose(ours_cmp, van_cmp, atol=1e-6)
    logger.separator()
    if is_identical:
        logger.log("[RESULT] Ours와 Vanilla 결과가 동일합니다 (atol=1e-6)")
    else:
        logger.log("[RESULT] Ours와 Vanilla 결과가 다릅니다")

        # 채널별 MAE Top 10
        per_channel_mae = np.mean(abs_diff, axis=0)
        top_channels = np.argsort(per_channel_mae)[::-1][:10]
        logger.log("[채널별 MAE Top 10]")
        for rank, ch in enumerate(top_channels):
            logger.kv(f"  #{rank+1} ch={ch}", f"MAE={per_channel_mae[ch]:.8f}")

        # 프레임별 MAE Top 5
        per_frame_mae = np.mean(abs_diff, axis=1)
        top_frames = np.argsort(per_frame_mae)[::-1][:5]
        logger.log("[프레임별 MAE Top 5]")
        for rank, fr in enumerate(top_frames):
            logger.kv(f"  #{rank+1} frame={fr}", f"MAE={per_frame_mae[fr]:.8f}, "
                                                  f"time={fr/30:.3f}s")

        # Pearson 상관계수 (전체 flatten)
        from scipy import stats
        corr, p_val = stats.pearsonr(ours_cmp.flatten(), van_cmp.flatten())
        logger.separator()
        logger.kv("Pearson r (전체)", f"{corr:.8f}")
        logger.kv("p-value", f"{p_val:.2e}")
