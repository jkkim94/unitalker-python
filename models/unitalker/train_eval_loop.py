import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import os
import csv

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils.utils import log_datasetloss, write_to_tensorboard
from core.utils.utils import write_to_wandb
from models.unitalker.loss.loss import get_annot_types_without_lve

# 학습 1 step (iteration): batch 로드 → forward → loss → backward → optimizer.step
# 상세 shape/용어: docs/UNITALKER_TRAINING_CODE_TRACE.md


def _e1_first_batch_guard(
    args, loss_submodule, rec_weight, bs_weight, lve_weight,
    rec_loss, bs_loss, mouth_loss, loss, model,
) -> None:
    """Stop before the first optimizer step if E1 is not Control MSE + linear lip geometry."""
    import json
    from pathlib import Path

    from models.unitalker.loss.metahuman_riglogic_loss import MetahumanBlendShapeLoss

    problems = []
    if not isinstance(loss_submodule, MetahumanBlendShapeLoss):
        problems.append(f'loss module is {type(loss_submodule).__name__}, not MetahumanBlendShapeLoss')
    mouth_idx = getattr(loss_submodule, 'mouth_idx', None)
    n_lip = int(mouth_idx.numel()) if mouth_idx is not None else 0
    if n_lip <= 0:
        problems.append('mouth_idx is empty, so mouth_metric is not the lip-vertex term')
    if abs(rec_weight) > 0:
        problems.append(f'rec_weight is {rec_weight}; full-head vertex MSE would enter the objective')
    if abs(bs_weight - 1.0) > 1e-8:
        problems.append(f'blendshape_weight is {bs_weight}, expected 1.0 for control MSE')
    if abs(lve_weight - 0.05) > 1e-8:
        problems.append(f'lve_weight is {lve_weight}, expected 0.05')
    raw_control = float(bs_loss.detach())
    raw_lip = float(mouth_loss.detach())
    raw_rec = float(rec_loss.detach())
    weighted_control = bs_weight * raw_control
    weighted_lip = lve_weight * raw_lip
    grad_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_sq += float(param.grad.detach().float().pow(2).sum())
    grad_norm = grad_sq ** 0.5
    if grad_norm <= 0:
        problems.append('model parameter gradient is zero')
    if not torch.isfinite(loss):
        problems.append('loss is not finite')
    if weighted_control > 0 and weighted_lip > 10.0 * weighted_control:
        problems.append(
            f'weighted lip {weighted_lip:.6f} is more than 10x weighted control {weighted_control:.6f}'
        )
    record = {
        'loss_module': type(loss_submodule).__name__,
        'lip_vertex_count': n_lip,
        'control_mse_raw': raw_control,
        'control_mse_weighted': weighted_control,
        'linear_lip_geometry_raw': raw_lip,
        'linear_lip_geometry_weighted': weighted_lip,
        'full_head_vertex_mse_raw': raw_rec,
        'full_head_vertex_mse_weighted': rec_weight * raw_rec,
        'objective': float(loss.detach()),
        'model_grad_norm': grad_norm,
        'geometry_definition': 'MetahumanBlendShapeLoss.mouth_metric[0]: mean over frames of max squared lip-vertex distance',
        'problems': problems,
    }
    out = Path('artifacts/e1_pilot')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'FIRST_BATCH.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    line = (
        '[E1-FIRST-BATCH] '
        f"control_mse {raw_control:.6f} weighted {weighted_control:.6f} | "
        f"linear_lip {raw_lip:.6f} weighted {weighted_lip:.6f} | "
        f"full_head_vertex_mse {raw_rec:.6f} weight {rec_weight} | "
        f"grad_norm {grad_norm:.6f} lip_verts {n_lip}"
    )
    if getattr(args, 'logger', None) is not None:
        args.logger.info(line)
    print(line, flush=True)
    if problems:
        raise SystemExit('E1 pilot stopped before the first optimizer step: ' + '; '.join(problems))


def train_epoch(data_loader, model, loss_module, optimizer, epoch, args):
    # 메모리 모니터링 (psutil이 있으면 사용)
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info_start = process.memory_info()
        if hasattr(args, 'logger'):
            args.logger.info(f'[MEM] Epoch {epoch} train START - RSS: {mem_info_start.rss / 1024**3:.2f} GB')
        enable_mem_monitoring = True
    except ImportError:
        enable_mem_monitoring = False
        if hasattr(args, 'logger'):
            args.logger.warning('[MEM] psutil not available, memory monitoring disabled')
    
    model.train()  # dropout 등 train 모드 (batch norm은 D8 TCN에서 LN 사용)
    train_meter = args.train_meter
    metric_name_list = args.train_metric_name_list
    zero_loss = torch.tensor(0.0)
    mixed_dataset_name = 'mixed_dataset'
    pbar = tqdm(data_loader)
    pbar.set_description(f'Epoch: {epoch} train')
    
    batch_count = 0
    for batch in pbar:  # iteration 1회 = config batch_size개 clip (D8는 보통 1)
        # GT motion [B,T,C], template [B,1,C], waveform [B,N_audio], speaker id [B]
        # MetaHuman D8 예: B=1, T≈duration*30, C=200, N_audio≈duration*16000
        data = batch['data'].to(args.device, non_blocking=True)
        template = batch['template'].to(args.device, non_blocking=True)
        audio = batch['audio'].to(args.device, non_blocking=True)
        identity = batch['id'].to(args.device, non_blocking=True)
        if args.random_id_prob > 0.0:
            if np.random.random() < args.random_id_prob:
                identity = identity.fill_(args.identity_num - 1)
        annot_type = batch['annot_type'][0]
        dataset_name = batch['dataset_name'][0]
        loss_dataset_name = dataset_name
        if dataset_name not in train_meter:
            dataset_name = args.dataset[0]

        viseme_labels_batch = batch.get('viseme_labels', None)
        if viseme_labels_batch is not None:
            viseme_labels_batch = viseme_labels_batch.to(args.device, non_blocking=True)
        # forward (순전파): audio → WavLM → decoder → head → out_motion [B,T,C]
        out_motion, out_pca, gt_pca, viseme_logits = model(
            audio, template, data, style_idx=identity, annot_type=annot_type,
            viseme_labels=viseme_labels_batch)

        # loss: rec_weight·rec(vertex MSE) + blendshape_weight·bs(control MSE) + lve_weight·LVE
        rec_weight = float(getattr(args, 'rec_weight', 1.0))
        bs_weight = float(getattr(args, 'blendshape_weight', 0.0))
        loss_submodule = loss_module.loss_module_dict[annot_type] if annot_type in loss_module.loss_module_dict else None
        if loss_submodule is not None and hasattr(loss_submodule, 'forward_separate'):
            _, rec_loss, bs_loss = loss_submodule.forward_separate(
                out_motion, data, **_dataset_kwargs(loss_submodule, loss_dataset_name))
            total_loss = rec_weight * rec_loss + bs_weight * bs_loss
        else:
            rec_loss = loss_module(out_motion, data, annot_type)
            bs_loss = zero_loss
            total_loss = rec_weight * rec_loss
        if out_pca is not None:
            pca_loss = loss_module.pca_loss(out_pca, gt_pca)
        else:
            pca_loss = zero_loss
        # mouth_metric: vertex LVE(BlendShapeLoss) 또는 lip-control MSE(ControlLoss).
        # lve_weight>0 일 때만 backward에 포함. 명칭은 validate_epoch에서 분리 기록.
        lve_weight = getattr(args, 'lve_weight', 0.0)
        annot_types_without_lve = get_annot_types_without_lve(args)
        if (
            annot_type not in annot_types_without_lve
            and loss_submodule is not None
            and hasattr(loss_submodule, 'mouth_metric')
            and lve_weight > 0
        ):
            mouth_loss = loss_submodule.mouth_metric(out_motion, data)[0]
        else:
            mouth_loss = zero_loss
        # viseme auxiliary classification loss
        viseme_aux_weight = getattr(args, 'viseme_aux_weight', 0.0)
        if viseme_logits is not None and viseme_aux_weight > 0 and 'viseme_labels' in batch:
            viseme_labels = batch['viseme_labels'].to(args.device, non_blocking=True)
            B, T, C = viseme_logits.shape
            viseme_loss = nn.functional.cross_entropy(
                viseme_logits.reshape(B * T, C),
                viseme_labels.reshape(B * T),
                ignore_index=-1)
        else:
            viseme_loss = zero_loss
        # multi-task loss (scalar). autograd graph 끝 — 여기서 backward 시작
        loss = total_loss + args.pca_weight * pca_loss + lve_weight * mouth_loss + viseme_aux_weight * viseme_loss
        if bool(getattr(args, 'e1_pilot_guard', False)) and not torch.isfinite(loss):
            raise SystemExit(
                f'E1 pilot stopped: loss is not finite at epoch {epoch} batch {batch_count}'
            )
        optimizer.zero_grad()  # 이전 step gradient 초기화
        loss.backward()  # 역전파: ∂loss/∂param 계산 → param.grad에 저장
        if bool(getattr(args, 'e1_pilot_guard', False)) and batch_count == 0 and epoch == 1:
            _e1_first_batch_guard(
                args, loss_submodule, rec_weight, bs_weight, float(lve_weight),
                rec_loss, bs_loss, mouth_loss, loss, model,
            )
        grad_clip_norm = float(getattr(args, 'grad_clip_norm', 0.0))
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # grad 폭주 방지
        optimizer.step()  # Adam: W ← W - lr * (보정된 grad)
        for k, v in zip(metric_name_list, (rec_loss, pca_loss, bs_loss, mouth_loss, viseme_loss)):
            train_meter[dataset_name][k].update(v.item())
            train_meter[mixed_dataset_name][k].update(v.item())
        if (
            annot_type == 'metahuman_control'
            and mouth_loss is not zero_loss
            and 'lip_control_lve' in train_meter[dataset_name]
        ):
            lip_val = mouth_loss.item()
            train_meter[dataset_name]['lip_control_lve'].update(lip_val)
            train_meter[mixed_dataset_name]['lip_control_lve'].update(lip_val)
        
        # 메모리 모니터링: 100 배치마다
        batch_count += 1
        if enable_mem_monitoring and batch_count % 100 == 0:
            mem_info = process.memory_info()
            if hasattr(args, 'logger'):
                args.logger.info(f'[MEM] Epoch {epoch} train batch {batch_count} - RSS: {mem_info.rss / 1024**3:.2f} GB (delta: {(mem_info.rss - mem_info_start.rss) / 1024**3:.2f} GB)')

    # 메모리 모니터링: epoch 종료 시
    if enable_mem_monitoring:
        mem_info_end = process.memory_info()
        if hasattr(args, 'logger'):
            args.logger.info(f'[MEM] Epoch {epoch} train END - RSS: {mem_info_end.rss / 1024**3:.2f} GB (delta: {(mem_info_end.rss - mem_info_start.rss) / 1024**3:.2f} GB)')
    
    log_datasetloss(args.logger, epoch, 'train', train_meter, only_mix=True)
    write_to_tensorboard(
        args.writer, epoch, 'train', train_meter, only_mix=False)
    # W&B 로깅 (commit=False로 설정하여 val/test와 함께 커밋)
    if hasattr(args, 'wandb_logger'):
        wandb_step = epoch
        if hasattr(args, 'start_epoch') and hasattr(args, 'WANDB') and getattr(args.WANDB, 'resume_step', None) is not None:
            wandb_step = args.WANDB.resume_step + (epoch - args.start_epoch)
        write_to_wandb(args.wandb_logger, epoch, 'train', train_meter, only_mix=False, commit=False, step_override=wandb_step)

    # reset 전에 rec_L2_loss 캡처 (train.py 조기 종료 판단에 사용)
    rec_L2_loss_avg = train_meter['mixed_dataset']['rec_L2_loss'].avg if 'rec_L2_loss' in train_meter['mixed_dataset'] else None
    for dataset_name, sub_metric_dict in train_meter.items():
        for metric_name in sub_metric_dict.keys():
            sub_metric_dict[metric_name].reset()
    return rec_L2_loss_avg


def validate_epoch(data_loader, model, loss_module, epoch, phase, args):
    # 메모리 모니터링 (psutil이 있으면 사용)
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info_start = process.memory_info()
        if hasattr(args, 'logger'):
            args.logger.info(f'[MEM] Epoch {epoch} {phase} START - RSS: {mem_info_start.rss / 1024**3:.2f} GB')
        enable_mem_monitoring = True
    except ImportError:
        enable_mem_monitoring = False
    
    assert phase in ('val', 'test')
    model.eval()  # inference 모드 — dropout off, BN/LN eval
    val_meter = args.val_meter
    metric_name_list = args.val_metric_name_list
    mixed_dataset_name = 'mixed_dataset'

    # validation/test: forward + metric만. backward/optimizer 없음 (가중치 갱신 X)
    with torch.no_grad():
        pbar = tqdm(data_loader)
        pbar.set_description(f'Epoch: {epoch} {phase:<5}')
        for batch in pbar:
            data = batch['data'].to(args.device, non_blocking=True)
            template = batch['template'].to(args.device, non_blocking=True)
            audio = batch['audio'].to(args.device, non_blocking=True)
            identity = batch['id'].to(args.device, non_blocking=True)
            annot_type = batch['annot_type'][0]
            dataset_name = batch['dataset_name'][0]
            loss_dataset_name = dataset_name
            if dataset_name not in val_meter:
                dataset_name = args.dataset[0]
            if args.random_id_prob >= 1.0:
                identity = identity.fill_(args.identity_num - 1)
            viseme_labels_batch = batch.get('viseme_labels', None)
            if viseme_labels_batch is not None:
                viseme_labels_batch = viseme_labels_batch.to(args.device, non_blocking=True)
            out_motion, _, _, _ = model(
                audio,
                template,
                data,
                style_idx=identity,
                annot_type=annot_type,
                viseme_labels=viseme_labels_batch)
            # 분리된 loss 계산 (blendshape annot_type인 경우)
            loss_submodule = loss_module.loss_module_dict[annot_type] if annot_type in loss_module.loss_module_dict else None
            if loss_submodule is not None and hasattr(loss_submodule, 'forward_separate'):
                _, rec_loss, bs_loss = loss_submodule.forward_separate(
                    out_motion, data, **_dataset_kwargs(loss_submodule, loss_dataset_name))
                val_meter[dataset_name]["rec_L2_loss"].update(rec_loss.item())
                val_meter[mixed_dataset_name]["rec_L2_loss"].update(rec_loss.item())
                if annot_type == "metahuman_control":
                    val_meter[dataset_name]["control_MSE"].update(bs_loss.item())
                    val_meter[mixed_dataset_name]["control_MSE"].update(bs_loss.item())
                else:
                    val_meter[dataset_name]["blendshape_loss"].update(bs_loss.item())
                    val_meter[mixed_dataset_name]["blendshape_loss"].update(bs_loss.item())
                # mouth_metric 의미는 loss 종류에 따름:
                # - MetahumanBlendShapeLoss (controls_to_vertices): 진짜 lip vertex LVE/LVD
                # - MetahumanControlLoss: lip control MSE/MAE — LVE 키에 넣지 않음
                if hasattr(loss_submodule, 'mouth_metric'):
                    mouth_metric_L2, mouth_metric_L2_norm = loss_submodule.mouth_metric(
                        out_motion, data)
                    is_vertex_lve = hasattr(loss_submodule, 'controls_to_vertices')
                    if is_vertex_lve:
                        if "LVE" in val_meter[dataset_name]:
                            val_meter[dataset_name]["LVE"].update(mouth_metric_L2.item())
                            val_meter[mixed_dataset_name]["LVE"].update(mouth_metric_L2.item())
                        if "LVD" in val_meter[dataset_name]:
                            val_meter[dataset_name]["LVD"].update(mouth_metric_L2_norm.item())
                            val_meter[mixed_dataset_name]["LVD"].update(mouth_metric_L2_norm.item())
                    else:
                        if "mouth_control_mse" in val_meter[dataset_name]:
                            val_meter[dataset_name]["mouth_control_mse"].update(
                                mouth_metric_L2.item())
                            val_meter[mixed_dataset_name]["mouth_control_mse"].update(
                                mouth_metric_L2.item())
                        if "mouth_control_mae" in val_meter[dataset_name]:
                            val_meter[dataset_name]["mouth_control_mae"].update(
                                mouth_metric_L2_norm.item())
                            val_meter[mixed_dataset_name]["mouth_control_mae"].update(
                                mouth_metric_L2_norm.item())
                        # 하위호환: 구 로그 키 lip_control_lve (= mouth_control_mse)
                        if "lip_control_lve" in val_meter[dataset_name]:
                            lip_val = mouth_metric_L2.item()
                            val_meter[dataset_name]["lip_control_lve"].update(lip_val)
                            val_meter[mixed_dataset_name]["lip_control_lve"].update(lip_val)
            else:
                rec_loss = loss_module(out_motion, data, annot_type)
                bs_loss = torch.tensor(0.0, device=out_motion.device)
                if annot_type in get_annot_types_without_lve(args):
                    val_meter[dataset_name]["rec_L2_loss"].update(rec_loss.item())
                    val_meter[mixed_dataset_name]["rec_L2_loss"].update(rec_loss.item())
                else:
                    mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(
                        out_motion, data, annot_type)
                    for k, v in zip(
                        metric_name_list,
                        (rec_loss, mouth_metric_L2, mouth_metric_L2_norm, bs_loss),
                    ):
                        val_meter[dataset_name][k].update(v.item())
                        val_meter[mixed_dataset_name][k].update(v.item())
        
        # 메모리 모니터링: validation/test 종료 시
        if enable_mem_monitoring:
            mem_info_end = process.memory_info()
            if hasattr(args, 'logger'):
                args.logger.info(f'[MEM] Epoch {epoch} {phase} END - RSS: {mem_info_end.rss / 1024**3:.2f} GB (delta: {(mem_info_end.rss - mem_info_start.rss) / 1024**3:.2f} GB)')
        
        if getattr(args, 'logger', None) is not None:
            log_datasetloss(
                args.logger, epoch, phase, val_meter, only_mix=True)
        if getattr(args, 'writer', None) is not None:
            write_to_tensorboard(
                args.writer, epoch, phase, val_meter, only_mix=False)
        # W&B 로깅
        # phase가 'test'일 때만 commit=True로 설정하여 train/val/test를 한 번에 커밋
        if hasattr(args, 'wandb_logger'):
            commit = (phase == 'test')  # test일 때만 commit=True
            wandb_step = epoch
            if hasattr(args, 'start_epoch') and hasattr(args, 'WANDB') and getattr(args.WANDB, 'resume_step', None) is not None:
                wandb_step = args.WANDB.resume_step + (epoch - args.start_epoch)
            write_to_wandb(args.wandb_logger, epoch, phase, val_meter, only_mix=False, commit=commit, step_override=wandb_step)
        # reset 전에 checkpoint metric 캡처 (train.py에서 best 체크포인트 판단에 사용)
        checkpoint_metric = _resolve_val_checkpoint_metric(args)
        checkpoint_dataset = getattr(args, 'val_checkpoint_dataset', None) or 'mixed_dataset'
        meters = val_meter.get(checkpoint_dataset, {})
        checkpoint_avg = meters[checkpoint_metric].avg if checkpoint_metric in meters else None
        for dataset_name, sub_metric_dict in val_meter.items():
            for metric_name in sub_metric_dict.keys():
                sub_metric_dict[metric_name].reset()
    return checkpoint_avg if phase == 'val' else None


def _dataset_kwargs(loss_submodule, dataset_name: str) -> dict:
    if getattr(loss_submodule, 'accepts_dataset_name', False):
        return {'dataset_name': dataset_name}
    return {}


def _resolve_val_checkpoint_metric(args) -> str:
    explicit = getattr(args, 'val_checkpoint_metric', None)
    if explicit:
        return str(explicit)
    rec_weight = float(getattr(args, 'rec_weight', 1.0))
    lve_weight = float(getattr(args, 'lve_weight', 0.0))
    if rec_weight <= 0.0 and lve_weight > 0.0:
        return 'LVE'
    return 'rec_L2_loss'


# ARKit 51 blendshape 이름 (3DETF/EmoTalk 알파벳순, tongueOut 제외)
# inhouse_blendshape_weight, 3DETF_blendshape_weight 모두 이 순서로 출력
_ARKIT_51_BS_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
]


def _get_blendshape_names(annot_type):
    """annot_type에 따라 blendshape 이름 목록 반환.
    3DETF(52-dim)와 inhouse(51-dim) 모두 ARKit 알파벳순."""
    if annot_type == '3DETF_blendshape_weight':
        emotalk_path = project_root / 'data' / 'shared_resources' / 'binary_resources' / 'EmoTalk.npz'
        if emotalk_path.exists():
            return np.load(str(emotalk_path), allow_pickle=True)['bs_names'].tolist()
        # EmoTalk.npz 없으면 51개 + tongueOut = 52개
        return list(_ARKIT_51_BS_NAMES) + ['tongueOut']
    if annot_type == 'inhouse_blendshape_weight':
        return list(_ARKIT_51_BS_NAMES)
    return None


def _save_one_blendshape_result(batch, out_motion, blendshape_dir, dataset_name, bs_names, args):
    """배치 1개에 대한 추론 결과를 CSV/WAV로 저장."""
    csv_header = ['frame', 'time_sec'] + bs_names
    blendshape_weights = out_motion[0].cpu().numpy()

    if 'audio_path' in batch:
        audio_path = batch['audio_path'][0] if isinstance(batch['audio_path'], list) else batch['audio_path']
        sample_name = os.path.basename(audio_path).replace('.wav', '').replace('.mp3', '')
    else:
        sample_name = 'sample_0'

    if 'original_audio' in batch:
        raw = batch['original_audio'][0]
        if isinstance(raw, torch.Tensor):
            raw = raw.cpu().numpy()
        audio_data = raw[0] if len(raw.shape) > 1 else raw
        audio_data = np.asarray(audio_data).squeeze().astype(np.float32)
        try:
            import soundfile as sf
            audio_save_path = os.path.join(blendshape_dir, f'{dataset_name}_{sample_name}.wav')
            sf.write(audio_save_path, audio_data, 16000)
        except Exception:
            pass

    csv_filename = f"{dataset_name}_{sample_name}.csv"
    csv_path = os.path.join(blendshape_dir, csv_filename)
    sample_rows = []
    for frame_idx in range(blendshape_weights.shape[0]):
        row = {'frame': frame_idx, 'time_sec': frame_idx / 30.0}
        for i, bs_name in enumerate(bs_names):
            row[bs_name] = blendshape_weights[frame_idx, i]
        sample_rows.append(row)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(sample_rows)


def save_blendshape_samples(test_loader, model, loss_module, epoch, args, subdir=None):
    """
    best.pt 저장 시(subdir='best'): test set에서 N개 샘플을 골라 추론 후 blendshape_samples/best/에 저장.
    N = args.save_blendshape_n_samples (기본 5). 길이 순 정렬 후 가장 긴 것부터 N개 선택.
    """
    if not hasattr(args, 'annot_type_list'):
        return
    has_3detf = '3DETF_blendshape_weight' in args.annot_type_list
    has_inhouse = 'inhouse_blendshape_weight' in args.annot_type_list
    if not has_3detf and not has_inhouse:
        return

    try:
        n_samples = getattr(args, 'save_blendshape_n_samples', 5)
        if hasattr(args, 'logger'):
            args.logger.info(f'=> save_blendshape_samples: epoch {epoch}, subdir={subdir}, n_samples={n_samples}')
        blendshape_dir = os.path.join(args.save_path, 'blendshape_samples', subdir if subdir else f'epoch_{epoch:03d}')
        os.makedirs(blendshape_dir, exist_ok=True)

        # test set에서 blendshape 타입 배치만 수집, 프레임 수 기준 내림차순
        candidates = []
        for batch in test_loader:
            annot_type = batch['annot_type'][0]
            if annot_type not in ('3DETF_blendshape_weight', 'inhouse_blendshape_weight'):
                continue
            frames = batch['data'].shape[1]
            candidates.append((frames, batch))
        candidates.sort(key=lambda x: -x[0])

        if not candidates:
            if hasattr(args, 'logger'):
                args.logger.warning('No blendshape sample found in test_loader')
            return

        # 길이 순으로 상위 N개 선택
        selected = [c[1] for c in candidates[:n_samples]]
        if hasattr(args, 'logger'):
            args.logger.info(f'=> Selected {len(selected)} test samples (max frames: {candidates[0][0]})')

        model.eval()
        with torch.no_grad():
            for batch in selected:
                annot_type = batch['annot_type'][0]
                dataset_name = batch['dataset_name'][0]
                bs_names = _get_blendshape_names(annot_type)
                if bs_names is None:
                    continue
                data = batch['data'].to(args.device, non_blocking=True)
                template = batch['template'].to(args.device, non_blocking=True)
                audio = batch['audio'].to(args.device, non_blocking=True)
                identity = batch['id'].to(args.device, non_blocking=True)
                viseme_labels_batch = batch.get('viseme_labels', None)
                if viseme_labels_batch is not None:
                    viseme_labels_batch = viseme_labels_batch.to(args.device, non_blocking=True)
                out_motion, _, _, _ = model(
                    audio, template, data, style_idx=identity, annot_type=annot_type,
                    viseme_labels=viseme_labels_batch)
                _save_one_blendshape_result(
                    batch, out_motion, blendshape_dir, dataset_name, bs_names, args)

        if hasattr(args, 'logger'):
            args.logger.info(f'=> Saved {len(selected)} blendshape results to {blendshape_dir}')
    except Exception as e:
        if hasattr(args, 'logger'):
            args.logger.warning(f'Error saving blendshape samples: {e}')
            import traceback
            args.logger.warning(traceback.format_exc())
        pass
