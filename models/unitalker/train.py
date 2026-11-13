#!/usr/bin/env python
# yapf: disable
import os
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unitalker.dataset.dataset import get_dataloaders
from models.unitalker.loss.loss import UniTalkerLoss
from models.unitalker.model.unitalker import UniTalker
from core.utils.utils import (
    get_average_meter_dict, get_logger, get_parser, get_audio_encoder_dim,
    load_ckpt, seed_everything, filter_unitalker_state_dict
)
from core.trainer.wandb_logger import create_wandb_logger
from models.unitalker.train_eval_loop import train_epoch, validate_epoch, save_blendshape_samples

# yapf: enable


@torch.no_grad()
def init_from_pretrained_extras(model, args, train_dataset, logger):
    """체크포인트 로드 후 추가 초기화.

    style_emb_init_from_pretrained: dataset json info.pretrained_style_idx가 있는 identity는
        pretrained style embedding row를 그대로 복사 (예: D10 = D3/D4/D6 화자).
    metahuman_head_init_npz: metahuman_control head가 체크포인트에 없으면
        W_mh = A^T · W_3DETF, b_mh = A^T · b_3DETF + c 로 초기화 (A: ARKit52→MH 매핑, npz).
    """
    use_style = bool(getattr(args, 'style_emb_init_from_pretrained', False))
    head_npz = getattr(args, 'metahuman_head_init_npz', None)
    if not use_style and not head_npz:
        return
    state = torch.load(args.weight_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    if use_style:
        emb_key = 'decoder.learnable_style_emb.weight'
        src_emb = state.get(emb_key)
        emb = model.decoder.learnable_style_emb.weight
        copied = 0
        for new_idx, src_idx in enumerate(getattr(train_dataset, 'pretrained_style_idx', [])):
            if src_emb is not None and src_idx is not None and 0 <= src_idx < len(src_emb):
                emb[new_idx].copy_(src_emb[src_idx].to(emb.device))
                copied += 1
        logger.info(f'=> style embedding rows copied from pretrained: {copied}/{len(emb)}')

    if head_npz:
        import numpy as np
        head_key = 'out_head_dict.metahuman_control'
        if f'{head_key}.weight' in state:
            logger.info('=> metahuman head already in checkpoint; skip ARKit head init')
        elif ('metahuman_control' not in model.out_head_dict
              or not isinstance(model.out_head_dict['metahuman_control'], torch.nn.Linear)):
            logger.warning('=> metahuman_head_init_npz requires a single Linear metahuman_control head')
        else:
            npz_path = Path(head_npz)
            if not npz_path.is_absolute():
                npz_path = project_root / npz_path
            init = np.load(npz_path)
            mapping = torch.from_numpy(init['head_init_weight'].astype('float32'))  # (52, 260)
            bias0 = torch.from_numpy(init['head_init_bias'].astype('float32'))  # (260,)
            w_arkit = state['out_head_dict.3DETF_blendshape_weight.weight'].float()  # (52, D)
            b_arkit = state['out_head_dict.3DETF_blendshape_weight.bias'].float()  # (52,)
            head = model.out_head_dict['metahuman_control']
            head.weight.copy_((mapping.T @ w_arkit).to(head.weight.device))
            head.bias.copy_((mapping.T @ b_arkit + bias0).to(head.bias.device))
            logger.info(f'=> metahuman head initialized from 3DETF head via {npz_path}')


def main():
    seed_everything(42)
    args = get_parser()  # configs/*.yaml → CfgNode (core/utils/config.py)
    args.audio_encoder_feature_dim = get_audio_encoder_dim(args.audio_encoder_repo)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    args.identity_num = train_loader.dataset.get_identity_num()
    args.annot_type_list = train_loader.dataset.annot_type_list
    if args.random_id_prob > 0.0:
        args.identity_num = args.identity_num + 1

    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, 'train.log')
    logger = get_logger(log_file)
    writer = SummaryWriter(args.save_path)
    
    # W&B 초기화 (공통 모듈 사용)
    wandb_logger = create_wandb_logger(args)
    if wandb_logger:
        logger.info(f'=> W&B initialized: {wandb_logger.run_name} (ID: {wandb_logger.run_id})')
        if wandb_logger.run_url:
            logger.info(f'=> W&B dashboard: {wandb_logger.run_url}')
        logger.info(f'=> W&B initialized status: {wandb_logger._initialized}')
        if hasattr(wandb_logger, 'wandb_run'):
            logger.info(f'=> W&B run object exists: {wandb_logger.wandb_run is not None}')
    else:
        logger.warning('=> W&B logger is None - metrics will not be logged to W&B')
    
    logger.info('=> creating model ...')
    model = UniTalker(args)
    try:
        device = torch.device('cuda')
        _ = torch.zeros(1).to(device)
    except RuntimeError as e:
        logger.warning(f'CUDA unavailable ({e}), using CPU')
        device = torch.device('cpu')
    args.device = device
    model = model.to(device)
    
    # Optimizer 생성 (체크포인트 로드 전에 생성)
    optimizer_weight_decay = float(getattr(args, 'weight_decay', 0.0))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=optimizer_weight_decay)
    
    # 체크포인트 로드 (모델 + optimizer)
    loaded_checkpoint = None
    if args.weight_path is not None and os.path.exists(args.weight_path):
        logger.info(f'=> loading checkpoint {args.weight_path} ...')
        loaded_checkpoint = torch.load(args.weight_path, map_location='cpu')
        
        # 새로운 체크포인트 형식 (epoch, optimizer 포함)
        if isinstance(loaded_checkpoint, dict) and 'model_state_dict' in loaded_checkpoint:
            # 모델 로드
            id_embedding_key = 'decoder.learnable_style_emb.weight'
            model_state = loaded_checkpoint['model_state_dict']
            if id_embedding_key in model_state:
                if len(model_state[id_embedding_key]) != len(model.state_dict()[id_embedding_key]):
                    target_id_num = len(model.state_dict()[id_embedding_key])
                    model_state[id_embedding_key] = model_state[id_embedding_key][-1][
                        None].repeat(target_id_num, 1)
            if args.re_init_decoder_and_head:
                ckpt_keys = list(model_state.keys())
                encoder_keys = [k for k in ckpt_keys if 'audio_encoder' in k]
                reinit_keys = list(set(ckpt_keys) - set(encoder_keys))
                for k in reinit_keys:
                    if k in model.state_dict():
                        model_state[k] = model.state_dict()[k]
            model.load_state_dict(model_state, strict=False)
            
            # Optimizer state 로드 (있는 경우)
            if 'optimizer_state_dict' in loaded_checkpoint:
                try:
                    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                    logger.info('=> Optimizer state loaded')
                except Exception as e:
                    logger.warning(f'=> Could not load optimizer state: {e}')
        else:
            # 기존 형식 (모델만)
            load_ckpt(model, args.weight_path, args.re_init_decoder_and_head)
        init_from_pretrained_extras(model, args, train_loader.dataset, logger)
    
    logger.info(args)
    model.summary(logger)

    loss_module = UniTalkerLoss(args).to(device)  # annot_type별 loss (metahuman_control → RigLogic rec+bs)
    if getattr(loss_module, 'dataset_mask_info', None):
        logger.info(f'=> per-dataset loss channel masks (active channels): {loss_module.dataset_mask_info}')
    if getattr(loss_module, 'metahuman_lve_info', None):
        info = loss_module.metahuman_lve_info
        logger.info(f'=> MetaHuman LVE lip mask: {info}')
        if wandb_logger:
            flat = {
                'metahuman.lip_vertex_count': info.get('lip_vertex_count'),
                'metahuman.lip_vertex_mask_path': info.get('lip_vertex_mask_path'),
                'metahuman.lip_control_channel_count': info.get('lip_control_channel_count'),
                'metahuman.lve_weight': float(getattr(args, 'lve_weight', 0.0)),
            }
            meta = info.get('lip_vertex_mask_meta') or {}
            for k, v in meta.items():
                flat[f'metahuman.lip_mask.{k}'] = v
            wandb_logger.log_config(flat)

    args.train_meter, args.train_metric_name_list = \
        get_average_meter_dict(args, 'train')
    args.val_meter, args.val_metric_name_list = \
        get_average_meter_dict(args, 'val')

    if (
        float(getattr(args, 'lve_weight', 0.0)) > 0
        and (
            getattr(args, 'metahuman_lip_vertex_count', None)
            or getattr(args, 'metahuman_riglogic_mouth_vertex_count', None)
        )
    ):
        from core.utils.utils import AverageMeter
        for dataset_name in args.train_meter:
            args.train_meter[dataset_name]['lip_control_lve'] = AverageMeter()
        for dataset_name in args.val_meter:
            args.val_meter[dataset_name]['lip_control_lve'] = AverageMeter()
    args.logger = logger
    args.writer = writer
    args.wandb_logger = wandb_logger  # W&B 로거 추가
    args.save_sample_results = True  # 학습 중간 샘플 결과 저장 활성화
    
    # 모델 추적 (모델 생성 후)
    if wandb_logger and hasattr(args, 'WANDB') and getattr(args.WANDB, 'watch_model', False):
        wandb_logger.watch(model, 
                          log=getattr(args.WANDB, 'watch_log', 'gradients'),
                          log_freq=getattr(args.WANDB, 'watch_log_freq', 100))

    # Resume from checkpoint - epoch 정보 확인
    start_epoch = 0
    if args.weight_path is not None and os.path.exists(args.weight_path):
        # 이미 로드한 체크포인트에서 epoch 정보 추출 (재로드 방지)
        if loaded_checkpoint is not None:
            # 새로운 형식의 체크포인트에서 epoch 정보 추출
            if isinstance(loaded_checkpoint, dict) and 'epoch' in loaded_checkpoint:
                start_epoch = loaded_checkpoint['epoch']
                logger.info(f'=> Resuming from epoch {start_epoch} (from checkpoint)')
            else:
                # 기존 형식: 파일명에서 epoch 추출 (예: 086.pt -> 86)
                checkpoint_name = os.path.basename(args.weight_path)
                if checkpoint_name.endswith('.pt'):
                    try:
                        epoch_str = checkpoint_name.replace('.pt', '')
                        start_epoch = int(epoch_str)
                        logger.info(f'=> Resuming from epoch {start_epoch} (from filename)')
                    except ValueError:
                        logger.warning(f'=> Could not extract epoch from {checkpoint_name}, starting from 0')
        else:
            # 체크포인트가 로드되지 않은 경우 파일명에서 추출
            checkpoint_name = os.path.basename(args.weight_path)
            if checkpoint_name.endswith('.pt'):
                try:
                    epoch_str = checkpoint_name.replace('.pt', '')
                    start_epoch = int(epoch_str)
                    logger.info(f'=> Resuming from epoch {start_epoch} (from filename)')
                except ValueError:
                    logger.warning(f'=> Could not extract epoch from {checkpoint_name}, starting from 0')
    
    # Best validation loss 추적 (가장 낮은 loss만 체크포인트 저장)
    best_val_loss = float('inf')
    best_checkpoint_path = None
    args.start_epoch = start_epoch  # WandB resume_step 계산용
    scheduler = None
    if bool(getattr(args, 'use_val_plateau_scheduler', False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(getattr(args, 'val_scheduler_factor', 0.5)),
            patience=int(getattr(args, 'val_scheduler_patience', 8)),
            threshold=float(getattr(args, 'val_scheduler_threshold', 1e-6)),
            threshold_mode='rel',
            cooldown=int(getattr(args, 'val_scheduler_cooldown', 0)),
            min_lr=float(getattr(args, 'val_scheduler_min_lr', 1e-6)),
        )
        logger.info(
            '=> Val ReduceLROnPlateau enabled '
            f"(factor={getattr(args, 'val_scheduler_factor', 0.5)}, "
            f"patience={getattr(args, 'val_scheduler_patience', 8)}, "
            f"min_lr={getattr(args, 'val_scheduler_min_lr', 1e-6)})"
        )
    train_plateau_enabled = bool(getattr(args, 'early_stop_on_train_plateau', False))
    train_plateau_patience = int(getattr(args, 'train_plateau_patience', 30))
    train_plateau_min_delta = float(getattr(args, 'train_plateau_min_delta', 1e-6))
    best_train_loss = float('inf')
    train_plateau_count = 0
    if train_plateau_enabled:
        logger.info(
            f'=> Train-loss plateau early stop enabled '
            f'(patience={train_plateau_patience}, min_delta={train_plateau_min_delta})'
        )
    val_plateau_enabled = bool(getattr(args, 'early_stop_on_val_plateau', False))
    val_plateau_patience = int(getattr(args, 'val_plateau_patience', 25))
    val_plateau_min_delta = float(getattr(args, 'val_plateau_min_delta', 1e-6))
    min_epochs_before_stop = int(getattr(args, 'min_epochs_before_stop', 0))
    best_val_for_plateau = float('inf')
    val_plateau_count = 0
    if val_plateau_enabled:
        logger.info(
            f'=> Val-loss plateau early stop enabled '
            f'(patience={val_plateau_patience}, min_delta={val_plateau_min_delta}, '
            f'min_epochs_before_stop={min_epochs_before_stop})'
        )
    
    for epoch in range(start_epoch, args.epochs):
        # staged fine-tuning: epoch0 encoder freeze → decoder/head만 학습, epoch1+ encoder unfreeze
        if args.fix_encoder_first:
            if epoch == 0:
                model.audio_encoder._freeze_wav2vec2_parameters(True)  # requires_grad=False
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=optimizer_weight_decay)
            elif epoch == 1:
                model.audio_encoder._freeze_wav2vec2_parameters(
                    args.freeze_wav2vec)
                model.audio_encoder.feature_extractor._freeze_parameters()
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr,
                    weight_decay=optimizer_weight_decay)
            if scheduler is not None:
                scheduler.optimizer = optimizer
        epoch_plus = epoch + 1
        current_train_loss = train_epoch(train_loader, model, loss_module, optimizer, epoch_plus,
                                         args)
        should_stop_on_train_plateau = False
        should_stop_on_val_plateau = False
        if train_plateau_enabled and current_train_loss is not None:
            if current_train_loss < (best_train_loss - train_plateau_min_delta):
                best_train_loss = current_train_loss
                train_plateau_count = 0
            else:
                train_plateau_count += 1
                logger.info(
                    f'=> Train loss plateau count {train_plateau_count}/{train_plateau_patience} '
                    f'(current={current_train_loss:.8f}, best={best_train_loss:.8f})'
                )
                if train_plateau_count >= train_plateau_patience:
                    should_stop_on_train_plateau = True

        if epoch_plus % args.eval_every == 0:
            current_val_loss = validate_epoch(val_loader, model, loss_module, epoch_plus, 'val',
                           args)
            if scheduler is not None and current_val_loss is not None:
                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(current_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != prev_lr:
                    logger.info(f'=> LR reduced by val plateau: {prev_lr:.8f} -> {new_lr:.8f}')
                else:
                    logger.info(f'=> LR unchanged: {new_lr:.8f}')
            
            # Validation loss (validate_epoch에서 reset 전에 캡처한 값 반환)
            
            # Best validation loss보다 낮으면 체크포인트 저장
            if current_val_loss is not None and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                
                # 이전 best 체크포인트 삭제
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    try:
                        os.remove(best_checkpoint_path)
                        logger.info(f'=> Removed previous best checkpoint: {best_checkpoint_path}')
                    except Exception as e:
                        logger.warning(f'=> Failed to remove previous checkpoint: {e}')
                
                # 새로운 best 체크포인트 저장 (서버 추론 시 inference_config 사용, 하드코딩 없음)
                best_checkpoint_path = os.path.join(args.save_path, 'best.pt')
                inference_dataset = getattr(args, 'inference_dataset', None)
                if not inference_dataset:
                    demo_ds = getattr(args, 'demo_dataset', None) or []
                    train_ds = getattr(args, 'dataset', None) or []
                    if len(demo_ds) == 1:
                        inference_dataset = demo_ds[0]
                    elif len(demo_ds) > 1:
                        inference_dataset = demo_ds[-1]
                    elif len(train_ds) == 1:
                        inference_dataset = train_ds[0]
                    elif len(train_ds) > 1:
                        inference_dataset = train_ds[-1]
                from models.unitalker.dataset.dataset_config import dataset_config as _ds_cfg
                _inf_annot = _ds_cfg.get(str(inference_dataset), {}).get("annot_type")
                if _inf_annot == "metahuman_control" or inference_dataset in (
                    "D8", "D8_sanity", "D9", "D9_Nuridam",
                ):
                    inference_output_kind = "metahuman_260"
                else:
                    inference_output_kind = "arkit_51"
                checkpoint = {
                    'epoch': epoch_plus,
                    'val_loss': best_val_loss,
                    'model_state_dict': filter_unitalker_state_dict(model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'inference_config': {
                        'dataset_for_blendshape': inference_dataset,
                        'template_idx': 0,
                        'output_kind': inference_output_kind,
                    },
                }
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f'=> Saved best checkpoint (epoch {epoch_plus}, val_loss: {best_val_loss:.8f}) to {best_checkpoint_path}')
                # best.pt 갱신 시에만 blendshape_samples/best 로 추론 결과 저장
                if test_loader is not None:
                    save_blendshape_samples(test_loader, model, loss_module, epoch_plus, args, subdir='best')

                # W&B에 모델 저장 (선택적)
                if wandb_logger and hasattr(args, 'WANDB') and getattr(args.WANDB, 'log_model', False):
                    wandb_logger.log_model(best_checkpoint_path, name='best_checkpoint')
            if val_plateau_enabled and current_val_loss is not None:
                if current_val_loss < (best_val_for_plateau - val_plateau_min_delta):
                    best_val_for_plateau = current_val_loss
                    val_plateau_count = 0
                else:
                    val_plateau_count += 1
                    logger.info(
                        f'=> Val loss plateau count {val_plateau_count}/{val_plateau_patience} '
                        f'(current={current_val_loss:.8f}, best={best_val_for_plateau:.8f})'
                    )
                    if epoch_plus >= min_epochs_before_stop and val_plateau_count >= val_plateau_patience:
                        should_stop_on_val_plateau = True

        if epoch_plus % args.test_every == 0 and test_loader is not None:
            validate_epoch(test_loader, model, loss_module, epoch_plus, 'test',
                           args)
            
            ## 에폭별 체크포인트 저장 (run_d8_epoch_inference.py에서 사용)
            #epoch_ckpt_path = os.path.join(args.save_path, f'epoch_{epoch_plus:03d}.pt')
            #torch.save({
            #    'epoch': epoch_plus,
            #    'val_loss': current_val_loss,
            #    'model_state_dict': filter_unitalker_state_dict(model.state_dict()),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #}, epoch_ckpt_path)
            #if hasattr(args, 'logger'):
            #    args.logger.info(f'=> Saved epoch checkpoint to {epoch_ckpt_path}')
        
        if should_stop_on_train_plateau:
            logger.info(
                f'=> Early stopping at epoch {epoch_plus}: '
                f'train rec_L2_loss did not improve for {train_plateau_patience} epochs'
            )
            break
        if should_stop_on_val_plateau:
            logger.info(
                f'=> Early stopping at epoch {epoch_plus}: '
                f'val rec_L2_loss did not improve for {val_plateau_patience} evals'
            )
            break
    
    # W&B 종료
    if wandb_logger:
        wandb_logger.finish()
        logger.info('=> W&B finished')


if __name__ == '__main__':
    main()
