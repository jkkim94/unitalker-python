"""
Weights & Biases (W&B) 통합 로거
모든 모델에서 공통으로 사용할 수 있는 W&B 로깅 유틸리티
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any
import torch

# W&B 선택적 import (설치되지 않아도 에러 없음)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBLogger:
    """W&B 로거 클래스 - TensorBoard와 함께 사용 가능"""
    
    def __init__(self, 
                 project: str,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 entity: Optional[str] = None,
                 tags: Optional[list] = None,
                 enabled: bool = True):
        """
        Args:
            project: W&B 프로젝트 이름
            name: 실행 이름 (None이면 자동 생성)
            config: 하이퍼파라미터 딕셔너리
            entity: W&B 엔티티 (팀/사용자 이름)
            tags: 태그 리스트
            enabled: W&B 활성화 여부
        """
        self.enabled = enabled and os.getenv('WANDB_DISABLED', 'false').lower() != 'true' and WANDB_AVAILABLE
        
        if not self.enabled:
            self.run = None
            return
        
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")
            self.run = None
            return
        
        # W&B 초기화
        wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            tags=tags,
            reinit=True  # 이미 초기화된 경우 재초기화
        )
        self.run = wandb.run
    
    def log(self, 
            metrics: Dict[str, float], 
            step: Optional[int] = None,
            commit: bool = True):
        """메트릭 로깅"""
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_model_summary(self, model: torch.nn.Module):
        """모델 구조 로깅"""
        if self.enabled and self.run is not None:
            # 모델 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.config.update({
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # float32 기준
            })
    
    def watch(self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100):
        """모델 가중치/그래디언트 추적"""
        if self.enabled and self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def log_artifact(self, 
                     artifact_path: str,
                     artifact_type: str = "model",
                     name: Optional[str] = None):
        """아티팩트 로깅 (모델 체크포인트 등)"""
        if self.enabled and self.run is not None:
            artifact = wandb.Artifact(
                name=name or Path(artifact_path).stem,
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
    
    def log_checkpoint(self, checkpoint_path: str, epoch: int):
        """체크포인트를 W&B에 업로드"""
        if self.enabled and self.run is not None:
            artifact = wandb.Artifact(
                name=f"checkpoint-epoch-{epoch}",
                type="model"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """W&B 실행 종료"""
        if self.enabled and self.run is not None:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_wandb_logger(config: Any) -> Optional[WandBLogger]:
    """
    Config에서 W&B 로거 생성
    
    Args:
        config: 학습 config 객체 (args)
    
    Returns:
        WandBLogger 인스턴스 또는 None (비활성화 시)
    """
    # W&B 설정 확인
    wandb_config = getattr(config, 'WANDB', None)
    
    if wandb_config is None:
        # WANDB 섹션이 없으면 비활성화
        return None
    
    # enabled 확인 (dict 또는 객체 모두 지원)
    if isinstance(wandb_config, dict):
        enabled = wandb_config.get('enabled', False)
    else:
        enabled = getattr(wandb_config, 'enabled', False)
    
    if not enabled:
        return None
    
    # Config를 딕셔너리로 변환 (W&B에 로깅할 하이퍼파라미터)
    config_dict = {}
    if hasattr(config, '__dict__'):
        # WANDB 섹션은 제외하고 로깅
        config_dict = {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('_') and not callable(v) and k != 'WANDB'}
    
    # W&B 설정 추출
    if isinstance(wandb_config, dict):
        project = wandb_config.get('project', 'trainFramework')
        run_name = wandb_config.get('run_name') or getattr(config, 'run_name', None)
        entity = wandb_config.get('entity')
        tags = wandb_config.get('tags', [])
    else:
        project = getattr(wandb_config, 'project', 'trainFramework')
        run_name = getattr(wandb_config, 'run_name', None) or getattr(config, 'run_name', None)
        entity = getattr(wandb_config, 'entity', None)
        tags = getattr(wandb_config, 'tags', [])
    
    # W&B 초기화
    logger = WandBLogger(
        project=project,
        name=run_name,
        config=config_dict,
        entity=entity,
        tags=tags,
        enabled=True
    )
    
    # wandb_watch, wandb_upload_checkpoints를 config에 추가 (나중에 사용)
    if isinstance(wandb_config, dict):
        config.wandb_watch = wandb_config.get('wandb_watch', False)
        config.wandb_upload_checkpoints = wandb_config.get('wandb_upload_checkpoints', False)
    else:
        config.wandb_watch = getattr(wandb_config, 'wandb_watch', False)
        config.wandb_upload_checkpoints = getattr(wandb_config, 'wandb_upload_checkpoints', False)
    
    return logger


def log_to_wandb(wandb_logger: Optional[WandBLogger],
                 metrics: Dict[str, float],
                 step: Optional[int] = None,
                 prefix: str = ""):
    """
    W&B에 메트릭 로깅 (편의 함수)
    
    Args:
        wandb_logger: WandBLogger 인스턴스
        metrics: 로깅할 메트릭 딕셔너리
        step: 스텝 번호
        prefix: 메트릭 이름 앞에 붙일 접두사
    """
    if wandb_logger is None:
        return
    
    # 접두사 추가
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    wandb_logger.log(metrics, step=step)
