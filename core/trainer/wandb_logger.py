"""
Weights & Biases (W&B) 통합 로거
UniTalker(Vanilla D0–D7) + MetaHuman(D8/D9) 학습 메트릭·설정만 기록한다.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any
import torch

VANILLA_DATASET_KEYS = frozenset({f"D{i}" for i in range(8)})
METAHUMAN_DATASET_KEYS = frozenset({
    "D8", "D8_sanity", "D9", "D9_Nuridam",
    "D8_Metahuman_Control", "D9_Metahuman_Lipsync", "D8_Metahuman_LipsyncOnly",
})

# W&B에 기록할 loss/metric 키 (viseme auxiliary 등 미사용 항목 제외)
UNITALKER_WANDB_METRICS = frozenset({
    "rec_L2_loss",
    "pca_loss",
    "blendshape_loss",
    "mouth_loss",
    "lip_control_lve",
    "LVE",
    "LVD",
    "control_MSE",
})


def is_unitalker_wandb_metric(metric_name: str) -> bool:
    """viseme_loss 등 UniTalker/MetaHuman 학습과 무관한 메트릭은 W&B에 올리지 않음."""
    if metric_name in UNITALKER_WANDB_METRICS:
        return True
    if "viseme" in metric_name.lower():
        return False
    return False


def is_unitalker_wandb_dataset(dataset_name: str) -> bool:
    """W&B에 기록할 dataset 키: Vanilla(D0–D7), mixed, MetaHuman(D8/D9)만."""
    if dataset_name == "mixed_dataset":
        return True
    if dataset_name in VANILLA_DATASET_KEYS:
        return True
    if dataset_name in METAHUMAN_DATASET_KEYS:
        return True
    return False


def build_unitalker_wandb_config(config) -> Dict[str, Any]:
    """전체 args 대신 UniTalker + MetaHuman 관련 설정만 W&B config에 남긴다."""
    out: Dict[str, Any] = {"framework": "unitalker"}

    section_names = ("DATA", "LOSS", "NETWORK", "MODEL", "TRAIN", "TEST")
    for section in section_names:
        block = getattr(config, section, None)
        if block is None:
            continue
        if hasattr(block, "__dict__"):
            for k, v in block.__dict__.items():
                if k.startswith("_") or callable(v) or "viseme" in k.lower():
                    continue
                out[f"{section}.{k}"] = v
        elif isinstance(block, dict):
            for k, v in block.items():
                if "viseme" in str(k).lower():
                    continue
                out[f"{section}.{k}"] = v

    datasets = getattr(config, "dataset", None) or []
    if isinstance(datasets, str):
        datasets = datasets.split(",")
    out["DATA.dataset_list"] = list(datasets)
    out["DATA.has_metahuman"] = any(str(d) in METAHUMAN_DATASET_KEYS for d in datasets)
    out["DATA.has_vanilla"] = any(str(d) in VANILLA_DATASET_KEYS for d in datasets)

    metahuman_keys = (
        "metahuman_loss_exclude_head_pose",
        "metahuman_lip_vertex_mask",
        "metahuman_lip_control_use_all_lipsync",
        "metahuman_riglogic_basis_path",
        "use_metahuman_riglogic_rec",
        "metahuman_riglogic_mouth_vertex_count",
        "metahuman_head_init_npz",
        "style_emb_init_from_pretrained",
        "val_checkpoint_dataset",
        "lve_weight",
    )
    for key in metahuman_keys:
        if hasattr(config, key):
            out[f"metahuman.{key}"] = getattr(config, key)

    if hasattr(config, "WANDB"):
        wandb_block = config.WANDB
        if hasattr(wandb_block, "__dict__"):
            for k, v in wandb_block.__dict__.items():
                if k.startswith("_") or k == "enabled" or callable(v) or "viseme" in k.lower():
                    continue
                out[f"WANDB.{k}"] = v

    return out


def filter_unitalker_wandb_tags(tags: list | None) -> list[str]:
    """W&B tags: unitalker / MetaHuman / Vanilla 학습 관련만 유지."""
    if not tags:
        return ["unitalker"]
    keywords = (
        "unitalker",
        "metahuman",
        "vanilla",
        "lipsync",
        "d8",
        "d9",
        "riglogic",
        "mixed",
        "pretrained",
        "scratch",
        "ctrl",
        "lve",
        "d0",
        "d7",
    )
    kept: list[str] = []
    for tag in tags:
        tl = str(tag).lower()
        if any(k in tl for k in keywords):
            kept.append(str(tag))
    return kept or ["unitalker"]


# W&B 선택적 import (설치되지 않아도 에러 없음)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBLogger:
    """W&B 로거 클래스 - TensorBoard와 호환되는 인터페이스"""
    
    def __init__(self, 
                 project: str = "trainFramework",
                 entity: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 tags: Optional[list] = None,
                 group: Optional[str] = None,
                 job_type: str = "train",
                 resume: Optional[str] = None,
                 run_id: Optional[str] = None,
                 dir: Optional[str] = None,
                 reinit: bool = False,
                 mode: str = "online"):
        """
        Args:
            project: W&B 프로젝트 이름
            entity: W&B 엔티티 (팀/사용자 이름)
            name: 실행 이름 (None이면 자동 생성)
            config: 하이퍼파라미터 딕셔너리
            tags: 태그 리스트
            group: 실행 그룹 (여러 실행을 그룹화)
            job_type: 작업 타입 (train, eval, test 등)
            resume: 재개할 실행 ID ("allow", "must", "never", 또는 run_id)
            dir: 로그 저장 디렉토리
            reinit: 이미 초기화된 경우 재초기화 여부
            mode: "online", "offline", "disabled"
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.job_type = job_type
        self.resume = resume
        self._wandb_run_id = run_id  # 명시적 run ID 지정 (property와 충돌 방지)
        self.dir = dir
        self.reinit = reinit
        self.mode = mode
        
        self.wandb_run = None
        self._initialized = False
    
    def init(self):
        """W&B 초기화"""
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")
            self._initialized = False
            return
        
        if self._initialized and not self.reinit:
            return
        
        # Config를 평탄화 (중첩된 dict를 dot notation으로 변환)
        flat_config = self._flatten_config(self.config)
        
        # W&B 초기화
        # run_id가 지정되면 기존 run에 이어서 로깅
        init_kwargs = {
            'project': self.project,
            'entity': self.entity,
            'name': self.name,
            'config': flat_config,
            'tags': self.tags,
            'group': self.group,
            'job_type': self.job_type,
            'dir': self.dir,
            'reinit': self.reinit,
            'mode': self.mode
        }
        
        if self._wandb_run_id:
            # 명시적 run_id가 있으면 해당 run에 이어서 로깅
            init_kwargs['id'] = self._wandb_run_id
            init_kwargs['resume'] = 'must'  # 반드시 기존 run에 이어서 로깅
        else:
            # run_id가 없으면 resume 설정 사용
            init_kwargs['resume'] = self.resume
        
        self.wandb_run = wandb.init(**init_kwargs)
        
        self._initialized = True
        
        # 메트릭 정의 (과학적 표기법을 위한 설정)
        # W&B는 자동으로 작은 숫자에 대해 과학적 표기법을 사용하지만,
        # 명시적으로 정의하여 일관성 확보
        self._define_metrics()
    
    def log(self, 
            metrics: Dict[str, float],
            step: Optional[int] = None,
            commit: bool = True):
        """
        메트릭 로깅
        
        Args:
            metrics: {metric_name: value} 딕셔너리
            step: 스텝 번호 (None이면 자동 증가)
            commit: 즉시 커밋 여부
        """
        if not WANDB_AVAILABLE or not self._initialized:
            return
        
        # step을 metrics에 추가하지 않고, log의 step 파라미터만 사용
        # global_step은 wandb.define_metric에서 step_metric으로 사용됨
        self.wandb_run.log(metrics, step=step, commit=commit)
    
    def log_epoch(self,
                  epoch: int,
                  phase: str,
                  metrics: Dict[str, float],
                  dataset_name: Optional[str] = None):
        """
        Epoch별 메트릭 로깅 (TensorBoard 호환)
        
        Args:
            epoch: 에포크 번호
            phase: 'train', 'val', 'test'
            metrics: 메트릭 딕셔너리
            dataset_name: 데이터셋 이름 (선택적)
        """
        if not self._initialized:
            self.init()
        
        # 메트릭 이름에 phase와 dataset_name 추가
        logged_metrics = {}
        for key, value in metrics.items():
            if dataset_name:
                metric_name = f"{phase}/{dataset_name}/{key}"
            else:
                metric_name = f"{phase}/{key}"
            logged_metrics[metric_name] = value
        
        # Epoch도 로깅
        logged_metrics[f"{phase}/epoch"] = epoch
        
        self.log(logged_metrics, step=epoch)
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """
        모델 가중치/그래디언트 추적
        
        Args:
            model: PyTorch 모델
            log: "gradients", "parameters", "all"
            log_freq: 로깅 빈도
        """
        if not self._initialized:
            self.init()
        
        self.wandb_run.watch(model, log=log, log_freq=log_freq)
    
    def log_model(self, model_path: str, name: Optional[str] = None):
        """
        모델 체크포인트를 W&B Artifact로 저장
        
        Args:
            model_path: 모델 파일 경로
            name: Artifact 이름
        """
        if not self._initialized:
            self.init()
        
        artifact = wandb.Artifact(
            name=name or "model",
            type="model"
        )
        artifact.add_file(model_path)
        self.wandb_run.log_artifact(artifact)
    
    def log_config(self, config: Dict):
        """Config 업데이트"""
        if not self._initialized:
            self.init()
        
        flat_config = self._flatten_config(config)
        self.wandb_run.config.update(flat_config)
    
    def finish(self):
        """W&B 실행 종료"""
        if self._initialized and self.wandb_run:
            wandb.finish()
            self._initialized = False
    
    def _flatten_config(self, config: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """중첩된 config를 평탄화"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _define_metrics(self):
        """메트릭 정의 (과학적 표기법 표시를 위한 설정)"""
        if not WANDB_AVAILABLE or not self._initialized or self.wandb_run is None:
            return
        
        # 일반적인 loss 메트릭들 정의
        loss_metrics = [
            'train/rec_L2_loss',
            'train/pca_loss',
            'train/blendshape_loss',
            'train/mouth_loss',
            'train/lip_control_lve',
            'train/epoch',
            'val/rec_L2_loss',
            'val/LVE',
            'val/LVD',
            'val/lip_control_lve',
            'val/epoch',
            'test/rec_L2_loss',
            'test/LVE',
            'test/LVD',
            'test/lip_control_lve',
            'test/epoch',
        ]
        
        for metric_name in loss_metrics:
            try:
                # step_metric을 _step으로 설정 (W&B 기본 step metric)
                # summary statistics를 설정하여 자동으로 과학적 표기법 사용 유도
                wandb.define_metric(
                    metric_name,
                    step_metric='_step',
                    summary='min'  # 최소값 요약 (과학적 표기법 자동 적용)
                )
            except Exception:
                # define_metric 실패해도 계속 진행 (호환성 문제 가능)
                pass
    
    # TensorBoard 호환 메서드들
    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """TensorBoard add_scalar 호환"""
        self.log({tag: scalar_value}, step=global_step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], 
                    global_step: Optional[int] = None):
        """TensorBoard add_scalars 호환"""
        metrics = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self.log(metrics, step=global_step)
    
    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None):
        """TensorBoard add_text 호환"""
        if not self._initialized:
            self.init()
        self.wandb_run.log({tag: wandb.Html(text_string)}, step=global_step)
    
    @property
    def run_id(self) -> Optional[str]:
        """현재 실행 ID 반환"""
        if self._initialized and self.wandb_run:
            return self.wandb_run.id
        return None
    
    @property
    def run_name(self) -> Optional[str]:
        """현재 실행 이름 반환"""
        if self._initialized and self.wandb_run:
            return self.wandb_run.name
        return None

    @property
    def run_url(self) -> Optional[str]:
        """W&B 대시보드 URL (프로젝트/entity 포함)."""
        if self._initialized and self.wandb_run:
            return getattr(self.wandb_run, "url", None) or getattr(self.wandb_run, "get_url", lambda: None)()
        return None


def create_wandb_logger(config) -> Optional[WandBLogger]:
    """
    Config에서 W&B 로거 생성
    
    Args:
        config: 설정 객체 (WANDB 속성 포함 또는 평탄화된 config)
    
    Returns:
        WandBLogger 인스턴스 또는 None (비활성화 시)
    """
    # W&B 설정 확인 (두 가지 방식 지원)
    # 1. WANDB 섹션이 있는 경우 (중첩된 구조)
    if hasattr(config, 'WANDB'):
        wandb_config = config.WANDB
        enabled = getattr(wandb_config, 'enabled', False)
        project = getattr(wandb_config, 'project', 'trainFramework')
        entity = getattr(wandb_config, 'entity', None)
        name = getattr(wandb_config, 'name', None)
        tags = getattr(wandb_config, 'tags', [])
        group = getattr(wandb_config, 'group', None)
        job_type = getattr(wandb_config, 'job_type', 'train')
        resume = getattr(wandb_config, 'resume', None)
        dir_path = getattr(wandb_config, 'dir', None)
        reinit = getattr(wandb_config, 'reinit', False)
        mode = getattr(wandb_config, 'mode', 'online')
        watch_model = getattr(wandb_config, 'watch_model', False)
        watch_log = getattr(wandb_config, 'watch_log', 'gradients')
        watch_log_freq = getattr(wandb_config, 'watch_log_freq', 100)
        log_model = getattr(wandb_config, 'log_model', False)
    # 2. 평탄화된 config에서 W&B 관련 키를 직접 찾기
    elif hasattr(config, 'wandb_enabled') or (hasattr(config, '__dict__') and 'wandb_enabled' in config.__dict__):
        # 평탄화된 config에서 직접 읽기
        enabled = getattr(config, 'wandb_enabled', False) or getattr(config, 'WANDB.enabled', False)
        if not enabled:
            return None
        project = getattr(config, 'wandb_project', getattr(config, 'WANDB.project', 'trainFramework'))
        entity = getattr(config, 'wandb_entity', getattr(config, 'WANDB.entity', None))
        name = getattr(config, 'wandb_name', getattr(config, 'WANDB.name', None))
        tags = getattr(config, 'wandb_tags', getattr(config, 'WANDB.tags', []))
        group = getattr(config, 'wandb_group', getattr(config, 'WANDB.group', None))
        job_type = getattr(config, 'wandb_job_type', getattr(config, 'WANDB.job_type', 'train'))
        resume = getattr(config, 'wandb_resume', getattr(config, 'WANDB.resume', None))
        dir_path = getattr(config, 'wandb_dir', getattr(config, 'WANDB.dir', None))
        reinit = getattr(config, 'wandb_reinit', getattr(config, 'WANDB.reinit', False))
        mode = getattr(config, 'wandb_mode', getattr(config, 'WANDB.mode', 'online'))
        watch_model = getattr(config, 'wandb_watch_model', getattr(config, 'WANDB.watch_model', False))
        watch_log = getattr(config, 'wandb_watch_log', getattr(config, 'WANDB.watch_log', 'gradients'))
        watch_log_freq = getattr(config, 'wandb_watch_log_freq', getattr(config, 'WANDB.watch_log_freq', 100))
        log_model = getattr(config, 'wandb_log_model', getattr(config, 'WANDB.log_model', False))
    else:
        # W&B 설정이 없으면 비활성화
        return None
    
    # W&B 활성화 여부 확인
    if not enabled:
        return None
    
    config_dict = build_unitalker_wandb_config(config)

    run_name = getattr(wandb_config, 'name', None)
    if run_name is None and hasattr(config, 'TRAIN'):
        if hasattr(config.TRAIN, 'run_name'):
            run_name = config.TRAIN.run_name
        elif hasattr(config.TRAIN, 'save_path'):
            # save_path에서 실행 이름 추출
            save_path = config.TRAIN.save_path
            run_name = Path(save_path).name if save_path else None
    
    # W&B 로거 생성
    logger = WandBLogger(
        project=getattr(wandb_config, 'project', 'unitalker'),
        entity=getattr(wandb_config, 'entity', None),
        name=run_name,
        config=config_dict,
        tags=filter_unitalker_wandb_tags(getattr(wandb_config, 'tags', [])),
        group=getattr(wandb_config, 'group', None),
        job_type=getattr(wandb_config, 'job_type', 'train'),
        resume=getattr(wandb_config, 'resume', None),
        run_id=getattr(wandb_config, 'run_id', None),  # 명시적 run_id 지원
        dir=getattr(wandb_config, 'dir', None),
        reinit=getattr(wandb_config, 'reinit', False),
        mode=getattr(wandb_config, 'mode', 'online')
    )
    
    logger.init()
    return logger
