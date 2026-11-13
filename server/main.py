"""
FastAPI 서버: 에폭 학습 결과 로드 API
다른 PC의 Next.js 프론트엔드와 통신하기 위한 백엔드 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
import base64
import tempfile
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import httpx
import time
import numpy as np
import csv
import io
import re
import glob
import asyncio
import hashlib

def _init_cuda():
    """서비스 시작 시 CUDA 초기화 — 이전 프로세스의 오염된 GPU 상태 방지."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.init()
            print(f"[startup] CUDA 초기화 완료: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[startup] CUDA 초기화 실패 (CPU 전용 모드): {e}")

_init_cuda()

app = FastAPI(
    title="Training Results API",
    version="1.0.0",
    description="""
    학습 결과 및 Blendshape 추론을 제공하는 백엔드 API 서버입니다.
    설계: `docs/SERVICE_MODEL_STRUCTURE.md` 참고.
    
    ## Front User 선택 항목 (최대한 선택 가능)
    - **학습 모델(run)**: 사용할 학습 실험 선택
    - **체크포인트**: 해당 모델 내 체크포인트 (예: best.pt)
    - **오디오**: 추론할 오디오 입력
    - **화자(identity)**: 출력에 적용할 스타일(학습된 화자 중 선택). 입력 목소리는 학습 참여 여부 무관.
    - **출력**: Blendshape로 고정 (FLAME/PCA는 서비스 대상 아님)
    
    ## 주요 API
    - GET /api/runs: 학습 모델(run) 목록 (output_kind, channels, engines, dnaId, hasCheckpoint)
    - GET /api/checkpoints: run별 체크포인트 목록
    - GET /api/speakers: run별 화자(identity) 수
    - POST /api/inference: 오디오 → Blendshape (run, checkpoint, speaker, engine=pytorch|onnx)
    - FaceTool 계약: docs/FACETOOL_SERVICE_CONTRACT.md
    - GET /api/epochs, POST /api/epoch, GET /api/gt: 레거시 뷰어용 (FaceTool GT 평가 비대상)
    
    - Swagger UI: `/docs`, ReDoc: `/redoc`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 설정 - Next.js 서버 사이드에서 호출하는 경우 필요 없음
# 브라우저에서 직접 호출하는 경우에만 필요
# 환경 변수 ENABLE_CORS=true로 활성화 가능
if os.getenv("ENABLE_CORS", "false").lower() == "true":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한 권장
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def _log_inference_mode():
    use_local = os.getenv("USE_LOCAL_INFERENCE", "").strip().lower() in ("1", "true", "yes")
    proxy_url = os.getenv("NEW_API_URL", "http://172.30.1.11:8000")
    mode = "로컬 GPU 추론" if use_local else f"프록시({proxy_url})"
    print(f"[server] USE_LOCAL_INFERENCE={os.getenv('USE_LOCAL_INFERENCE', '')!r} -> {mode}")


class FolderPathRequest(BaseModel):
    folderPath: str
    run: Optional[str] = None  # "d8_speaker_internal" | "d8_speaker_independent" | None -> 기본
    csvFileName: Optional[str] = None  # 지정 시 해당 CSV만 로드 (개별 조회). 없으면 기존처럼 1개 자동 선택


class InferenceRequest(BaseModel):
    """추론 요청. 출력은 Blendshape만 (FLAME/PCA 미포함)."""
    audio_base64: str
    enable_eye_blink: Optional[bool] = True
    # SERVICE_MODEL_STRUCTURE: 사용자 선택 항목
    run: Optional[str] = None       # 학습 모델(실험 폴더명). 없으면 기본 run 사용
    checkpoint: Optional[str] = None  # 체크포인트 파일명 (예: best.pt). 없으면 best.pt
    speaker: Optional[int] = None   # 화자(identity) 인덱스. 출력 스타일 적용. 없으면 0
    compare_vanilla: Optional[bool] = False  # True이면 Vanilla UniTalker와 비교 추론 수행
    source_audio_filename: Optional[str] = None  # 디버그용 원본 파일명 → inference_debug/*_input_source.txt (GT 비교 미사용)
    engine: Optional[str] = None  # "pytorch"(기본) | "onnx" — FaceTool 계약


class EpochResponse(BaseModel):
    success: bool
    csvContent: Optional[str] = None
    audioBase64: Optional[str] = None
    audioFileName: Optional[str] = None
    csvFileName: Optional[str] = None
    folderName: Optional[str] = None
    gtCsvContent: Optional[str] = None
    gtFileName: Optional[str] = None
    lve: Optional[float] = None  # LVE 메트릭 (test/LVE)
    lvd: Optional[float] = None  # LVD 메트릭 (test/LVD)
    valLve: Optional[float] = None  # Validation LVE 메트릭 (val/LVE)
    valLvd: Optional[float] = None  # Validation LVD 메트릭 (val/LVD)
    error: Optional[str] = None


class EpochsListResponse(BaseModel):
    success: bool
    epochs: List[str]
    basePath: str
    error: Optional[str] = None


class EpochFilesListResponse(BaseModel):
    """폴더 내 파일 목록 (프론트 개별 CSV 선택용)"""
    success: bool
    folderPath: str
    files: List[dict]  # [{"name": "xxx_style0.csv", "type": "csv"}, ...]
    wavFileName: Optional[str] = None  # 폴더 내 대표 WAV 1개 (재생용). 없으면 null
    error: Optional[str] = None


class GtResponse(BaseModel):
    """GET /api/gt 응답 (클립명으로 GT CSV 조회)"""
    success: bool
    gtCsvContent: Optional[str] = None
    gtFileName: Optional[str] = None
    error: Optional[str] = None


class RunsListResponse(BaseModel):
    success: bool
    runs: List[dict]  # [{"id": "results_d8_speaker_internal", "path": "...", "epochCount": 5}, ...]
    baseDir: str
    error: Optional[str] = None


class CheckpointsListResponse(BaseModel):
    success: bool
    checkpoints: List[str]  # ["best.pt", ...]
    run: str
    error: Optional[str] = None


class SpeakersResponse(BaseModel):
    success: bool
    count: int
    indices: List[int]  # [0, 1, ..., count-1]
    run: str
    error: Optional[str] = None


def _get_project_root() -> Path:
    """서버 스크립트 기준 프로젝트 루트 (trainFramework/)"""
    return Path(__file__).resolve().parent.parent


def _get_unitalker_results_dir() -> Path:
    """models/unitalker/ 경로"""
    return _get_project_root() / "models/unitalker"


# 서비스/API run 식별자
PRETRAINED_RUN_ID = "pretrained"
OFFICIAL_PRETRAINED_CKPT = "UniTalker-B-D0-D7.pt"
# v5 서비스 범위外: 디스크 보존만, /api/runs 목록에 노출하지 않음
_LEGACY_EXCLUDED_RUN_IDS = frozenset()

def _get_pretrained_dir() -> Path:
    return _get_unitalker_results_dir() / "pretrained"


def _get_official_pretrained_checkpoint() -> Path:
    return _get_pretrained_dir() / OFFICIAL_PRETRAINED_CKPT


def _resolve_checkpoint_path(run_id: str, checkpoint_name: Optional[str] = None) -> Path:
    """run_id + 체크포인트 파일명 -> 절대 경로."""
    root = _get_project_root()
    if run_id == PRETRAINED_RUN_ID:
        return _get_pretrained_dir() / (checkpoint_name or OFFICIAL_PRETRAINED_CKPT)
    return root / "models" / "unitalker" / run_id / (checkpoint_name or "best.pt")


def _append_pretrained_run(runs: List[dict]) -> None:
    """UniTalker 공식 제공 체크포인트를 run 목록에 추가."""
    ckpt = _get_official_pretrained_checkpoint()
    if not ckpt.is_file():
        return
    pt_dir = _get_pretrained_dir()
    from server.run_config import get_run_metadata
    from server.service_manifest import resolve_run_service_meta
    csv_meta = get_run_metadata(PRETRAINED_RUN_ID)
    svc = resolve_run_service_meta(PRETRAINED_RUN_ID, csv_meta, root=_get_project_root())
    runs.append({
        "id": PRETRAINED_RUN_ID,
        "path": str(pt_dir),
        "blendshapePath": str(pt_dir),
        "epochCount": 0,
        "hasBlendshapeSamples": False,
        "label": svc.get("label") or "UniTalker official (UniTalker-B-D0-D7)",
        "strategy": svc.get("strategy") or "PRETRAINED",
        "output_kind": svc.get("output_kind") or "arkit_51",
        "channels": svc.get("channels") or 51,
        "dataset": svc.get("dataset") or "D3_3DETF",
        "engines": svc.get("engines") or {"pytorch": True, "onnx": False},
        "dnaId": svc.get("dna_id"),
        "hasCheckpoint": bool(svc.get("has_checkpoint")),
        "featureNamesHash": svc.get("feature_names_hash"),
        "isDefault": bool(svc.get("is_default")),
    })


def list_available_runs() -> List[dict]:
    """
    models/unitalker 하위에서 실제 학습 결과 디렉토리를 동적으로 탐지해 반환.
    - .pt 체크포인트
    - train.log/train_console.log
    - TensorBoard 이벤트 파일
    - blendshape_samples, inference_debug
    중 하나라도 있으면 run 후보로 간주한다.
    """
    unitalker_dir = _get_unitalker_results_dir()
    if not unitalker_dir.exists():
        return []
    runs = []
    for item in unitalker_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name == PRETRAINED_RUN_ID:
            continue
        run_id = item.name.strip()
        # 비정상 공백/개행이 섞인 폴더명은 무시한다.
        if run_id != item.name:
            continue
        if run_id in _LEGACY_EXCLUDED_RUN_IDS:
            continue
        blendshape_path = item / "blendshape_samples"
        inference_debug_path = item / "inference_debug"
        has_blendshape_samples = blendshape_path.exists() and blendshape_path.is_dir()
        has_inference_debug = inference_debug_path.exists() and inference_debug_path.is_dir()
        has_train_log = (item / "train.log").is_file() or (item / "train_console.log").is_file()
        pt_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".pt"]
        has_checkpoint = len(pt_files) > 0
        has_tb_event = any(
            f.is_file() and f.name.startswith("events.out.tfevents.")
            for f in item.iterdir()
        )

        # run 후보 판정: 학습/추론 산출물 중 하나라도 있으면 포함
        if not (has_blendshape_samples or has_inference_debug or has_train_log or has_checkpoint or has_tb_event):
            continue

        epoch_count = 0
        blend_path_used = str(item)
        if has_blendshape_samples:
            epoch_dirs = [d for d in blendshape_path.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
            epoch_count = len(epoch_dirs)
            if epoch_count == 0:
                if (blendshape_path / "best1").exists():
                    epoch_count += 1
                if (blendshape_path / "best").exists():
                    epoch_count += 1
                if epoch_count == 0:
                    epoch_count = 1
            blend_path_used = str(blendshape_path)
        elif has_inference_debug:
            csv_count = sum(1 for f in inference_debug_path.iterdir() if f.is_file() and f.suffix == ".csv")
            if csv_count > 0:
                epoch_count = 1

        mtime = item.stat().st_mtime
        from server.run_config import get_run_metadata
        from server.service_manifest import resolve_run_service_meta
        csv_meta = get_run_metadata(run_id)
        svc = resolve_run_service_meta(run_id, csv_meta, root=_get_project_root())
        output_kind = svc.get("output_kind") or infer_output_kind_for_run(run_id, csv_meta)
        runs.append({
            "id": run_id,
            "path": str(item),
            "blendshapePath": blend_path_used,
            "epochCount": epoch_count,
            "hasBlendshapeSamples": has_blendshape_samples,
            "label": svc.get("label") or run_id,
            "strategy": svc.get("strategy"),
            "output_kind": output_kind,
            "channels": svc.get("channels") or (260 if output_kind == "metahuman_260" else 51),
            "dataset": svc.get("dataset"),
            "engines": svc.get("engines") or {"pytorch": has_checkpoint, "onnx": False},
            "dnaId": svc.get("dna_id"),
            "hasCheckpoint": bool(svc.get("has_checkpoint")),
            "featureNamesHash": svc.get("feature_names_hash"),
            "isDefault": bool(svc.get("is_default")),
            "mtime": mtime,
        })
    _append_pretrained_run(runs)
    runs.sort(key=lambda r: (0 if r["id"] == PRETRAINED_RUN_ID else 1, -r.get("mtime", 0)))
    for r in runs:
        r.pop("mtime", None)
    return runs


def _resolve_run_to_folder(run: Optional[str]) -> str:
    """run(폴더명 또는 CSV 별칭)을 실제 폴더명으로 변환. run 없으면 스캔 결과 중 첫 번째 사용"""
    from server.run_config import resolve_run_to_folder as _resolve
    available_ids = [r["id"] for r in list_available_runs()]
    resolved = _resolve(run, available_ids)
    if resolved:
        return resolved
    # run 없고 스캔 결과도 없을 때만 기본 폴더명 (경로는 존재하지 않을 수 있음)
    return "results"


def get_base_path(run: Optional[str] = None):
    """기본 경로 가져오기. run=폴더명 또는 레거시 별칭. blendshape_samples 없으면 run 루트(추론 결과 inference_debug 서빙용)."""
    root = _get_project_root()
    if os.getenv("TRAIN_RESULTS_BASE_PATH"):
        p = Path(os.getenv("TRAIN_RESULTS_BASE_PATH").replace("~", os.getenv("HOME", str(Path.home()))))
        return Path(p)
    folder = _resolve_run_to_folder(run)
    run_root = root / "models/unitalker" / folder
    blendshape_path = run_root / "blendshape_samples"
    if blendshape_path.exists() and blendshape_path.is_dir():
        return Path(blendshape_path)
    return Path(run_root)


def get_data_root():
    """데이터셋 루트 경로 가져오기"""
    root = _get_project_root()
    data_root = os.getenv("DATA_ROOT", str(root / "data/unitalker_data"))
    expanded = data_root.replace("~", os.getenv("HOME", str(Path.home())))
    return Path(expanded)


def get_results_path(run: Optional[str] = None):
    """학습 결과 루트 경로 가져오기 (TensorBoard 로그 위치). run=폴더명 또는 레거시 별칭"""
    root = _get_project_root()
    if os.getenv("TRAIN_RESULTS_PATH"):
        p = Path(os.getenv("TRAIN_RESULTS_PATH").replace("~", os.getenv("HOME", str(Path.home()))))
    else:
        folder = _resolve_run_to_folder(run)
        p = root / "models/unitalker" / folder
    return Path(p)


def get_epoch_metrics_from_tensorboard(epoch: int, results_path: Path) -> dict:
    """
    TensorBoard 이벤트 파일에서 특정 epoch의 LVE/LVD 메트릭 읽기
    여러 이벤트 파일을 검색하여 해당 epoch의 메트릭을 찾습니다.
    
    Returns:
        dict: {'test/LVE': float, 'test/LVD': float, 'val/LVE': float, 'val/LVD': float} 또는 None
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # TensorBoard 이벤트 파일 찾기
        event_files = sorted(
            glob.glob(str(results_path / "events.out.tfevents.*")),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not event_files:
            print(f"[get_epoch_metrics] No TensorBoard event files found in {results_path}")
            return None
        
        # 모든 이벤트 파일을 검색하여 해당 epoch 찾기
        # 최신 파일부터 검색하되, epoch가 없으면 이전 파일도 확인
        metrics = {}
        
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                
                scalar_tags = ea.Tags().get('scalars', [])
                
                # 메트릭 키와 fallback 키 매핑
                metric_mappings = {
                    'test/LVE': ['test/LVE', 'test/LVE_mixed_dataset'],
                    'test/LVD': ['test/LVD', 'test/LVD_mixed_dataset'],
                    'val/LVE': ['val/LVE', 'val/LVE_mixed_dataset'],
                    'val/LVD': ['val/LVD', 'val/LVD_mixed_dataset'],
                }
                
                for target_key, possible_keys in metric_mappings.items():
                    if target_key in metrics:  # 이미 찾았으면 스킵
                        continue
                        
                    for key in possible_keys:
                        if key in scalar_tags:
                            scalar_events = ea.Scalars(key)
                            for event in scalar_events:
                                if event.step == epoch:
                                    metrics[target_key] = event.value
                                    print(f"[get_epoch_metrics] Found {target_key} = {event.value:.6e} from {os.path.basename(event_file)} at epoch {epoch}")
                                    break
                            if target_key in metrics:
                                break
                
                # 모든 메트릭을 찾았으면 종료
                if len(metrics) == 4:
                    break
                    
            except Exception as e:
                print(f"[get_epoch_metrics] Error reading {os.path.basename(event_file)}: {e}")
                continue
        
        # 이벤트 파일 읽기
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # 스칼라 태그 목록 확인
        scalar_tags = ea.Tags().get('scalars', [])
        
        metrics = {}
        
        # 메트릭 키와 fallback 키 매핑
        # 먼저 기본 키를 시도하고, 없으면 _mixed_dataset 버전을 시도
        metric_mappings = {
            'test/LVE': ['test/LVE', 'test/LVE_mixed_dataset'],
            'test/LVD': ['test/LVD', 'test/LVD_mixed_dataset'],
            'val/LVE': ['val/LVE', 'val/LVE_mixed_dataset'],
            'val/LVD': ['val/LVD', 'val/LVD_mixed_dataset'],
        }
        
        for target_key, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in scalar_tags:
                    # 해당 epoch의 값 찾기
                    scalar_events = ea.Scalars(key)
                    for event in scalar_events:
                        if event.step == epoch:
                            metrics[target_key] = event.value
                            print(f"[get_epoch_metrics] Found {target_key} = {event.value:.6e} from {key} at epoch {epoch}")
                            break
                    if target_key in metrics:
                        break
        
        if metrics:
            return metrics
        else:
            print(f"[get_epoch_metrics] No metrics found for epoch {epoch} (available tags: {[t for t in scalar_tags if 'LVE' in t or 'LVD' in t][:10]})")
            return None
            
    except ImportError:
        print("[get_epoch_metrics] Warning: tensorboard package not installed")
        return None
    except Exception as e:
        print(f"[get_epoch_metrics] Error reading TensorBoard: {e}")
        import traceback
        traceback.print_exc()
        return None


# 서비스 출력용 ARKit 51 blendshape 이름 (3DETF/EmoTalk 알파벳순, tongueOut 제외)
# pretrained 모델: 3DETF_blendshape_weight head(52-dim) → tongueOut 제거 → 51-dim
# finetuned 모델: inhouse_blendshape_weight head(51-dim) → D8 학습 데이터가 알파벳순
# 모든 서비스 출력이 이 순서를 따름
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

# 3DETF/EmoTalk 52채널: 알파벳순 ARKit + tongueOut
_ARKIT_52_SORTED = sorted(_ARKIT_51_BS_NAMES + ['tongueOut'])


def infer_output_kind_for_run(run_id: str, meta: dict) -> str:
    from server.run_config import infer_output_kind
    if meta.get("output_kind"):
        return meta["output_kind"]
    ds = meta.get("dataset_for_blendshape") or ""
    return infer_output_kind(ds, run_id=run_id)


def prepare_service_blendshape_output(
    bs_arr,
    dataset_for_blendshape: str,
    data_root: str,
    run_id: str = "",
    output_kind: Optional[str] = None,
):
    """
    API/프론트용 blendshape 정렬.
    - arkit_51: 52ch tongueOut 제거 후 51 ARKit 이름
    - metahuman_260: D8 feature_names 260열
    """
    import numpy as np
    from models.unitalker.dataset.dataset_config import dataset_config
    from models.unitalker.riglogic_eval.controls import expand_to_full_columns

    arr = np.asarray(bs_arr, dtype=np.float32)
    names = None
    ds_key = dataset_for_blendshape.replace("_3DETF", "")

    try:
        if ds_key in dataset_config:
            fn_path = Path(data_root) / dataset_config[ds_key]["dirname"] / "feature_names.txt"
            if fn_path.is_file():
                loaded = [x.strip() for x in fn_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                if len(loaded) == arr.shape[1]:
                    names = loaded
    except Exception:
        names = None

    is_metahuman = (
        output_kind == "metahuman_260"
        or "metahuman" in dataset_for_blendshape.lower()
        or arr.shape[1] == 260
    )
    if is_metahuman:
        d8_fn = Path(data_root) / dataset_config["D8"]["dirname"] / "feature_names.txt"
        full_names = None
        if d8_fn.is_file():
            full_names = [
                x.strip() for x in d8_fn.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
        partial_names = names or [f"ctrl_{i:03d}" for i in range(arr.shape[1])]
        if full_names and len(full_names) == 260:
            if arr.shape[1] != 260:
                if arr.shape[1] < 260 and set(partial_names).issubset(set(full_names)):
                    arr = expand_to_full_columns(arr, partial_names, full_names)
                elif arr.shape[1] < 260:
                    padded = np.zeros((arr.shape[0], 260), dtype=np.float32)
                    padded[:, :arr.shape[1]] = arr
                    arr = padded
            names = full_names
        else:
            names = partial_names
        output_kind = "metahuman_260"
        print(
            f"[blendshape] MetaHuman rig control {arr.shape[1]}ch "
            f"(run={run_id}, dataset={dataset_for_blendshape})"
        )
    elif arr.shape[1] == 52:
        drop_idx = None
        if names and "tongueOut" in names:
            drop_idx = names.index("tongueOut")
        else:
            drop_idx = _ARKIT_52_SORTED.index("tongueOut")
        arr = np.delete(arr, drop_idx, axis=1)
        if names and drop_idx is not None:
            names = [n for i, n in enumerate(names) if i != drop_idx]
        else:
            names = list(_ARKIT_51_BS_NAMES)
        print(
            f"[blendshape] 52->51: tongueOut 제거 (run={run_id}, dataset={dataset_for_blendshape})"
        )
    elif arr.shape[1] == 51:
        names = list(_ARKIT_51_BS_NAMES)
    elif arr.shape[1] < 51:
        names = list(_ARKIT_51_BS_NAMES[: arr.shape[1]])
    else:
        names = names or [f"ctrl_{i:03d}" for i in range(arr.shape[1])]

    if names is None:
        names = list(_ARKIT_51_BS_NAMES[: arr.shape[1]])

    if not output_kind:
        output_kind = "arkit_51"

    return arr, names, output_kind


def arkit_full_name_to_glb_morph(name: str) -> str:
    """ARKit 전체 이름(browDownLeft) → GLB morph 축약 이름(browDownL)."""
    return name.replace("Left", "L").replace("Right", "R")


def get_blendshape_names(csv_filename: Optional[str] = None):
    """서비스 출력용 blendshape 이름 반환 (ARKit 51개, 알파벳순).
    
    3DETF(EmoTalk) 기반 표준 ARKit 순서를 사용.
    csv_filename 인자는 하위 호환성을 위해 유지하나, 항상 동일한 51개 이름을 반환.
    """
    return list(_ARKIT_51_BS_NAMES)


def find_gt_file(csv_filename: str, data_root: Path) -> Optional[Path]:
    """
    CSV 파일명에서 GT 파일 경로 찾기
    
    예: D3_RD_Radio11_001_03.csv -> 
        data_root/D3D4_3DETF/D3_HDTF/blendshape/RD_Radio11_001.npy
    예: LipReading_sample.csv -> data_root/LipReading_Dataset_Video/train/sample.npy
    """
    # CSV 파일명 파싱
    name_without_ext = csv_filename.replace('.csv', '')
    
    dataset_name = None
    sample_name = None

    for prefix in ['LipReading', 'D3', 'D4', 'D6', 'D7']:
        if name_without_ext.startswith(prefix + '_'):
            dataset_name = prefix
            sample_name = name_without_ext[len(prefix) + 1:]  # prefix_ 이후
            break
    
    if dataset_name is None or sample_name is None:
        return None
    
    # 데이터셋 경로 매핑
    dataset_paths = {
        'D3': 'D3D4_3DETF/D3_HDTF',
        'D4': 'D3D4_3DETF/D4_RAVDESS',
        'D6': 'D6_unitalker_Chinese_speech',
        'D7': 'D7_unitalker_song',
        'LipReading': 'LipReading_Dataset_Video',
    }
    
    if dataset_name not in dataset_paths:
        return None
    
    # 샘플 이름에서 마지막 숫자 부분 제거 (예: RD_Radio11_001_03 -> RD_Radio11_001)
    # 마지막 _숫자 패턴 제거
    sample_parts = sample_name.rsplit('_', 1)
    if len(sample_parts) == 2 and sample_parts[1].isdigit():
        base_sample_name = sample_parts[0]
    else:
        base_sample_name = sample_name
    
    subpath = dataset_paths[dataset_name]
    if dataset_name in ('LipReading', 'D6', 'D7'):
        for subdir in ('train', 'test', 'val'):
            gt_path = data_root / subpath / subdir / f'{sample_name}.npy'
            if gt_path.exists():
                return gt_path
    else:
        gt_path = data_root / subpath / 'blendshape' / f'{base_sample_name}.npy'
        if gt_path.exists():
            return gt_path
        gt_path_alt = data_root / subpath / 'blendshape' / f'{sample_name}.npy'
        if gt_path_alt.exists():
            return gt_path_alt
    return None


def _gt_from_source_filename(full_path: Path, tag: str, data_root: Path) -> Optional[tuple]:
    """
    inference_debug에 저장된 {tag}_input_source.txt(원본 오디오 파일명)가 있으면
    해당 이름으로 find_gt_file를 시도해 (dataset_prefix, sample_name) 또는 None 반환.
    """
    source_file = full_path / f"{tag}_input_source.txt"
    if not source_file.exists():
        return None
    try:
        raw = source_file.read_text(encoding="utf-8").strip()
        stem = raw.replace(".wav", "").replace(".WAV", "").strip()
        if not stem:
            return None
        # 프론트에서 "speaker0_파일명" 형태로 올 수 있음 → GT 조회용으로 speaker 접두사 제거
        stem = re.sub(r"^(?:speaker\d+_|s\d+_)", "", stem)
        if not stem:
            return None
    except Exception:
        return None
    prefixes = ["D3", "D4", "D6", "D7", "LipReading"]
    for prefix in prefixes:
        csv_filename = f"{prefix}_{stem}.csv"
        if find_gt_file(csv_filename, data_root):
            return (prefix, stem)
    return None


def _file_hash(path: Path) -> str:
    """파일 내용 MD5 해시 (동일 오디오 감지용)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dataset_audio_match(input_wav_path: Path, data_root: Path) -> Optional[tuple]:
    """
    inference_debug에 저장된 입력 WAV가 학습/검증 데이터셋의 어느 샘플과 동일한지 찾기.
    반환: (dataset_prefix, sample_name) 예: ('D3', 'RD_Radio1_000_00'), ('LipReading', 'lip_J_1_...') 또는 None.
    """
    if not input_wav_path.exists() or not input_wav_path.is_file():
        return None
    input_size = input_wav_path.stat().st_size
    input_hash = _file_hash(input_wav_path)
    # 스캔 대상: (데이터셋 디렉토리명, find_gt_file용 prefix). D3/D4는 하위 전체에서 .wav 검색
    datasets_with_subdirs = [
        ("LipReading_Dataset_Video", "LipReading"),
        ("D6_unitalker_Chinese_speech", "D6"),
        ("D7_unitalker_song", "D7"),
    ]
    for dirname, prefix in datasets_with_subdirs:
        base = data_root / dirname
        if not base.exists():
            continue
        for subdir in ("train", "test", "val"):
            sub = base / subdir
            if not sub.exists():
                continue
            for wav_path in sub.glob("*.wav"):
                if not wav_path.is_file():
                    continue
                if wav_path.stat().st_size != input_size:
                    continue
                if _file_hash(wav_path) == input_hash:
                    return (prefix, wav_path.stem)
    # D3, D4: train/test/val가 아닌 blendshape/audio 등 다른 구조일 수 있으므로 하위 전체 검색
    for dirname, prefix in [("D3D4_3DETF/D3_HDTF", "D3"), ("D3D4_3DETF/D4_RAVDESS", "D4")]:
        base = data_root / dirname
        if not base.exists():
            continue
        for wav_path in base.rglob("*.wav"):
            if not wav_path.is_file():
                continue
            if wav_path.stat().st_size != input_size:
                continue
            if _file_hash(wav_path) == input_hash:
                return (prefix, wav_path.stem)
    return None


def npy_to_csv(npy_path: Path, bs_names: Optional[List[str]] = None) -> Optional[str]:
    """NPY 파일을 CSV 형식으로 변환"""
    try:
        gt_data = np.load(str(npy_path))
        
        # blendshape 이름이 없으면 기본값 사용
        if bs_names is None:
            num_blendshapes = gt_data.shape[1] if len(gt_data.shape) > 1 else 52
            bs_names = [f'bs_{i}' for i in range(num_blendshapes)]
        
        # CSV 헤더
        csv_header = ['frame', 'time_sec'] + bs_names
        
        # CSV 행 생성
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=csv_header)
        writer.writeheader()
        
        for frame_idx in range(gt_data.shape[0]):
            row = {
                'frame': frame_idx,
                'time_sec': frame_idx / 30.0  # FPS = 30
            }
            for i, bs_name in enumerate(bs_names):
                if i < gt_data.shape[1]:
                    row[bs_name] = float(gt_data[frame_idx, i])
                else:
                    row[bs_name] = 0.0
            writer.writerow(row)
        
        return output.getvalue()
    except Exception as e:
        print(f"[npy_to_csv] Error converting {npy_path}: {e}")
        return None


@app.get("/", tags=["Health"])
async def root():
    """
    서버 상태 확인
    
    서버가 정상적으로 실행 중인지 확인합니다.
    """
    return {
        "message": "Training Results API Server",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/api/runs", response_model=RunsListResponse, tags=["Runs"])
async def list_runs():
    """
    사용 가능한 학습 모델(run) 목록 조회.
    Front: "학습 모델" 선택용.
    """
    try:
        runs = list_available_runs()
        base_dir = str(_get_unitalker_results_dir())
        return RunsListResponse(
            success=True,
            runs=runs,
            baseDir=base_dir,
        )
    except Exception as error:
        print(f"[API/runs] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/checkpoints", response_model=CheckpointsListResponse, tags=["Service"])
async def list_checkpoints(run: Optional[str] = None):
    """
    해당 run(학습 모델)에서 사용 가능한 체크포인트 목록.
    Front: "체크포인트" 선택용. 출력은 Blendshape 추론에 사용.
    """
    try:
        folder = _resolve_run_to_folder(run)
        results_path = get_results_path(run=run)
        if not results_path.exists():
            return CheckpointsListResponse(success=True, checkpoints=[], run=folder)
        pt_files = sorted([f.name for f in results_path.iterdir() if f.is_file() and f.suffix == ".pt"])
        if folder == PRETRAINED_RUN_ID and OFFICIAL_PRETRAINED_CKPT in pt_files:
            pt_files = [OFFICIAL_PRETRAINED_CKPT] + [f for f in pt_files if f != OFFICIAL_PRETRAINED_CKPT]
        if not pt_files and (results_path / "best.pt").exists():
            pt_files = ["best.pt"]
        return CheckpointsListResponse(success=True, checkpoints=pt_files or [], run=folder)
    except Exception as error:
        print(f"[API/checkpoints] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/speakers", response_model=SpeakersResponse, tags=["Service"])
async def list_speakers(run: Optional[str] = None):
    """
    해당 run(학습 모델)의 화자(identity) 수.
    Front: "화자" 선택 옵션 개수용. 학습된 화자만 존재하며, 입력 오디오는 학습 참여 여부 무관.
    """
    try:
        folder = _resolve_run_to_folder(run)
        results_path = get_results_path(run=run)
        if folder == PRETRAINED_RUN_ID:
            ckpt_path = _get_official_pretrained_checkpoint()
        else:
            ckpt_path = results_path / "best.pt"
        if not ckpt_path.exists():
            return SpeakersResponse(success=True, count=0, indices=[], run=folder)
        try:
            import torch
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            key = "decoder.learnable_style_emb.weight"
            if key in state:
                n = state[key].shape[0]
                return SpeakersResponse(success=True, count=n, indices=list(range(n)), run=folder)
        except Exception as e:
            print(f"[API/speakers] Could not read checkpoint: {e}")
        return SpeakersResponse(success=True, count=1, indices=[0], run=folder)
    except Exception as error:
        print(f"[API/speakers] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/epochs", response_model=EpochsListResponse, tags=["Epoch"])
async def list_epochs(run: Optional[str] = None):
    """
    에폭 폴더 목록 조회
    
    blendshape_samples 아래의 epoch_*, best, best1 폴더를 반환합니다.
    run: 결과 폴더명 (GET /api/runs로 목록 조회)
    - 레거시: d8_speaker_internal, d8_speaker_independent, d3d4
    - 폴더명: results_d8_speaker_internal, results_d8_speaker_internal_LVE_LOSS, results 등
    """
    try:
        base_path = get_base_path(run=run)
        
        print(f"[API/epochs] Listing folders in: {base_path}")

        # 폴더가 없으면 빈 목록 반환 (학습 전이거나 blendshape_samples 미생성)
        if not base_path.exists() or not base_path.is_dir():
            return EpochsListResponse(
                success=True,
                epochs=[],
                basePath=str(base_path)
            )

        # 폴더 목록 읽기 (epoch_* + best + best1, 또는 inference_debug)
        epochs = []
        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith("epoch_"):
                epochs.append(item.name)
            elif item.name == "best":
                epochs.append("best")
            elif item.name == "best1":
                epochs.append("best1")
            elif item.name == "inference_debug":
                if any(f.suffix == ".csv" for f in item.iterdir() if f.is_file()):
                    epochs.append("inference_debug")

        # run에 blendshape_samples가 있으면 base_path가 그쪽이라 inference_debug는 run 루트에 있음 → 별도 추가
        if base_path.name == "blendshape_samples":
            run_root = base_path.parent
            inference_debug_dir = run_root / "inference_debug"
            if inference_debug_dir.exists() and inference_debug_dir.is_dir():
                if any(f.suffix == ".csv" for f in inference_debug_dir.iterdir() if f.is_file()):
                    if "inference_debug" not in epochs:
                        epochs.append("inference_debug")

        # 정렬 (epoch_001, ... / best, best1 / inference_debug)
        def _epoch_sort_key(x):
            if x.startswith("epoch_"):
                return (0, x)
            if x == "best":
                return (1, "")
            if x == "best1":
                return (2, "")
            if x == "inference_debug":
                return (3, "")
            return (4, x)
        epochs.sort(key=_epoch_sort_key)

        return EpochsListResponse(
            success=True,
            epochs=epochs,
            basePath=str(base_path)
        )

    except HTTPException:
        raise
    except Exception as error:
        print(f"[API/epochs] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/epoch/list", response_model=EpochsListResponse, tags=["Epoch"])
async def list_epochs_alt(run: Optional[str] = None):
    """
    에폭 폴더 목록 조회 (대체 경로)
    
    `/api/epochs`와 동일. run=폴더명 (GET /api/runs 참고)
    """
    return await list_epochs(run=run)


@app.get("/api/epoch/files", response_model=EpochFilesListResponse, tags=["Epoch"])
async def list_epoch_folder_files(run: Optional[str] = None, folderPath: Optional[str] = None):
    """
    에폭 폴더 내 파일 목록 조회 (프론트 개별 CSV 선택용)
    
    - **run**: 결과 폴더명 (GET /api/runs 참고). 생략 시 기본 run
    - **folderPath**: 에폭 폴더 이름 (예: "best", "best1", "epoch_001")
    - **응답**: files (CSV/WAV 목록), wavFileName (재생용 대표 WAV 1개)
    """
    try:
        if not folderPath or not folderPath.strip():
            raise HTTPException(status_code=400, detail="folderPath is required")
        normalized = folderPath.replace("..", "").strip().strip("/")
        if normalized.upper().startswith("EPOCH_"):
            normalized = "epoch_" + normalized.split("_", 1)[-1]
        # inference_debug는 run 루트 아래에 있음 (blendshape_samples와 별도)
        if normalized == "inference_debug":
            full_path = get_results_path(run=run) / "inference_debug"
        else:
            base_path = get_base_path(run=run)
            full_path = base_path / normalized
        if not full_path.exists() or not full_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder not found: {normalized}")
        files = []
        wav_file = None
        for f in full_path.iterdir():
            if not f.is_file():
                continue
            name = f.name
            if name.endswith(".csv"):
                files.append({"name": name, "type": "csv"})
            elif name.endswith(".wav") or name.endswith(".mp3"):
                files.append({"name": name, "type": "wav"})
                if wav_file is None:
                    wav_file = name
        files.sort(key=lambda x: (0 if x["type"] == "csv" else 1, x["name"]))
        return EpochFilesListResponse(
            success=True,
            folderPath=normalized,
            files=files,
            wavFileName=wav_file,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API/epoch/files] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gt", response_model=GtResponse, tags=["Epoch"])
async def get_gt_by_name(
    name: str,
    run: Optional[str] = None,
    folderPath: Optional[str] = None,
):
    """
    [레거시] 클립명으로 GT CSV 조회. FaceTool 학습 결과 평가는 로컬 NPY를 사용하며 이 API를 호출하지 않습니다.
    docs/FACETOOL_EVAL_BACKEND_SCOPE.md 참고.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    name = name.strip()

    # inference_debug: (1) 폴더 내 _gt.csv 검색 (2) 없으면 입력 WAV를 데이터셋과 매칭해 GT NPY 조회
    if folderPath and folderPath.strip() == "inference_debug" and run:
        full_path = get_results_path(run=run) / "inference_debug"
        if full_path.exists() and full_path.is_dir():
            candidates = [
                f"{name}_gt.csv",
            ]
            if name.endswith("_input_blendshape"):
                candidates.append(f"{name.replace('_input_blendshape', '')}_gt.csv")
            for gt_file in candidates:
                gt_path = full_path / gt_file
                if gt_path.exists():
                    content = gt_path.read_text(encoding="utf-8")
                    return GtResponse(
                        success=True,
                        gtCsvContent=content,
                        gtFileName=gt_file,
                    )
            # 폴더에 _gt.csv 없음: (1) 원본 파일명 sidecar 확인 (2) WAV 내용 매칭
            tag = name.replace("_input_blendshape", "") if name.endswith("_input_blendshape") else name
            data_root = get_data_root()
            match = _gt_from_source_filename(Path(full_path), tag, data_root)
            if not match:
                input_wav = full_path / f"{tag}_input.wav"
                match = find_dataset_audio_match(Path(input_wav), data_root)
            if match:
                dataset_prefix, sample_name = match
                csv_filename = f"{dataset_prefix}_{sample_name}.csv"
                gt_npy_path = find_gt_file(csv_filename, data_root)
                if gt_npy_path:
                    bs_names = get_blendshape_names(csv_filename=csv_filename)
                    gt_csv_content = npy_to_csv(gt_npy_path, bs_names)
                    if gt_csv_content:
                        return GtResponse(
                            success=True,
                            gtCsvContent=gt_csv_content,
                            gtFileName=gt_npy_path.name,
                        )
        return GtResponse(success=True, gtCsvContent=None, gtFileName=None)

    # 데이터셋 클립: find_gt_file로 NPY 찾아서 CSV 변환
    data_root = get_data_root()
    csv_filename = name if name.endswith(".csv") else f"{name}.csv"
    gt_npy_path = find_gt_file(csv_filename, data_root)
    if gt_npy_path:
        bs_names = get_blendshape_names(csv_filename=csv_filename)
        gt_csv_content = npy_to_csv(gt_npy_path, bs_names)
        if gt_csv_content:
            return GtResponse(
                success=True,
                gtCsvContent=gt_csv_content,
                gtFileName=gt_npy_path.name,
            )
    raise HTTPException(
        status_code=404,
        detail=f"GT npy not found for clip: {name}",
    )


@app.post("/api/epoch", response_model=EpochResponse, tags=["Epoch"])
async def load_epoch_results(request: FolderPathRequest):
    """
    [레거시] 에폭 학습 결과 로드 (CSV/WAV, 선택적 gtCsvContent/LVE).
    FaceTool GT vs Prediction 평가는 로컬 NPY를 사용 — gtCsvContent는 평가 UI 필수 아님.
    docs/FACETOOL_EVAL_BACKEND_SCOPE.md 참고.
    """
    try:
        folder_path = request.folderPath

        if not folder_path:
            raise HTTPException(status_code=400, detail="folderPath is required")

        # 경로 정규화 및 보안 검사
        normalized_path = folder_path.replace("..", "").strip().strip("/")
        # EPOCH_002 -> epoch_002 (대소문자 통일)
        if normalized_path.upper().startswith("EPOCH_"):
            normalized_path = "epoch_" + normalized_path.split("_", 1)[-1]

        # inference_debug는 run 루트 아래에 있음 (blendshape_samples와 별도)
        if normalized_path == "inference_debug":
            full_path = get_results_path(run=request.run) / "inference_debug"
        else:
            base_path = get_base_path(run=request.run)
            full_path = base_path / normalized_path

        print(f"[API/epoch] Reading folder: {full_path}")

        # 폴더 존재 확인
        if not full_path.exists() or not full_path.is_dir():
            print(f"[API/epoch] Not found: {full_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Folder not found: {normalized_path}"
            )

        # 폴더 내 파일 목록 읽기
        files = [f.name for f in full_path.iterdir() if f.is_file()]

        # CSV 선택: csvFileName 지정 시 해당 파일, 없으면 가장 긴 이름 1개
        csv_files = sorted([f for f in files if f.endswith(".csv")], key=lambda x: -len(x))
        if getattr(request, "csvFileName", None):
            requested = request.csvFileName.strip()
            csv_file = requested if (requested in files and requested.endswith(".csv")) else (csv_files[0] if csv_files else None)
        else:
            csv_file = csv_files[0] if csv_files else None

        # WAV 파일 찾기 (CSV와 동일한 이름의 WAV 파일 우선)
        wav_file = None
        if csv_file:
            # CSV와 같은 이름의 WAV 찾기
            csv_base = csv_file.rsplit('.', 1)[0]
            matching_wav = f"{csv_base}.wav"
            if matching_wav in files:
                wav_file = matching_wav
            else:
                # 없으면 아무 WAV 파일
                wav_file = next((f for f in files if f.endswith(".wav") or f.endswith(".mp3")), None)

        if not csv_file:
            raise HTTPException(
                status_code=404, detail="CSV file not found in folder"
            )

        # CSV 파일 읽기
        csv_path = full_path / csv_file
        csv_content = csv_path.read_text(encoding="utf-8")

        # WAV 파일이 있으면 base64로 인코딩
        audio_base64 = None
        if wav_file:
            wav_path = full_path / wav_file
            audio_bytes = wav_path.read_bytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # GT 파일 찾기 및 변환
        gt_csv_content = None
        gt_file_name = None
        try:
            data_root = get_data_root()
            gt_npy_path = find_gt_file(csv_file, data_root)
            
            if gt_npy_path:
                print(f"[API/epoch] Found GT file: {gt_npy_path}")
                bs_names = get_blendshape_names(csv_filename=csv_file)
                gt_csv_content = npy_to_csv(gt_npy_path, bs_names)
                if gt_csv_content:
                    gt_file_name = gt_npy_path.name
                    print(f"[API/epoch] Converted GT to CSV: {gt_file_name}")
            
            # NPY 없으면 같은 폴더의 *_gt.csv 사용 (train inference 등)
            if not gt_csv_content:
                csv_base = csv_file.rsplit('.', 1)[0]
                candidates = []
                if csv_base.endswith('_pred'):
                    candidates.append(csv_base[:-5] + '_gt.csv')  # _pred -> _gt
                else:
                    candidates.append(csv_base + '_gt.csv')
                if csv_base.endswith('_input_blendshape'):
                    candidates.append(csv_base.replace('_input_blendshape', '') + '_gt.csv')
                for gt_csv_name in candidates:
                    gt_csv_path = full_path / gt_csv_name
                    if gt_csv_path.exists():
                        gt_csv_content = gt_csv_path.read_text(encoding="utf-8")
                        gt_file_name = gt_csv_name
                        print(f"[API/epoch] Using GT CSV from folder: {gt_file_name}")
                        break
            # inference_debug이고 아직 GT 없으면: (1) 원본 파일명 sidecar (2) WAV 매칭
            if not gt_csv_content and normalized_path == "inference_debug":
                tag = csv_base.replace("_input_blendshape", "") if csv_base.endswith("_input_blendshape") else csv_base
                match = _gt_from_source_filename(Path(full_path), tag, data_root)
                if not match:
                    input_wav_path = full_path / f"{tag}_input.wav"
                    match = find_dataset_audio_match(Path(input_wav_path), data_root)
                if match:
                    dataset_prefix, sample_name = match
                    csv_filename = f"{dataset_prefix}_{sample_name}.csv"
                    gt_npy_path = find_gt_file(csv_filename, data_root)
                    if gt_npy_path:
                        bs_names = get_blendshape_names(csv_filename=csv_filename)
                        gt_csv_content = npy_to_csv(gt_npy_path, bs_names)
                        if gt_csv_content:
                            gt_file_name = gt_npy_path.name
                            print(f"[API/epoch] GT from dataset match: {gt_file_name}")
        except Exception as e:
            print(f"[API/epoch] Warning: Could not load GT file: {e}")

        # Epoch 번호 추출 (예: "epoch_001" -> 1)
        epoch_num = None
        try:
            match = re.search(r'epoch_(\d+)', normalized_path)
            if match:
                epoch_num = int(match.group(1))
        except Exception as e:
            print(f"[API/epoch] Warning: Could not extract epoch number: {e}")

        # LVE/LVD 메트릭 읽기
        lve = None
        lvd = None
        val_lve = None
        val_lvd = None
        
        if epoch_num is not None:
            try:
                results_path = get_results_path(run=request.run)
                metrics = get_epoch_metrics_from_tensorboard(epoch_num, results_path)
                
                if metrics:
                    lve = metrics.get('test/LVE')
                    lvd = metrics.get('test/LVD')
                    val_lve = metrics.get('val/LVE')
                    val_lvd = metrics.get('val/LVD')
                    
                    if lve is not None or lvd is not None:
                        print(f"[API/epoch] Loaded metrics for epoch {epoch_num}: "
                              f"LVE={lve}, LVD={lvd}, Val LVE={val_lve}, Val LVD={val_lvd}")
            except Exception as e:
                print(f"[API/epoch] Warning: Could not load LVE/LVD metrics: {e}")

        return EpochResponse(
            success=True,
            csvContent=csv_content,
            audioBase64=audio_base64,
            audioFileName=wav_file,
            csvFileName=csv_file,
            folderName=normalized_path,
            gtCsvContent=gt_csv_content,
            gtFileName=gt_file_name,
            lve=lve,
            lvd=lvd,
            valLve=val_lve,
            valLvd=val_lvd,
        )

    except HTTPException:
        raise
    except Exception as error:
        print(f"[API/epoch] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


def _run_local_inference_sync(audio_path: str, checkpoint_path: str, data_root: str,
                              dataset_for_blendshape: str, identity_idx: int,
                              template_idx: int, output_dir: str,
                              logger=None, compare_vanilla: bool = False):
    """동기 로컬 추론 (스레드에서 실행). 반환: (blendshape_ndarray, blendshape_names) 또는 예외."""
    import numpy as np
    from models.unitalker.infer_audio_to_csv import infer_audio_to_blendshape
    from models.unitalker.dataset.dataset_config import dataset_config

    if logger:
        logger.section("LOCAL INFERENCE SYNC")
        logger.summary("in: audio_path(str), checkpoint_path(str), data_root(str), dataset(str), identity_idx(int), template_idx(int)")

    res = infer_audio_to_blendshape(
        audio_path=audio_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        data_root=data_root,
        dataset_for_blendshape=dataset_for_blendshape,
        identity_idx=identity_idx,
        template_idx=template_idx,
        fps=30,
        logger=logger,
    )
    bs_arr = res["blendshape"]
    run_id = os.environ.get("_CURRENT_INFER_RUN_ID", "")
    from server.run_config import resolve_output_kind
    resolved_kind = resolve_output_kind(
        run_id, checkpoint_path, dataset_for_blendshape, logger=logger)
    bs_arr, names, output_kind = prepare_service_blendshape_output(
        bs_arr,
        dataset_for_blendshape=dataset_for_blendshape,
        data_root=data_root,
        run_id=run_id,
        output_kind=resolved_kind,
    )

    if logger:
        logger.section("DEBUG FILE SAVE")
        logger.summary(f"out: blendshape ndarray(float32 {bs_arr.shape}), names list(str {len(names)}개)")

    # ── 디버그 파일 저장: output_dir(= results/inference_debug/) ──
    try:
        # audio_path 자체가 이미 output_dir에 {tag}_input.wav 로 저장되어 있음
        audio_stem = os.path.splitext(os.path.basename(audio_path))[0]
        # audio_stem 예: "20260209_143000_speaker0_input" → tag 추출
        tag = audio_stem.replace("_input", "")

        # 1) 추론 결과 npy 저장
        pred_dst = os.path.join(output_dir, f"{tag}_pred.npy")
        np.save(pred_dst, bs_arr)
        print(f"[debug] 추론결과 저장: {pred_dst}  shape={bs_arr.shape}")
        if logger:
            logger.summary(f"out: pred.npy, input_blendshape.csv")

        # FaceTool 평가는 로컬 GT NPY 사용 — inference_debug _gt.csv 자동 생성하지 않음
        # (docs/FACETOOL_EVAL_BACKEND_SCOPE.md)
        print(f"[debug] 저장 완료: {output_dir}")
    except Exception as e:
        print(f"[debug] 디버그 파일 저장 실패 (추론 자체에는 영향 없음): {e}")
        if logger:
            logger.error(f"디버그 파일 저장 실패: {e}")

    # ── Vanilla 비교 추론 ──
    if compare_vanilla and logger:
        try:
            from models.unitalker.infer_vanilla import infer_vanilla_blendshape, compare_ours_vs_vanilla
            logger.separator()
            logger.log("=" * 60)
            logger.log("  VANILLA 비교 추론 시작")
            logger.log("=" * 60)
            vanilla_np = infer_vanilla_blendshape(
                audio_path=audio_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                data_root=data_root,
                dataset_name=dataset_for_blendshape,
                identity_idx=identity_idx,
                fps=30,
                logger=logger,
            )
            if vanilla_np is not None:
                compare_ours_vs_vanilla(bs_arr, vanilla_np, logger)
            else:
                logger.warning("Vanilla 추론 결과가 없어 비교를 수행할 수 없습니다")
        except Exception as e:
            logger.error(f"Vanilla 비교 추론 실패: {e}")
            import traceback
            logger.log(traceback.format_exc())

    return bs_arr, names, output_kind


@app.post("/api/inference", tags=["Inference"])
async def inference(request: InferenceRequest):
    """
    오디오 → Blendshape 추론. 출력은 Blendshape만 (FLAME/PCA 미포함).
    
    - **USE_LOCAL_INFERENCE=1** 이면 외부 서버 없이 로컬에서 추론 (run 필수, checkpoint 기본 best.pt).
    - 아니면 NEW_API_URL 추론 서버로 프록시.
    """
    use_local = os.getenv("USE_LOCAL_INFERENCE", "").strip().lower() in ("1", "true", "yes")
    print(
        f"[API/inference] POST received run={request.run!r} checkpoint={request.checkpoint!r} "
        f"speaker={request.speaker} local={use_local} audio_len={len(request.audio_base64 or '')}"
    )
    try:
        if not request.audio_base64:
            raise HTTPException(
                status_code=400,
                detail="audio_base64 is required"
            )

        if use_local:
            run = request.run or ""
            if not run:
                raise HTTPException(status_code=400, detail="run is required when USE_LOCAL_INFERENCE=1")
            run_id = _resolve_run_to_folder(run)
            engine = (request.engine or "pytorch").strip().lower()
            if engine not in ("pytorch", "onnx"):
                raise HTTPException(status_code=400, detail="engine must be 'pytorch' or 'onnx'")

            from server.run_config import get_run_metadata, resolve_inference_config
            from server.service_manifest import resolve_run_service_meta

            svc = resolve_run_service_meta(
                run_id, get_run_metadata(run_id), root=_get_project_root()
            )
            run_folder = (
                _get_pretrained_dir()
                if run_id == PRETRAINED_RUN_ID
                else _get_project_root() / "models" / "unitalker" / run_id
            )

            default_ckpt = (
                OFFICIAL_PRETRAINED_CKPT if run_id == PRETRAINED_RUN_ID else "best.pt"
            )
            checkpoint_path = _resolve_checkpoint_path(
                run_id, request.checkpoint or default_ckpt
            )
            # pytorch 경로는 ckpt 필수; onnx는 .onnx만 있으면 됨
            if engine == "pytorch" and not checkpoint_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Checkpoint not found: {run_id}/{checkpoint_path.name}",
                )
            data_root = str(get_data_root())
            identity_idx = request.speaker if request.speaker is not None else 0

            debug_base = checkpoint_path.parent if checkpoint_path.is_file() else run_folder
            debug_dir = str(debug_base / "inference_debug")
            os.makedirs(debug_dir, exist_ok=True)

            from datetime import datetime
            ts = datetime.now().strftime("%m%d_%H%M")
            base = f"{ts}_s{identity_idx}"
            src_name = (getattr(request, "source_audio_filename", None) or "").strip()
            if src_name:
                raw = src_name.replace(".wav", "").replace(".WAV", "").strip()
                stem = re.sub(r"^(?:speaker\d+_|s\d+_)", "", raw).strip()
                stem = re.sub(r"[^\w\-.]", "_", stem)
                wav_name = f"{base}_{stem}" if stem else base
            else:
                wav_name = base

            from server.inference_logger import InferenceLogger
            logger = InferenceLogger(log_dir=debug_dir, tag=wav_name)

            audio_bytes_raw = base64.b64decode(request.audio_base64)
            logger.section("REQUEST")
            logger.summary(
                "in: run(str), checkpoint(str), speaker(int), engine(str), audio_base64(str)"
            )
            logger.kv("engine", engine)

            # dataset/template: manifest → PT → CSV → state_dict
            if checkpoint_path.is_file():
                dataset_for_blendshape, template_idx = resolve_inference_config(
                    run_id, str(checkpoint_path), logger=logger
                )
            else:
                dataset_for_blendshape = svc.get("dataset_for_blendshape") or "D8"
                template_idx = int(svc.get("template_idx") or 0)
            if run_id == PRETRAINED_RUN_ID:
                dataset_for_blendshape = "D3_3DETF"
                template_idx = 0

            print(
                f"[API/inference] run_id={run_id} engine={engine} speaker={identity_idx} "
                f"-> dataset={dataset_for_blendshape}, template_idx={template_idx}"
            )

            os.environ["_CURRENT_INFER_RUN_ID"] = run_id
            audio_path = os.path.join(debug_dir, f"{wav_name}_input.wav")
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes_raw)
            except Exception as e:
                logger.error(f"Invalid audio_base64: {e}")
                logger.finish()
                raise HTTPException(status_code=400, detail=f"Invalid audio_base64: {e}")

            if getattr(request, "source_audio_filename", None) and str(request.source_audio_filename).strip():
                try:
                    source_path = os.path.join(debug_dir, f"{wav_name}_input_source.txt")
                    with open(source_path, "w", encoding="utf-8") as f:
                        f.write(request.source_audio_filename.strip())
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not save source_audio_filename: {e}")

            try:
                loop = asyncio.get_event_loop()
                executor = getattr(app.state, "_inference_executor", None)
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=1)
                    app.state._inference_executor = executor
                import functools

                if engine == "onnx":
                    from server.onnx_inference import infer_onnx_blendshape, resolve_onnx_path

                    try:
                        onnx_path = resolve_onnx_path(
                            run_folder, svc.get("onnx_file")
                        )
                    except FileNotFoundError as e:
                        raise HTTPException(status_code=404, detail=str(e))

                    def _onnx_job():
                        arr = infer_onnx_blendshape(
                            audio_path=audio_path,
                            onnx_path=str(onnx_path),
                            data_root=data_root,
                            dataset_for_blendshape=dataset_for_blendshape,
                            template_idx=template_idx,
                            style_idx=identity_idx,
                            fps=30.0,
                            logger=logger,
                        )
                        from server.run_config import resolve_output_kind
                        kind = resolve_output_kind(
                            run_id,
                            str(checkpoint_path) if checkpoint_path.is_file() else "",
                            dataset_for_blendshape,
                            logger=logger,
                        )
                        if not kind:
                            kind = svc.get("output_kind") or "metahuman_260"
                        return prepare_service_blendshape_output(
                            arr,
                            dataset_for_blendshape=dataset_for_blendshape,
                            data_root=data_root,
                            run_id=run_id,
                            output_kind=kind,
                        )

                    bs_arr, blendshape_names, output_kind = await loop.run_in_executor(
                        executor, _onnx_job
                    )
                else:
                    bs_arr, blendshape_names, output_kind = await loop.run_in_executor(
                        executor,
                        functools.partial(
                            _run_local_inference_sync,
                            audio_path,
                            str(checkpoint_path),
                            data_root,
                            dataset_for_blendshape,
                            identity_idx,
                            template_idx,
                            debug_dir,
                            logger=logger,
                            compare_vanilla=bool(request.compare_vanilla),
                        ),
                    )
            except HTTPException:
                logger.finish()
                raise
            except Exception as e:
                print(f"[API/inference] Local inference error: {e}")
                logger.error(f"Local inference error: {e}")
                logger.finish()
                raise HTTPException(status_code=500, detail=str(e))

            T = bs_arr.shape[0]
            # 계약 검증
            if len(blendshape_names) != bs_arr.shape[1]:
                logger.error("contract mismatch: names vs channels")
                logger.finish()
                raise HTTPException(
                    status_code=500,
                    detail="Inference contract mismatch: blendshapes length != channels",
                )

            logger.section("RESPONSE")
            logger.summary(
                f"out: frames={T} channels={len(blendshape_names)} engine={engine}"
            )
            logger.finish()

            response = {
                "success": True,
                "output_kind": output_kind,
                "channels": len(blendshape_names),
                "blendshapes": blendshape_names,
                "data": bs_arr.tolist(),
                "fps": 30,
                "frames": T,
                "engine": engine,
            }
            if output_kind == "arkit_51":
                response["blendshapes_glb"] = [
                    arkit_full_name_to_glb_morph(n) for n in blendshape_names
                ]
            return response
        # 프록시 경로
        inference_server_url = os.getenv(
            "NEW_API_URL",
            "http://172.30.1.11:8000"
        )
        start_time = time.time()
        payload = {
            "audio_base64": request.audio_base64,
            "enable_eye_blink": request.enable_eye_blink is not False,
        }
        if request.run is not None:
            payload["run"] = request.run
        if request.checkpoint is not None:
            payload["checkpoint"] = request.checkpoint
        if request.speaker is not None:
            payload["speaker"] = request.speaker

        print(
            f"[API/inference] Proxying (run={request.run}, checkpoint={request.checkpoint}, speaker={request.speaker})"
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{inference_server_url}/inference",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        end_time = time.time()
        print(f"[API/inference] Response in {(end_time - start_time):.2f}s")

        if response.status_code != 200:
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text or f"HTTP {response.status_code} Error"}
            print(f"[API/inference] Upstream error {response.status_code}: {error_data}")
            detail = error_data.get("message", error_data.get("detail", str(error_data))) if isinstance(error_data, dict) else str(error_data)
            raise HTTPException(status_code=response.status_code, detail=detail)

        return response.json()

    except HTTPException:
        raise
    except httpx.TimeoutException:
        inference_url = os.getenv("NEW_API_URL", "http://172.30.1.11:8000")
        print(f"[API/inference] Error: Request timeout to {inference_url}")
        raise HTTPException(status_code=504, detail="Request timeout. NEW_API_URL 추론 서버 응답 지연.")
    except httpx.ConnectError as e:
        print(f"[API/inference] Error: Cannot connect to inference server: {e}")
        raise HTTPException(
            status_code=503,
            detail="추론 서버에 연결할 수 없습니다. NEW_API_URL 확인 및 추론 서버 실행 여부 확인."
        )
    except Exception as error:
        print(f"[API/inference] Error: {error}")
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn

    # 서버 실행
    # host="0.0.0.0"으로 설정하여 다른 PC에서 접근 가능하도록
    uvicorn.run(app, host="0.0.0.0", port=8000)
