"""
Run/체크포인트별 추론 설정을 CSV 또는 PT 메타데이터에서 로드.
하드코딩 없이 dataset_for_blendshape, template_idx를 결정.
"""
import os
import csv
from pathlib import Path
from typing import Optional, Tuple

# 환경 변수: RUN_CONFIG_CSV 가 없으면 models/unitalker/run_config.csv 사용
def _get_run_config_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    env_path = os.getenv("RUN_CONFIG_CSV", "").strip()
    if env_path:
        p = Path(env_path.replace("~", os.getenv("HOME", str(Path.home()))))
        if p.is_absolute():
            return p
        return root / env_path
    return root / "models" / "unitalker" / "run_config.csv"


# run_id -> { "dataset_for_blendshape": str, "template_idx": int }
# alias -> run_id (별칭이 있으면)
_run_config_cache: Optional[dict] = None
_run_alias_to_id: Optional[dict] = None


def load_run_config_csv() -> Tuple[dict, dict]:
    """
    CSV 로드: run_id -> config, alias -> run_id.
    CSV 컬럼: run_id, dataset_for_blendshape, template_idx(선택), alias(선택, 쉼표 구분 가능)
    """
    global _run_config_cache, _run_alias_to_id
    if _run_config_cache is not None:
        return _run_config_cache, _run_alias_to_id or {}

    path = _get_run_config_path()
    by_run_id = {}
    alias_to_id = {}

    if not path.exists():
        _run_config_cache = by_run_id
        _run_alias_to_id = alias_to_id
        return by_run_id, alias_to_id

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                continue
            dataset = (row.get("dataset_for_blendshape") or "").strip()
            try:
                template_idx = int(row.get("template_idx") or "0")
            except ValueError:
                template_idx = 0
            output_kind = (row.get("output_kind") or "").strip() or None
            is_default_raw = (row.get("is_default") or "").strip().lower()
            is_default = is_default_raw in ("1", "true", "yes")
            by_run_id[run_id] = {
                "dataset_for_blendshape": dataset or None,
                "template_idx": template_idx,
                "label": (row.get("label") or "").strip() or None,
                "strategy": (row.get("strategy") or "").strip() or None,
                "output_kind": output_kind,
                "is_default": is_default,
            }
            aliases = (row.get("alias") or "").strip()
            for a in aliases.split(","):
                a = a.strip()
                if a:
                    alias_to_id[a] = run_id

    _run_config_cache = by_run_id
    _run_alias_to_id = alias_to_id
    return by_run_id, alias_to_id


def get_run_metadata(run_id: str) -> dict:
    """run_config.csv에서 label/strategy 등 메타데이터 반환."""
    by_run_id, _ = load_run_config_csv()
    return by_run_id.get(run_id, {})


def resolve_run_to_folder(run: Optional[str], available_run_ids: list) -> str:
    """
    run(폴더명 또는 별칭)을 실제 폴더명(run_id)으로 변환.
    available_run_ids: list_available_runs()의 id 목록.
    """
    if not run:
        return available_run_ids[0] if available_run_ids else ""
    _, alias_to_id = load_run_config_csv()
    # CSV 별칭
    if run in alias_to_id:
        run_id = alias_to_id[run]
        if run_id in available_run_ids:
            return run_id
        return run_id  # 폴더는 없을 수 있음, 호출 쪽에서 404 처리
    # 그대로 폴더명으로 사용
    if run in available_run_ids:
        return run
    return run


def get_inference_config_from_pt(checkpoint_path: str, logger=None) -> Optional[dict]:
    """
    PT 파일 내 inference_config 메타데이터 반환.
    없으면 None.
    """
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "inference_config" in ckpt:
            cfg = ckpt["inference_config"]
            if isinstance(cfg, dict) and cfg.get("dataset_for_blendshape"):
                result = {
                    "dataset_for_blendshape": cfg["dataset_for_blendshape"],
                    "template_idx": int(cfg.get("template_idx", 0)),
                    "output_kind": (cfg.get("output_kind") or "").strip() or None,
                }
                if logger:
                    logger.log("PT inference_config 메타데이터 발견")
                    logger.kv("dataset_for_blendshape", result["dataset_for_blendshape"])
                    logger.kv("template_idx", result["template_idx"])
                return result
        if logger:
            logger.log("PT inference_config 메타데이터 없음")
    except Exception as e:
        if logger:
            logger.warning(f"PT 메타데이터 읽기 실패: {e}")
    return None


def infer_inference_config_from_state_dict(state_dict: dict, logger=None) -> dict:
    """
    state_dict만으로 추론 설정 추정.
    - identity_num > 100 이고 3DETF head 있음 -> pretrained -> D3_3DETF, template 0
    - 3DETF head 없고 inhouse만 있음 -> 파인튜닝 -> LipReading (inhouse), template 0
    - 그 외 -> D3_3DETF, template 0
    """
    identity_key = "decoder.learnable_style_emb.weight"
    identity_num = 0
    if identity_key in state_dict:
        identity_num = state_dict[identity_key].shape[0]
    has_3detf = any("out_head_dict.3DETF_blendshape_weight" in k for k in state_dict)
    has_inhouse = any("out_head_dict.inhouse_blendshape_weight" in k for k in state_dict)
    has_arkit52 = any("out_head_dict.arkit_blendshape_52" in k for k in state_dict)
    has_metahuman = any("out_head_dict.metahuman_control" in k for k in state_dict)
    metahuman_dim = None
    for k, v in state_dict.items():
        if k.endswith("out_head_dict.metahuman_control.weight"):
            metahuman_dim = int(v.shape[0])
            break

    if logger:
        logger.kv("identity_num", identity_num)
        logger.kv("has_3detf_head", has_3detf)
        logger.kv("has_inhouse_head", has_inhouse)
        logger.kv("has_arkit52_head", has_arkit52)
        logger.kv("has_metahuman_head", has_metahuman)
        if metahuman_dim is not None:
            logger.kv("metahuman_head_dim", metahuman_dim)

    if has_metahuman and not (has_3detf or has_inhouse or has_arkit52):
        result = {
            "dataset_for_blendshape": "D8",
            "template_idx": 0,
            "output_kind": "metahuman_260",
        }
        if logger:
            logger.log(f"판정: MetaHuman only (dim={metahuman_dim})")
    elif has_arkit52:
        result = {
            "dataset_for_blendshape": "D3",
            "template_idx": 0,
            "output_kind": "arkit_51",
        }
        if logger:
            logger.log("판정: ARKit 52 head 존재 (D3 3DETF blendshape 사용)")
    elif identity_num > 100 and has_3detf:
        result = {
            "dataset_for_blendshape": "D3_3DETF",
            "template_idx": 0,
            "output_kind": "arkit_51",
        }
        if logger:
            logger.log(f"판정: pretrained (identity_num={identity_num} > 100, 3DETF head 존재)")
    elif not has_3detf and has_inhouse:
        # 파인튜닝: 3DETF 없고 inhouse만 있으면 inhouse용 데이터셋 사용 (run_config.csv로 세부 지정 권장)
        result = {
            "dataset_for_blendshape": "LipReading",
            "template_idx": 0,
            "output_kind": "arkit_51",
        }
        if logger:
            logger.log("판정: 파인튜닝 (3DETF head 없음, inhouse head만 존재)")
    else:
        result = {
            "dataset_for_blendshape": "D3_3DETF",
            "template_idx": 0,
            "output_kind": "arkit_51",
        }
        if logger:
            logger.log("판정: 기본값 (D3_3DETF, template 0)")

    if logger:
        logger.kv("inferred dataset", result["dataset_for_blendshape"])
        logger.kv("inferred template_idx", result["template_idx"])

    return result


def resolve_inference_config(
    run_id: str,
    checkpoint_path: str,
    logger=None,
) -> Tuple[str, int]:
    """
    (dataset_for_blendshape, template_idx) 반환.
    0) service_manifest.json 1) PT inference_config 2) CSV run_id 3) PT state_dict 추정.
    """
    if logger:
        logger.section("RESOLVE INFERENCE CONFIG")
        logger.kv("run_id", run_id)
        logger.kv("checkpoint_path", checkpoint_path)

    # 0) service_manifest.json
    if logger:
        logger.log("[0/4] service_manifest.json 확인...")
    try:
        from server.service_manifest import load_service_manifest
        mf = load_service_manifest(run_id)
        if mf and mf.get("dataset_for_blendshape"):
            ds = mf["dataset_for_blendshape"]
            ti = int(mf.get("template_idx", 0) or 0)
            if logger:
                logger.log("=> service_manifest 사용")
                logger.kv("dataset_for_blendshape", ds)
                logger.kv("template_idx", ti)
            return ds, ti
    except Exception as e:
        if logger:
            logger.warning(f"service_manifest 읽기 실패: {e}")

    # 1) PT 메타데이터
    if logger:
        logger.log("[1/4] PT inference_config 메타데이터 확인...")
    from_pt = get_inference_config_from_pt(checkpoint_path, logger=logger)
    if from_pt:
        if logger:
            logger.log("=> PT 메타데이터 사용")
        return from_pt["dataset_for_blendshape"], from_pt["template_idx"]

    # 2) CSV
    if logger:
        logger.log("[2/4] run_config.csv 확인...")
        logger.kv("csv_path", str(_get_run_config_path()))
    by_run_id, _ = load_run_config_csv()
    if run_id in by_run_id and by_run_id[run_id].get("dataset_for_blendshape"):
        c = by_run_id[run_id]
        if logger:
            logger.log(f"=> CSV에서 run_id='{run_id}' 설정 발견")
            logger.kv("dataset_for_blendshape", c["dataset_for_blendshape"])
            logger.kv("template_idx", c["template_idx"])
        return c["dataset_for_blendshape"], c["template_idx"]
    else:
        if logger:
            logger.log(f"CSV에 run_id='{run_id}' 없음 (등록된 run_ids: {list(by_run_id.keys())})")

    # 3) PT state_dict 추정
    if logger:
        logger.log("[3/4] PT state_dict에서 모델 구조 추정...")
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            if logger:
                total_keys = len(state)
                logger.kv("state_dict keys", total_keys)
            inferred = infer_inference_config_from_state_dict(state, logger=logger)
            if logger:
                logger.log("=> state_dict 추정 사용")
            return inferred["dataset_for_blendshape"], inferred["template_idx"]
    except Exception as e:
        if logger:
            logger.error(f"state_dict 로드 실패: {e}")

    # 기본: pretrained와 동일 (infer_audio_to_csv가 head 자동 선택)
    if logger:
        logger.warning("모든 방법 실패, 기본값 사용: D3_3DETF, template 0")
    return "D3_3DETF", 0


def infer_output_kind(
    dataset_for_blendshape: str,
    run_id: str = "",
    metahuman_dim: Optional[int] = None,
) -> str:
    """서비스 계약: arkit_51 | metahuman_260."""
    if run_id == "pretrained":
        return "arkit_51"
    if dataset_for_blendshape in ("D3_3DETF", "D3", "D6", "D7", "LipReading"):
        return "arkit_51"
    if dataset_for_blendshape in ("D8", "D8_sanity") or metahuman_dim == 260:
        return "metahuman_260"
    if "control260" in run_id:
        return "metahuman_260"
    return "arkit_51"


def resolve_output_kind(
    run_id: str,
    checkpoint_path: str,
    dataset_for_blendshape: str,
    logger=None,
) -> str:
    """output_kind 반환: manifest -> PT inference_config -> CSV -> dataset/state_dict 추정."""
    try:
        from server.service_manifest import load_service_manifest
        mf = load_service_manifest(run_id)
        if mf and mf.get("output_kind"):
            return mf["output_kind"]
    except Exception:
        pass

    from_pt = get_inference_config_from_pt(checkpoint_path, logger=None)
    if from_pt and from_pt.get("output_kind"):
        return from_pt["output_kind"]

    by_run_id, _ = load_run_config_csv()
    if run_id in by_run_id and by_run_id[run_id].get("output_kind"):
        return by_run_id[run_id]["output_kind"]

    metahuman_dim = None
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            for k, v in state.items():
                if k.endswith("out_head_dict.metahuman_control.weight"):
                    metahuman_dim = int(v.shape[0])
                    break
    except Exception:
        pass

    kind = infer_output_kind(
        dataset_for_blendshape, run_id=run_id, metahuman_dim=metahuman_dim)
    if logger:
        logger.kv("output_kind", kind)
    return kind
