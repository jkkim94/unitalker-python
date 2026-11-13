#!/usr/bin/env python
"""ARKit blendshape → MetaHuman 260ch control 오프라인 변환 (Unreal 불필요).

Epic PA_MetaHuman_ARKit_Mapping (UE 5.8 PoseAsset, ARKitRemap 추출본
util/mappings/arkitremap/pa_mapping.json, MPL-2.0)의 baseline-adjusted 값을 선형 결합:
    MH[t, c] = sum_i ARKit[t, i] * adjusted[pose_i][c]
ABP_MH_LiveLink 규칙으로 LipsTogether{DL,DR,UL,UR} = clamp(MouthClose / JawOpen, 0, 1).

D10 데이터셋: 학습 대상 채널은 매핑되면서 target(D8) GT가 실제로 쓰는 하관 채널로 한정
(loss_channel_mask.npy). 해당 채널 값은 소스별 quantile 매칭으로 target 분포에 맞춤.

생성:
    python util/arkit_to_metahuman.py build-d10 \
        --data-root data/unitalker_data --sources D3,D4,D6
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PA_MAPPING_PATH = PROJECT_ROOT / "util" / "mappings" / "arkitremap" / "pa_mapping.json"

# 3DETF / EmoTalk 52ch 순서 (EmoTalk.npz bs_names, 알파벳순)
ARKIT52_NAMES = [
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
    "tongueOut",
]

# D6/D7 inhouse_blendshape_weight 51ch 컬럼 순서 (알파벳순 아님).
# inhouse_arkit.npz 기저 형상 + pretrained 3DETF/inhouse head 출력 상관으로 식별.
INHOUSE51_NAMES = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeWideLeft", "eyeWideRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "browDownLeft", "browDownRight",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen",
    "mouthClose", "jawLeft", "jawRight", "jawForward",
    "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthRollUpper", "mouthRollLower", "mouthSmileLeft", "mouthSmileRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthFunnel", "mouthLeft", "mouthRight",
    "mouthShrugLower", "mouthShrugUpper", "noseSneerLeft", "noseSneerRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
]

LIPS_TOGETHER_NAMES = [
    "CTRL_expressions_mouthLipsTogetherDL",
    "CTRL_expressions_mouthLipsTogetherDR",
    "CTRL_expressions_mouthLipsTogetherUL",
    "CTRL_expressions_mouthLipsTogetherUR",
]

JAW_IDX = ARKIT52_NAMES.index("jawOpen")
CLOSE_IDX = ARKIT52_NAMES.index("mouthClose")

# UniTalker-B-D0-D7 학습 시 dataset 순서 (style embedding row offset 계산용)
PRETRAINED_DATASET_ORDER = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

SOURCES = {
    "D3": {"dirname": "D3D4_3DETF/D3_HDTF", "layout": "arkit52"},
    "D4": {"dirname": "D3D4_3DETF/D4_RAVDESS", "layout": "arkit52"},
    "D6": {"dirname": "D6_unitalker_Chinese_speech", "layout": "inhouse51"},
    "D7": {"dirname": "D7_unitalker_song", "layout": "inhouse51"},
}


def load_feature_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_mapping_matrix(feature_names: list[str], pa_path: Path = PA_MAPPING_PATH):
    """(52, C) 선형 매핑 행렬과 LipsTogether 채널 인덱스."""
    adjusted = json.loads(pa_path.read_text(encoding="utf-8"))["adjusted"]
    col = {n.lower(): i for i, n in enumerate(feature_names)}
    mat = np.zeros((len(ARKIT52_NAMES), len(feature_names)), dtype=np.float32)
    for pose, curves in adjusted.items():
        name = pose[0].lower() + pose[1:]
        if name not in ARKIT52_NAMES:
            continue
        row = ARKIT52_NAMES.index(name)
        for curve, value in curves.items():
            if not curve.startswith("ctrl_expressions_"):
                continue
            j = col.get(curve)
            if j is not None:
                mat[row, j] = value
    lips_idx = np.array([feature_names.index(n) for n in LIPS_TOGETHER_NAMES], dtype=np.int64)
    return mat, lips_idx


def inhouse51_to_arkit52(x: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], len(ARKIT52_NAMES)), dtype=np.float32)
    for i, name in enumerate(INHOUSE51_NAMES):
        out[:, ARKIT52_NAMES.index(name)] = x[:, i]
    return out


def arkit52_to_metahuman(x52: np.ndarray, mat: np.ndarray, lips_idx: np.ndarray,
                         jaw_eps: float = 0.02) -> np.ndarray:
    x = np.clip(x52.astype(np.float32), 0.0, 1.0)
    y = x @ mat
    jaw = x[:, JAW_IDX]
    close = x[:, CLOSE_IDX]
    # SafeDivide: pseudo-label jaw 노이즈(≈0)에서 비율 폭주 방지
    ratio = np.where(jaw > jaw_eps, close / np.maximum(jaw, jaw_eps), 0.0)
    y[:, lips_idx] = np.clip(ratio, 0.0, 1.0)[:, None]
    return np.clip(y, 0.0, 1.0)


def build_loss_mask(mat: np.ndarray, lips_idx: np.ndarray, feature_names: list[str],
                    scope: str = "target_used", target_std: np.ndarray | None = None,
                    min_target_std: float = 0.02) -> np.ndarray:
    """scope: all_mapped | lower_face | target_used (lower_face ∩ target GT std >= min_target_std)."""
    mask = (np.abs(mat).max(0) > 1e-6).astype(np.float32)
    mask[lips_idx] = 1.0
    if scope in ("lower_face", "target_used"):
        for i, n in enumerate(feature_names):
            low = n.lower()
            if "eye" in low or "brow" in low or "tongue" in low:
                mask[i] = 0.0
    elif scope != "all_mapped":
        raise ValueError(f"unknown mask scope: {scope}")
    if scope == "target_used":
        if target_std is None:
            raise ValueError("target_used scope requires target_std")
        mask[target_std < min_target_std] = 0.0
    return mask


class QuantileCalibrator:
    """채널별 경험적 CDF 매칭 (source → target). mid-rank로 0/1 점질량 처리."""

    def __init__(self, source: np.ndarray, target: np.ndarray, channels: np.ndarray,
                 max_samples: int = 50000, levels: int = 2001):
        rng = np.random.default_rng(0)
        self.channels = np.asarray(channels, dtype=np.int64)
        self.levels = np.linspace(0.0, 1.0, levels)
        self.src_sorted, self.tgt_quant = [], []
        for c in self.channels:
            s = source[:, c]
            if len(s) > max_samples:
                s = s[rng.choice(len(s), max_samples, replace=False)]
            self.src_sorted.append(np.sort(s))
            self.tgt_quant.append(np.quantile(target[:, c], self.levels))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        out = y.copy()
        for k, c in enumerate(self.channels):
            s = self.src_sorted[k]
            lo = np.searchsorted(s, y[:, c], side="left")
            hi = np.searchsorted(s, y[:, c], side="right")
            mid = (lo + hi) * 0.5 / len(s)
            out[:, c] = np.interp(mid, self.levels, self.tgt_quant[k])
        return out


def pretrained_style_offsets() -> dict[str, int]:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.unitalker.dataset.dataset_config import dataset_config
    offsets, acc = {}, 0
    for name in PRETRAINED_DATASET_ORDER:
        offsets[name] = acc
        acc += int(dataset_config[name]["subjects"])
    return offsets


def _even_subset(items: list, limit: int) -> list:
    if limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in idx]


def _load_source_arkit52(src_dir: Path, item: dict, layout: str, max_seconds: float) -> np.ndarray:
    x = np.load(src_dir / item["annot_path"], mmap_mode="r")
    x = np.asarray(x[: int(round(max_seconds * float(item["fps"])))], dtype=np.float32)
    if layout == "inhouse51":
        x = inhouse51_to_arkit52(x)
    return x


def _load_target_frames(target_dir: Path) -> np.ndarray:
    labels = json.loads((target_dir / "train.json").read_text(encoding="utf-8"))
    return np.concatenate([np.load(target_dir / it["annot_path"]).astype(np.float32) for it in labels["data"]])


def _fit_head_init(raw: np.ndarray, calibrated: np.ndarray, mat: np.ndarray, mask: np.ndarray):
    """calibrated ≈ a * raw + b (채널별) → head 초기화용 (52, C) weight, (C,) bias."""
    weight = np.zeros_like(mat)
    bias = np.zeros(mat.shape[1], dtype=np.float32)
    for c in np.where(mask > 0)[0]:
        r, t = raw[:, c], calibrated[:, c]
        if r.std() > 1e-6 and np.abs(mat[:, c]).max() > 1e-6:
            a = float(((r - r.mean()) * (t - t.mean())).mean() / r.var())
            weight[:, c] = mat[:, c] * a
            bias[c] = float(t.mean() - a * r.mean())
        else:
            bias[c] = float(t.mean())
    return weight, bias


def build_d10(args):
    data_root = Path(args.data_root).resolve()
    out_dir = data_root / args.out_dirname
    target_dir = data_root / args.target_dirname
    feature_names = load_feature_names(target_dir / "feature_names.txt")
    mat, lips_idx = build_mapping_matrix(feature_names)
    target = _load_target_frames(target_dir)
    mask = build_loss_mask(mat, lips_idx, feature_names, args.mask_scope,
                           target_std=target.std(0), min_target_std=args.min_target_std)
    offsets = pretrained_style_offsets()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    mask_idx = np.where(mask > 0)[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    id_list, style_idx = [], []
    id_offset, id_count, labels = {}, {}, {}
    for src in sources:
        src_dir = data_root / SOURCES[src]["dirname"]
        labels[src] = {split: json.loads((src_dir / f"{split}.json").read_text(encoding="utf-8"))
                       for split in ("train", "val", "test")}
        src_ids = labels[src]["train"]["info"]["id_list"]
        id_offset[src] = len(id_list)
        id_count[src] = len(src_ids)
        id_list += [f"{src}:{sid}" for sid in src_ids]
        style_idx += [offsets[src] + i for i in range(len(src_ids))]

    calibrators = {}
    raw_pool, cal_pool = [], []
    if args.calibrate == "quantile":
        for src in sources:
            src_dir = data_root / SOURCES[src]["dirname"]
            raw = np.concatenate([
                arkit52_to_metahuman(_load_source_arkit52(src_dir, it, SOURCES[src]["layout"], args.max_seconds),
                                     mat, lips_idx, args.jaw_eps)
                for it in labels[src]["train"]["data"]
            ])
            calibrators[src] = QuantileCalibrator(raw, target, mask_idx)
            sub = raw[:: max(1, len(raw) // 20000)]
            raw_pool.append(sub)
            cal_pool.append(calibrators[src](sub))
            print(f"[calib] {src}: {len(raw)} frames")

    split_limit = {"train": 0, "val": args.eval_clips_per_source, "test": args.eval_clips_per_source}
    stats = {}
    for split, limit in split_limit.items():
        (out_dir / split).mkdir(exist_ok=True)
        entries, total = [], 0.0
        for src in sources:
            src_dir = data_root / SOURCES[src]["dirname"]
            assert len(labels[src][split]["info"]["id_list"]) == id_count[src], src
            for item in _even_subset(labels[src][split]["data"], limit):
                x = _load_source_arkit52(src_dir, item, SOURCES[src]["layout"], args.max_seconds)
                y = arkit52_to_metahuman(x, mat, lips_idx, args.jaw_eps)
                if src in calibrators:
                    y = calibrators[src](y)
                stem = Path(item["annot_path"]).stem
                rel_annot = f"{split}/{src}__{stem}.npy"
                rel_audio = f"{split}/{src}__{stem}.wav"
                np.save(out_dir / rel_annot, y.astype(np.float32))
                audio, _ = librosa.load(str(src_dir / item["audio_path"]), sr=16000, mono=True,
                                        duration=args.max_seconds)
                sf.write(str(out_dir / rel_audio), audio, 16000, subtype="PCM_16")
                entries.append({
                    "annot_type": "metahuman_control",
                    "annot_path": rel_annot,
                    "audio_path": rel_audio,
                    "id": id_offset[src] + int(item["id"]),
                    "fps": item["fps"],
                    "dataset": "D10",
                })
                total += min(len(y) / float(item["fps"]), len(audio) / 16000.0)
        info = {
            "id_list": id_list,
            "pretrained_style_idx": style_idx,
            "data_source": "ARKit->MetaHuman (Epic PA_MetaHuman_ARKit_Mapping): " + ",".join(sources),
            "calibration": f"{args.calibrate} -> {args.target_dirname}",
            "total_duration": total,
        }
        (out_dir / f"{split}.json").write_text(
            json.dumps({"info": info, "data": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
        stats[split] = {"clips": len(entries), "seconds": round(total, 1)}

    if raw_pool:
        head_w, head_b = _fit_head_init(np.concatenate(raw_pool), np.concatenate(cal_pool), mat, mask)
    else:
        head_w, head_b = mat * mask[None], np.zeros(mat.shape[1], dtype=np.float32)

    np.save(out_dir / "id_template.npy", np.zeros((len(id_list), len(feature_names)), dtype=np.float32))
    np.save(out_dir / "loss_channel_mask.npy", mask)
    (out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    np.savez(out_dir / "arkit52_to_metahuman260.npz", matrix=mat, lips_together_idx=lips_idx,
             head_init_weight=head_w, head_init_bias=head_b,
             arkit_names=np.array(ARKIT52_NAMES), feature_names=np.array(feature_names), loss_mask=mask)
    print(json.dumps({"out_dir": str(out_dir), "identities": len(id_list),
                      "mask_channels": [feature_names[i] for i in mask_idx], "splits": stats},
                     indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-d10")
    p.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "unitalker_data"))
    p.add_argument("--out-dirname", default="D10_arkit2mh")
    p.add_argument("--target-dirname", default="D8_metahuman_control",
                   help="feature_names.txt, loss mask, quantile calibration 기준 GT")
    p.add_argument("--sources", default="D3,D4,D6")
    p.add_argument("--eval-clips-per-source", type=int, default=40)
    p.add_argument("--mask-scope", choices=["target_used", "lower_face", "all_mapped"], default="target_used")
    p.add_argument("--min-target-std", type=float, default=0.02)
    p.add_argument("--calibrate", choices=["quantile", "none"], default="quantile")
    p.add_argument("--jaw-eps", type=float, default=0.02)
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="DataItem.truncate와 동일하게 클립 앞부분만 사용")
    args = parser.parse_args()
    if args.cmd == "build-d10":
        build_d10(args)


if __name__ == "__main__":
    main()
