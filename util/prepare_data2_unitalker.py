#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf


def _topic_key(stem: str) -> str:
    return stem.split("__")[0]


def _parse_ratio(ratio_text: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in ratio_text.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"split ratio must have 3 values, got: {ratio_text}")
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"split ratio sum must be positive, got: {ratio_text}")
    return vals[0] / total, vals[1] / total, vals[2] / total


def _parse_chunk_seconds(chunk_text: str) -> List[float]:
    vals = [float(x.strip()) for x in chunk_text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"chunk-seconds must not be empty: {chunk_text}")
    if any(v <= 0 for v in vals):
        raise ValueError(f"chunk-seconds must be > 0: {chunk_text}")
    return vals


def _parse_keywords(text: str) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_name_list(path_text: str) -> List[str]:
    if not path_text:
        return []
    p = Path(path_text)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        raise FileNotFoundError(f"name list file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_groups(stems, seed=42, ratio=(0.8, 0.1, 0.1)):
    groups = {}
    for stem in stems:
        groups.setdefault(_topic_key(stem), []).append(stem)
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    n = len(group_keys)
    n_train = max(1, int(n * ratio[0]))
    n_val = max(1, int(n * ratio[1])) if n >= 3 else 0
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])
    if len(test_keys) == 0:
        # 최소 1개 테스트 보장
        last = group_keys[-1]
        train_keys.discard(last)
        test_keys.add(last)
    split = {"train": [], "val": [], "test": []}
    for k in group_keys:
        target = "train" if k in train_keys else ("val" if k in val_keys else "test")
        split[target].extend(sorted(groups[k]))
    return split


def _make_chunk_intervals(total_sec: float, chunk_seconds: List[float]):
    """예: total=40, chunk=[15,15,10] -> (0,15),(15,30),(30,40)."""
    intervals = []
    start = 0.0
    idx = 0
    while start < total_sec - 1e-6:
        dur = chunk_seconds[idx % len(chunk_seconds)]
        end = min(total_sec, start + dur)
        if end - start >= 1e-3:
            intervals.append((start, end))
        start = end
        idx += 1
    return intervals


def _extract_chunks(
    arr: np.ndarray,
    wav: np.ndarray,
    sr: int,
    fps: int,
    chunk_seconds: List[float] | None,
):
    """arr(T,D), wav(S,)를 동일 시간축으로 분할."""
    if chunk_seconds is None:
        return [(arr, wav)]
    total_sec = min(arr.shape[0] / max(fps, 1), len(wav) / max(sr, 1))
    chunks = []
    for start_sec, end_sec in _make_chunk_intervals(total_sec, chunk_seconds):
        f0 = int(round(start_sec * fps))
        f1 = int(round(end_sec * fps))
        s0 = int(round(start_sec * sr))
        s1 = int(round(end_sec * sr))
        if f1 - f0 < 2 or s1 - s0 < sr * 0.1:
            continue
        chunk_arr = arr[f0:f1]
        chunk_wav = wav[s0:s1]
        if len(chunk_arr) > 0 and len(chunk_wav) > 0:
            chunks.append((chunk_arr, chunk_wav))
    return chunks


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _build_one_dataset(
    src_dir: Path,
    dst_root: Path,
    dirname: str,
    annot_type: str,
    annot_dim: int,
    seed: int,
    split_ratio: Tuple[float, float, float],
    chunk_seconds: List[float] | None,
    keep_keywords: List[str],
    drop_keywords: List[str],
    keep_names: List[str],
    drop_names: List[str],
):
    npy_files = sorted(src_dir.glob("*.npy"))
    wav_files = {p.stem: p for p in src_dir.glob("*.wav")}
    stems = [p.stem for p in npy_files if p.stem in wav_files]
    if not stems:
        raise RuntimeError(f"No paired npy/wav in {src_dir}")
    sample = np.load(src_dir / f"{stems[0]}.npy")
    if sample.ndim != 2:
        raise ValueError(f"Unexpected rank for {src_dir / f'{stems[0]}.npy'}: {sample.shape}")
    raw_dim = int(sample.shape[1])
    if annot_dim <= 0:
        annot_dim = raw_dim
        print(f"[INFO] annot-dim auto-detected: {annot_dim}")
    else:
        annot_dim = min(int(annot_dim), raw_dim)
        print(f"[INFO] annot-dim fixed: {annot_dim} (raw={raw_dim})")

    # 채널 이름 로드 및 선택 인덱스 계산
    col_files = sorted(src_dir.glob("*.columns.txt"))
    feature_names = [f"ctrl_{i}" for i in range(annot_dim)]
    if col_files:
        with open(col_files[0], "r", encoding="utf-8") as f:
            loaded_names = [line.strip() for line in f if line.strip()]
        if len(loaded_names) >= annot_dim:
            feature_names = loaded_names[:annot_dim]
        else:
            feature_names = loaded_names + [f"ctrl_{i}" for i in range(len(loaded_names), annot_dim)]

    keep_lower = [k.lower() for k in keep_keywords]
    drop_lower = [k.lower() for k in drop_keywords]
    keep_name_set = {n.lower() for n in keep_names}
    drop_name_set = {n.lower() for n in drop_names}
    selected_idx = list(range(annot_dim))
    if keep_name_set:
        selected_idx = [i for i in selected_idx if feature_names[i].lower() in keep_name_set]
    if drop_name_set:
        selected_idx = [i for i in selected_idx if feature_names[i].lower() not in drop_name_set]
    if keep_lower:
        selected_idx = [
            i for i in selected_idx
            if any(k in feature_names[i].lower() for k in keep_lower)
        ]
    if drop_lower:
        selected_idx = [
            i for i in selected_idx
            if not any(k in feature_names[i].lower() for k in drop_lower)
        ]
    if not selected_idx:
        raise ValueError("No channels selected. Adjust keep/drop keywords.")
    if len(selected_idx) != annot_dim:
        print(
            f"[INFO] channel filtering: {annot_dim} -> {len(selected_idx)} "
            f"(keep_keywords={keep_keywords or 'ALL'}, drop_keywords={drop_keywords or 'NONE'}, "
            f"keep_names={len(keep_names)}, drop_names={len(drop_names)})"
        )
    selected_names = [feature_names[i] for i in selected_idx]
    selected_dim = len(selected_idx)

    split = _split_groups(stems, seed=seed, ratio=split_ratio)
    dst = dst_root / dirname
    for s in ("train", "val", "test"):
        _ensure_dir(dst / s)

    id_list = ["speaker0"]
    id_template = np.zeros((1, selected_dim), dtype=np.float32)
    np.save(dst / "id_template.npy", id_template)
    # 채널 이름 저장 (서비스 응답에서 컬럼명을 그대로 반환)
    with open(dst / "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(selected_names))

    for split_name, split_stems in split.items():
        items = []
        for stem in split_stems:
            src_npy = src_dir / f"{stem}.npy"
            src_wav = src_dir / f"{stem}.wav"
            arr = np.load(src_npy)
            if arr.ndim != 2:
                raise ValueError(f"Unexpected rank for {src_npy}: {arr.shape}")
            if arr.shape[1] < annot_dim:
                raise ValueError(f"Too few dims for {src_npy}: {arr.shape}, expected >= {annot_dim}")
            if arr.shape[1] > annot_dim:
                # 요청사항(정제 최소화)에 맞춰 앞쪽 채널만 사용
                arr = arr[:, :annot_dim]
            if len(selected_idx) != annot_dim:
                arr = arr[:, selected_idx]

            y, sr = librosa.load(src_wav, sr=16000)
            dur = max(len(y) / 16000.0, 1e-6)
            fps = int(round(arr.shape[0] / dur))
            if fps < 20 or fps > 60:
                fps = 30

            chunk_pairs = _extract_chunks(arr=arr, wav=y, sr=16000, fps=fps, chunk_seconds=chunk_seconds)
            for ci, (chunk_arr, chunk_wav) in enumerate(chunk_pairs):
                suffix = f"__c{ci:03d}" if chunk_seconds is not None else ""
                stem_out = f"{stem}{suffix}"
                rel_npy = f"{split_name}/{stem_out}.npy"
                rel_wav = f"{split_name}/{stem_out}.wav"
                np.save(dst / rel_npy, chunk_arr.astype(np.float32))
                sf.write(dst / rel_wav, chunk_wav.astype(np.float32), 16000)

                items.append(
                    {
                        "annot_type": annot_type,
                        "annot_path": rel_npy,
                        "audio_path": rel_wav,
                        "id": 0,
                        "fps": fps,
                        "dataset": dirname,
                    }
                )

        payload = {
            "info": {
                "id_list": id_list,
                "data_source": "DATA2",
                "total_duration": float(
                    sum(np.load(dst / x["annot_path"]).shape[0] / x["fps"] for x in items)
                ),
            },
            "data": items,
        }
        with open(dst / f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {dirname}: groups train/val/test={len(split['train'])}/{len(split['val'])}/{len(split['test'])}, "
        f"chunking={'on' if chunk_seconds is not None else 'off'}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data2-root", type=str, default="/home/motion/omotion/trainFramework/DATA2")
    parser.add_argument(
        "--src-dir",
        type=str,
        default="",
        help="NPY/WAV 소스 폴더 (지정 시 --dataset-name 필수). 예: AA2_Learning_Dataset/arkit",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="D11_AA2_ARKit52",
        help="unitalker_data 하위 폴더명 (src-dir 사용 시)",
    )
    parser.add_argument(
        "--annot-type",
        type=str,
        default="metahuman_control",
        help="json annot_type 값 (예: metahuman_control, arkit_blendshape_52)",
    )
    parser.add_argument(
        "--annot-dim",
        type=int,
        default=-1,
        help="출력 채널 수. -1이면 첫 npy에서 자동 감지. 입력 npy 채널이 더 크면 앞쪽부터 잘라 사용",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="8,1,1",
        help="train,val,test 비율 (예: 8,1,1)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=str,
        default="15,15,10",
        help="긴 샘플 시간 분할(초). 쉼표 패턴 반복. 빈값/none이면 분할 안함",
    )
    parser.add_argument("--output-root", type=str, default="/home/motion/omotion/trainFramework/data/unitalker_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-keywords",
        type=str,
        default="",
        help="채널 이름 포함 키워드(쉼표). 지정 시 해당 키워드 포함 채널만 유지",
    )
    parser.add_argument(
        "--drop-keywords",
        type=str,
        default="",
        help="채널 이름 제외 키워드(쉼표). 지정 시 해당 키워드 포함 채널 제거",
    )
    parser.add_argument(
        "--keep-names-file",
        type=str,
        default="",
        help="정확 채널명 keep 리스트 파일(줄바꿈 구분)",
    )
    parser.add_argument(
        "--drop-names-file",
        type=str,
        default="",
        help="정확 채널명 drop 리스트 파일(줄바꿈 구분)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root)
    _ensure_dir(out_root)

    if args.src_dir:
        src_dir = Path(args.src_dir)
        if not src_dir.is_absolute():
            src_dir = Path(__file__).resolve().parent.parent / src_dir
        ratio = _parse_ratio(args.split_ratio)
        keep_keywords = _parse_keywords(args.keep_keywords)
        drop_keywords = _parse_keywords(args.drop_keywords)
        keep_names = _load_name_list(args.keep_names_file)
        drop_names = _load_name_list(args.drop_names_file)
        chunk_seconds = None
        if str(args.chunk_seconds).strip().lower() not in ("", "none", "null", "off"):
            chunk_seconds = _parse_chunk_seconds(args.chunk_seconds)
        _build_one_dataset(
            src_dir=src_dir,
            dst_root=out_root,
            dirname=args.dataset_name,
            annot_type=args.annot_type,
            annot_dim=args.annot_dim,
            seed=args.seed,
            split_ratio=ratio,
            chunk_seconds=chunk_seconds,
            keep_keywords=keep_keywords,
            drop_keywords=drop_keywords,
            keep_names=keep_names,
            drop_names=drop_names,
        )
        return

    parser.error(
        "기본 DATA2/arkit -> D10 빌드는 제거되었습니다. "
        "--src-dir 과 --dataset-name 을 지정하세요 "
        "(예: --src-dir AA2_Learning_Dataset/arkit --dataset-name D11_AA2_ARKit52)"
    )


if __name__ == "__main__":
    main()
