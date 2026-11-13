#!/usr/bin/env python3
"""
데이터셋 분할 스크립트
VOCA 및 Multiface 데이터를 기존 UniTalker의 train/val/test 구조에 맞춰 분배합니다.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
import random


def analyze_existing_split_pattern(existing_dataset_dir: str) -> Dict:
    """기존 데이터셋의 분할 패턴 분석"""
    result = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(existing_dataset_dir, f"{split}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result[split] = data.get('data', [])
    
    return result


def calculate_split_ratio(existing_splits: Dict) -> Tuple[float, float, float]:
    """기존 분할 비율 계산"""
    train_count = len(existing_splits['train'])
    val_count = len(existing_splits['val'])
    test_count = len(existing_splits['test'])
    total = train_count + val_count + test_count
    
    if total == 0:
        # 기본 비율 (약 80/10/10)
        return (0.8, 0.1, 0.1)
    
    return (train_count / total, val_count / total, test_count / total)


def split_by_speaker(samples: List[Dict], split_ratio: Tuple[float, float, float], 
                     speaker_key: str = 'id') -> Dict[str, List[Dict]]:
    """화자별로 균등하게 분할"""
    # 화자별로 샘플 그룹화
    speaker_samples = defaultdict(list)
    for sample in samples:
        speaker_id = sample.get(speaker_key)
        speaker_samples[speaker_id].append(sample)
    
    # 각 화자별로 분할
    train_samples = []
    val_samples = []
    test_samples = []
    
    for speaker_id, speaker_data in speaker_samples.items():
        n_samples = len(speaker_data)
        n_train = int(n_samples * split_ratio[0])
        n_val = int(n_samples * split_ratio[1])
        
        # 랜덤 셔플 (재현 가능하도록 시드 설정)
        random.seed(42)
        shuffled = speaker_data.copy()
        random.shuffle(shuffled)
        
        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:n_train+n_val])
        test_samples.extend(shuffled[n_train+n_val:])
    
    return {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }


def reorganize_dataset_files(source_dir: str, target_dir: str, splits: Dict[str, List[Dict]], 
                            merge_with_existing: bool = False):
    """데이터셋 파일을 train/val/test 폴더로 재구성"""
    # 타겟 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # 기존 데이터 로드 (통합 모드인 경우)
    existing_data = {'train': [], 'val': [], 'test': []}
    if merge_with_existing:
        for split in ['train', 'val', 'test']:
            json_file = os.path.join(target_dir, f"{split}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_data[split] = data.get('data', [])
    
    # 각 split별로 파일 복사 및 경로 업데이트
    for split_name, samples in splits.items():
        split_dir = os.path.join(target_dir, split_name)
        
        for sample in samples:
            # annot_path와 audio_path에서 파일명 추출
            annot_path = sample.get('annot_path', '')
            audio_path = sample.get('audio_path', '')
            
            # 경로에서 파일명만 추출 (split/ 제거)
            annot_filename = os.path.basename(annot_path) if annot_path else None
            audio_filename = os.path.basename(audio_path) if audio_path else None
            
            # 소스에서 찾기 (원본 위치)
            if annot_filename:
                # 소스 디렉토리의 train/val/test 폴더에서 찾기
                found = False
                for src_split in ['train', 'val', 'test']:
                    src_file = os.path.join(source_dir, src_split, annot_filename)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(split_dir, annot_filename)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                        found = True
                        break
                
                if not found:
                    # 전체 소스 디렉토리에서 찾기 (재귀적)
                    for root, dirs, files in os.walk(source_dir):
                        if annot_filename in files:
                            src_file = os.path.join(root, annot_filename)
                            dst_file = os.path.join(split_dir, annot_filename)
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                            found = True
                            break
                        if found:
                            break
                
                if not found:
                    print(f"    Warning: {annot_filename} 파일을 찾을 수 없습니다.")
            
            if audio_filename:
                found = False
                for src_split in ['train', 'val', 'test']:
                    src_file = os.path.join(source_dir, src_split, audio_filename)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(split_dir, audio_filename)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                        found = True
                        break
                
                if not found:
                    for root, dirs, files in os.walk(source_dir):
                        if audio_filename in files:
                            src_file = os.path.join(root, audio_filename)
                            dst_file = os.path.join(split_dir, audio_filename)
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                            found = True
                            break
                        if found:
                            break
                
                if not found:
                    print(f"    Warning: {audio_filename} 파일을 찾을 수 없습니다.")
            
            # 경로 업데이트
            sample['annot_path'] = f"{split_name}/{annot_filename}" if annot_filename else sample.get('annot_path', '')
            sample['audio_path'] = f"{split_name}/{audio_filename}" if audio_filename else sample.get('audio_path', '')
        
        # 기존 데이터와 통합
        if merge_with_existing:
            samples = existing_data[split_name] + samples
            splits[split_name] = samples
    
    # JSON 파일 생성
    all_id_list = set()
    data_source = None
    
    for split_name, samples in splits.items():
        if samples:
            # 모든 샘플에서 id 수집
            for s in samples:
                all_id_list.add(s.get('id', 0))
                if data_source is None:
                    data_source = s.get('data_source', 'unknown')
    
    for split_name, samples in splits.items():
        if samples:
            # info 생성
            info = {
                "id_list": sorted(list(all_id_list)) if all_id_list else [],
                "data_source": data_source or 'unknown',
                "total_duration": 0.0  # TODO: 실제 duration 계산
            }
            
            json_data = {
                "info": info,
                "data": samples
            }
            
            json_path = os.path.join(target_dir, f"{split_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            existing_count = len(existing_data[split_name]) if merge_with_existing else 0
            new_count = len(samples) - existing_count
            print(f"  {split_name}.json 생성 완료: {len(samples)}개 샘플 (기존: {existing_count}, 신규: {new_count})")


def reorganize_voca_dataset(
    source_dir: str,
    target_dir: str,
    reference_dataset: str = None,
    merge_with_existing: bool = False
):
    """VOCA 데이터셋을 기존 구조에 맞춰 재구성"""
    print(f"\nVOCA 데이터셋 재구성: {source_dir} -> {target_dir}")
    if merge_with_existing:
        print("  모드: 기존 데이터셋과 통합")
    else:
        print("  모드: 새로 생성")
    
    # 참조 데이터셋이 있으면 그 분할 비율 사용
    if reference_dataset and os.path.exists(reference_dataset):
        existing_splits = analyze_existing_split_pattern(reference_dataset)
        split_ratio = calculate_split_ratio(existing_splits)
        print(f"  참조 데이터셋 분할 비율: {split_ratio[0]:.1%} / {split_ratio[1]:.1%} / {split_ratio[2]:.1%}")
    else:
        split_ratio = (0.8, 0.1, 0.1)
        print(f"  기본 분할 비율 사용: {split_ratio[0]:.1%} / {split_ratio[1]:.1%} / {split_ratio[2]:.1%}")
    
    # 소스 JSON 파일 로드
    all_samples = []
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(source_dir, f"{split}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_samples.extend(data.get('data', []))
    
    if not all_samples:
        print("  ERROR: 샘플 데이터를 찾을 수 없습니다.")
        return
    
    print(f"  총 샘플 수: {len(all_samples)}")
    
    # 화자별로 재분할
    splits = split_by_speaker(all_samples, split_ratio, speaker_key='id')
    
    print(f"  분할 결과:")
    print(f"    Train: {len(splits['train'])}개 ({len(splits['train'])/len(all_samples)*100:.1f}%)")
    print(f"    Val: {len(splits['val'])}개 ({len(splits['val'])/len(all_samples)*100:.1f}%)")
    print(f"    Test: {len(splits['test'])}개 ({len(splits['test'])/len(all_samples)*100:.1f}%)")
    
    # 파일 재구성
    reorganize_dataset_files(source_dir, target_dir, splits, merge_with_existing)


def reorganize_multiface_dataset(
    source_dir: str,
    target_dir: str,
    reference_dataset: str = None,
    merge_with_existing: bool = False
):
    """Multiface 데이터셋을 기존 구조에 맞춰 재구성"""
    print(f"\nMultiface 데이터셋 재구성: {source_dir} -> {target_dir}")
    if merge_with_existing:
        print("  모드: 기존 데이터셋과 통합")
    else:
        print("  모드: 새로 생성")
    
    # 참조 데이터셋이 있으면 그 분할 비율 사용
    if reference_dataset and os.path.exists(reference_dataset):
        existing_splits = analyze_existing_split_pattern(reference_dataset)
        split_ratio = calculate_split_ratio(existing_splits)
        print(f"  참조 데이터셋 분할 비율: {split_ratio[0]:.1%} / {split_ratio[1]:.1%} / {split_ratio[2]:.1%}")
    else:
        split_ratio = (0.8, 0.1, 0.1)
        print(f"  기본 분할 비율 사용: {split_ratio[0]:.1%} / {split_ratio[1]:.1%} / {split_ratio[2]:.1%}")
    
    # 소스 JSON 파일 로드
    all_samples = []
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(source_dir, f"{split}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_samples.extend(data.get('data', []))
    
    if not all_samples:
        print("  ERROR: 샘플 데이터를 찾을 수 없습니다.")
        return
    
    print(f"  총 샘플 수: {len(all_samples)}")
    
    # 화자별로 재분할
    splits = split_by_speaker(all_samples, split_ratio, speaker_key='id')
    
    print(f"  분할 결과:")
    print(f"    Train: {len(splits['train'])}개 ({len(splits['train'])/len(all_samples)*100:.1f}%)")
    print(f"    Val: {len(splits['val'])}개 ({len(splits['val'])/len(all_samples)*100:.1f}%)")
    print(f"    Test: {len(splits['test'])}개 ({len(splits['test'])/len(all_samples)*100:.1f}%)")
    
    # 파일 재구성
    reorganize_dataset_files(source_dir, target_dir, splits, merge_with_existing)


def main():
    parser = argparse.ArgumentParser(description="데이터셋 분할 및 재구성")
    parser.add_argument("--dataset", type=str, choices=['voca', 'multiface', 'both'],
                       default='both', help="처리할 데이터셋")
    parser.add_argument("--source_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 소스 디렉토리")
    parser.add_argument("--source_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 소스 디렉토리")
    parser.add_argument("--target_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 타겟 디렉토리")
    parser.add_argument("--target_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 타겟 디렉토리")
    parser.add_argument("--reference_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 참조 데이터셋 (분할 비율 참조)")
    parser.add_argument("--reference_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 참조 데이터셋 (분할 비율 참조)")
    parser.add_argument("--merge", action="store_true",
                       help="기존 데이터셋과 통합 (기존 파일 유지)")
    parser.add_argument("--overwrite", action="store_true",
                       help="기존 데이터셋 덮어쓰기 (기본값)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("데이터셋 분할 및 재구성")
    print("=" * 80)
    
    merge_mode = args.merge and not args.overwrite
    
    if args.dataset in ['voca', 'both']:
        if os.path.exists(args.source_voca):
            reorganize_voca_dataset(
                args.source_voca,
                args.target_voca,
                args.reference_voca if os.path.exists(args.reference_voca) else None,
                merge_with_existing=merge_mode
            )
        else:
            print(f"Warning: VOCA 소스 디렉토리를 찾을 수 없습니다: {args.source_voca}")
    
    if args.dataset in ['multiface', 'both']:
        if os.path.exists(args.source_multiface):
            reorganize_multiface_dataset(
                args.source_multiface,
                args.target_multiface,
                args.reference_multiface if os.path.exists(args.reference_multiface) else None,
                merge_with_existing=merge_mode
            )
        else:
            print(f"Warning: Multiface 소스 디렉토리를 찾을 수 없습니다: {args.source_multiface}")
    
    print("\n" + "=" * 80)
    print("작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
