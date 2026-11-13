#!/usr/bin/env python3
"""
데이터셋 통합 및 검증 스크립트
vocaset_new, meshtalk_new의 데이터를 원래 폴더로 복사하고 검증합니다.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
from collections import defaultdict


def verify_json_files(json_path: str, base_dir: str) -> Dict:
    """JSON 파일의 모든 항목이 실제로 존재하는지 검증"""
    result = {
        "valid": True,
        "total_samples": 0,
        "missing_audio": [],
        "missing_annot": [],
        "path_mismatches": [],
        "statistics": {}
    }
    
    if not os.path.exists(json_path):
        result["valid"] = False
        result["error"] = f"JSON 파일이 없습니다: {json_path}"
        return result
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('data', [])
    result["total_samples"] = len(samples)
    
    audio_files = set()
    annot_files = set()
    
    # JSON 파일이 있는 split 디렉토리 확인
    json_dir = os.path.dirname(json_path)
    split_name = os.path.basename(json_dir) if os.path.basename(json_dir) in ['train', 'val', 'test'] else None
    if not split_name:
        # json_path에서 split 추출
        if 'train.json' in json_path:
            split_name = 'train'
        elif 'val.json' in json_path:
            split_name = 'val'
        elif 'test.json' in json_path:
            split_name = 'test'
    
    for sample in samples:
        annot_path = sample.get('annot_path', '')
        audio_path = sample.get('audio_path', '')
        
        # 경로에서 파일명 추출
        annot_filename = os.path.basename(annot_path) if annot_path else None
        audio_filename = os.path.basename(audio_path) if audio_path else None
        
        # 실제 파일 경로 구성
        if annot_path:
            # 경로에 split/ 포함 여부 확인
            if annot_path.startswith('train/') or annot_path.startswith('val/') or annot_path.startswith('test/'):
                # split/ 포함된 경로는 그대로 사용
                actual_annot_path = os.path.join(base_dir, annot_path)
            elif split_name:
                # split 정보가 있으면 해당 split 디렉토리에서 찾기
                actual_annot_path = os.path.join(base_dir, split_name, annot_filename)
            else:
                # split 정보가 없으면 train/val/test에서 찾기
                found = False
                actual_annot_path = None
                for split in ['train', 'val', 'test']:
                    test_path = os.path.join(base_dir, split, annot_filename)
                    if os.path.exists(test_path):
                        actual_annot_path = test_path
                        found = True
                        break
                
                if not found:
                    actual_annot_path = os.path.join(base_dir, annot_filename)
            
            if not os.path.exists(actual_annot_path):
                result["missing_annot"].append({
                    "sample": sample,
                    "expected": annot_path,
                    "actual": actual_annot_path
                })
                result["valid"] = False
            else:
                annot_files.add(annot_filename)
        
        if audio_path:
            if audio_path.startswith('train/') or audio_path.startswith('val/') or audio_path.startswith('test/'):
                actual_audio_path = os.path.join(base_dir, audio_path)
            elif split_name:
                actual_audio_path = os.path.join(base_dir, split_name, audio_filename)
            else:
                found = False
                actual_audio_path = None
                for split in ['train', 'val', 'test']:
                    test_path = os.path.join(base_dir, split, audio_filename)
                    if os.path.exists(test_path):
                        actual_audio_path = test_path
                        found = True
                        break
                
                if not found:
                    actual_audio_path = os.path.join(base_dir, audio_filename)
            
            if not os.path.exists(actual_audio_path):
                result["missing_audio"].append({
                    "sample": sample,
                    "expected": audio_path,
                    "actual": actual_audio_path
                })
                result["valid"] = False
            else:
                audio_files.add(audio_filename)
        
        # 파일명 매칭 확인
        if annot_filename and audio_filename:
            # 기본 이름이 일치하는지 확인 (확장자 제외)
            annot_base = os.path.splitext(annot_filename)[0]
            audio_base = os.path.splitext(audio_filename)[0]
            
            if annot_base != audio_base:
                result["path_mismatches"].append({
                    "annot": annot_filename,
                    "audio": audio_filename,
                    "sample": sample
                })
    
    result["statistics"] = {
        "unique_audio_files": len(audio_files),
        "unique_annot_files": len(annot_files),
        "missing_audio_count": len(result["missing_audio"]),
        "missing_annot_count": len(result["missing_annot"]),
        "path_mismatches_count": len(result["path_mismatches"])
    }
    
    return result


def copy_dataset_files(source_dir: str, target_dir: str, dataset_name: str) -> Dict:
    """데이터셋 파일을 소스에서 타겟으로 복사"""
    print(f"\n[{dataset_name}] 파일 복사: {source_dir} -> {target_dir}")
    
    result = {
        "copied": 0,
        "skipped": 0,
        "errors": [],
        "statistics": {}
    }
    
    # 타겟 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # JSON 파일 복사
    for split in ['train', 'val', 'test']:
        source_json = os.path.join(source_dir, f"{split}.json")
        target_json = os.path.join(target_dir, f"{split}.json")
        
        if os.path.exists(source_json):
            shutil.copy2(source_json, target_json)
            print(f"  ✓ {split}.json 복사 완료")
        else:
            print(f"  ⚠ {split}.json이 소스에 없습니다")
    
    # 각 split별로 파일 복사
    total_files = 0
    copied_files = 0
    
    for split in ['train', 'val', 'test']:
        source_split_dir = os.path.join(source_dir, split)
        target_split_dir = os.path.join(target_dir, split)
        
        if not os.path.exists(source_split_dir):
            continue
        
        # 소스 디렉토리의 모든 파일 확인
        source_files = [f for f in os.listdir(source_split_dir) 
                       if os.path.isfile(os.path.join(source_split_dir, f))]
        
        for filename in source_files:
            total_files += 1
            source_file = os.path.join(source_split_dir, filename)
            target_file = os.path.join(target_split_dir, filename)
            
            try:
                if not os.path.exists(target_file):
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    result["copied"] += 1
                else:
                    result["skipped"] += 1
            except Exception as e:
                result["errors"].append(f"{split}/{filename}: {e}")
    
    result["statistics"] = {
        "total_files": total_files,
        "copied": copied_files,
        "skipped": result["skipped"]
    }
    
    print(f"  파일 복사 완료: {copied_files}개 복사, {result['skipped']}개 건너뜀")
    
    return result


def integrate_and_verify_dataset(
    source_dir: str,
    target_dir: str,
    dataset_name: str
) -> Dict:
    """데이터셋 통합 및 검증"""
    print("=" * 80)
    print(f"{dataset_name} 데이터셋 통합 및 검증")
    print("=" * 80)
    
    result = {
        "dataset": dataset_name,
        "copy_result": {},
        "verification": {},
        "valid": True
    }
    
    # 1. 파일 복사
    if not os.path.exists(source_dir):
        print(f"  ❌ 소스 디렉토리가 없습니다: {source_dir}")
        result["valid"] = False
        return result
    
    copy_result = copy_dataset_files(source_dir, target_dir, dataset_name)
    result["copy_result"] = copy_result
    
    if copy_result["errors"]:
        result["valid"] = False
    
    # 2. JSON 파일 검증
    print(f"\n[{dataset_name}] JSON 파일 검증")
    
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(target_dir, f"{split}.json")
        
        if not os.path.exists(json_path):
            print(f"  ⚠ {split}.json이 없습니다")
            continue
        
        print(f"\n  [{split}.json] 검증 중...")
        verification = verify_json_files(json_path, target_dir)
        result["verification"][split] = verification
        
        print(f"    총 샘플 수: {verification['total_samples']}")
        print(f"    고유 오디오 파일: {verification['statistics']['unique_audio_files']}")
        print(f"    고유 annotation 파일: {verification['statistics']['unique_annot_files']}")
        
        if verification["missing_audio"]:
            print(f"    ❌ 누락된 오디오 파일: {len(verification['missing_audio'])}개")
            for missing in verification["missing_audio"][:5]:
                print(f"      - {missing['expected']}")
            if len(verification["missing_audio"]) > 5:
                print(f"      ... 외 {len(verification['missing_audio']) - 5}개")
            result["valid"] = False
        
        if verification["missing_annot"]:
            print(f"    ❌ 누락된 annotation 파일: {len(verification['missing_annot'])}개")
            for missing in verification["missing_annot"][:5]:
                print(f"      - {missing['expected']}")
            if len(verification["missing_annot"]) > 5:
                print(f"      ... 외 {len(verification['missing_annot']) - 5}개")
            result["valid"] = False
        
        if verification["path_mismatches"]:
            print(f"    ⚠ 파일명 불일치: {len(verification['path_mismatches'])}개")
            for mismatch in verification["path_mismatches"][:5]:
                print(f"      - annot: {mismatch['annot']}, audio: {mismatch['audio']}")
            if len(verification["path_mismatches"]) > 5:
                print(f"      ... 외 {len(verification['path_mismatches']) - 5}개")
        
        # 파일명 매칭 확인
        if verification["path_mismatches"]:
            print(f"    ⚠ 파일명 불일치: {len(verification['path_mismatches'])}개")
            for mismatch in verification["path_mismatches"][:5]:
                print(f"      - annot: {mismatch['annot']}, audio: {mismatch['audio']}")
            if len(verification["path_mismatches"]) > 5:
                print(f"      ... 외 {len(verification['path_mismatches']) - 5}개")
        
        if verification["valid"] and not verification["missing_audio"] and not verification["missing_annot"]:
            if verification["path_mismatches"]:
                print(f"    ⚠ {split}.json 검증 통과 (파일명 불일치 있음)")
            else:
                print(f"    ✅ {split}.json 검증 통과 (모든 파일 존재, 파일명 매칭 완료)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="데이터셋 통합 및 검증")
    parser.add_argument("--source_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 소스 디렉토리 (new)")
    parser.add_argument("--source_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 소스 디렉토리 (new)")
    parser.add_argument("--target_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 타겟 디렉토리 (원래)")
    parser.add_argument("--target_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 타겟 디렉토리 (원래)")
    parser.add_argument("--dataset", type=str, choices=['voca', 'multiface', 'both'],
                       default='both', help="처리할 데이터셋")
    parser.add_argument("--dry_run", action="store_true",
                       help="실제 복사 없이 검증만 수행")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("데이터셋 통합 및 검증")
    if args.dry_run:
        print("  [DRY RUN 모드 - 실제 복사 없음]")
    print("=" * 80)
    
    voca_result = None
    multiface_result = None
    
    if args.dataset in ['voca', 'both']:
        if os.path.exists(args.source_voca):
            if not args.dry_run:
                voca_result = integrate_and_verify_dataset(
                    args.source_voca,
                    args.target_voca,
                    "VOCA"
                )
            else:
                print(f"\n[DRY RUN] VOCA: {args.source_voca} -> {args.target_voca}")
                # 검증만 수행
                for split in ['train', 'val', 'test']:
                    json_path = os.path.join(args.source_voca, f"{split}.json")
                    if os.path.exists(json_path):
                        verification = verify_json_files(json_path, args.source_voca)
                        print(f"  {split}.json: {verification['total_samples']}개 샘플")
        else:
            print(f"Warning: VOCA 소스 디렉토리를 찾을 수 없습니다: {args.source_voca}")
    
    if args.dataset in ['multiface', 'both']:
        if os.path.exists(args.source_multiface):
            if not args.dry_run:
                multiface_result = integrate_and_verify_dataset(
                    args.source_multiface,
                    args.target_multiface,
                    "Multiface"
                )
            else:
                print(f"\n[DRY RUN] Multiface: {args.source_multiface} -> {args.target_multiface}")
                for split in ['train', 'val', 'test']:
                    json_path = os.path.join(args.source_multiface, f"{split}.json")
                    if os.path.exists(json_path):
                        verification = verify_json_files(json_path, args.source_multiface)
                        print(f"  {split}.json: {verification['total_samples']}개 샘플")
        else:
            print(f"Warning: Multiface 소스 디렉토리를 찾을 수 없습니다: {args.source_multiface}")
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)
    
    if voca_result:
        status = "✅ 통합 완료" if voca_result.get("valid") else "❌ 문제 있음"
        print(f"VOCA: {status}")
        if voca_result.get("copy_result"):
            copy_stat = voca_result["copy_result"]["statistics"]
            print(f"  복사된 파일: {copy_stat.get('copied', 0)}개")
    
    if multiface_result:
        status = "✅ 통합 완료" if multiface_result.get("valid") else "❌ 문제 있음"
        print(f"Multiface: {status}")
        if multiface_result.get("copy_result"):
            copy_stat = multiface_result["copy_result"]["statistics"]
            print(f"  복사된 파일: {copy_stat.get('copied', 0)}개")
    
    if args.dry_run:
        print("\n실제 복사를 수행하려면 --dry_run 옵션을 제거하세요.")


if __name__ == "__main__":
    main()
