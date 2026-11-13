#!/usr/bin/env python3
"""
데이터셋 준비 통합 스크립트
VOCA 및 Multiface 데이터셋을 분석하고 변환합니다.
"""

import os
import sys
import argparse
from pathlib import Path

# 현재 스크립트의 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from dataset_analyzer import analyze_voca_dataset, analyze_multiface_dataset, print_analysis_report, save_analysis_json
from dataset_converter import convert_voca_to_unitalker_format, convert_multiface_to_unitalker_format


def main():
    parser = argparse.ArgumentParser(description="데이터셋 준비 통합 스크립트")
    parser.add_argument("--mode", type=str, choices=['analyze', 'convert', 'all'],
                       default='all', help="실행 모드: analyze(분석만), convert(변환만), all(전체)")
    parser.add_argument("--dataset", type=str, choices=['voca', 'multiface', 'both'],
                       default='both', help="처리할 데이터셋")
    
    # 경로 설정
    parser.add_argument("--voca_root", type=str,
                       default="/home/motion/omotion/datasets_raw/VOCA",
                       help="VOCA 데이터셋 루트 디렉토리")
    parser.add_argument("--multiface_root", type=str,
                       default="/home/motion/omotion/datasets_raw/multiface_data",
                       help="Multiface 데이터셋 루트 디렉토리")
    
    # 출력 경로
    parser.add_argument("--output_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 출력 디렉토리")
    parser.add_argument("--output_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 출력 디렉토리")
    parser.add_argument("--analysis_output", type=str,
                       default="/home/motion/omotion/trainFramework/util/dataset_analysis.json",
                       help="분석 결과 JSON 출력 경로")
    
    # 분할 비율
    parser.add_argument("--split_ratio", type=float, nargs=3,
                       default=[0.8, 0.1, 0.1], help="train/val/test 분할 비율")
    
    args = parser.parse_args()
    
    split_ratio = tuple(args.split_ratio)
    
    # 분석 단계
    if args.mode in ['analyze', 'all']:
        print("=" * 80)
        print("데이터셋 분석 단계")
        print("=" * 80)
        
        voca_result = None
        multiface_result = None
        
        if args.dataset in ['voca', 'both']:
            print("\nVOCA 데이터셋 분석 중...")
            voca_result = analyze_voca_dataset(args.voca_root)
        
        if args.dataset in ['multiface', 'both']:
            print("\nMultiface 데이터셋 분석 중...")
            multiface_result = analyze_multiface_dataset(args.multiface_root)
        
        if voca_result and multiface_result:
            print_analysis_report(voca_result, multiface_result)
            save_analysis_json(voca_result, multiface_result, args.analysis_output)
        elif voca_result:
            print("\n[VOCA 데이터셋]")
            print(f"  총 화자 수: {len(voca_result['speakers'])}")
            print(f"  총 샘플 수: {voca_result['total_samples']}")
        elif multiface_result:
            print("\n[Multiface 데이터셋]")
            print(f"  총 Entity 수: {len(multiface_result['entities'])}")
            print(f"  총 샘플 수: {multiface_result['total_samples']}")
    
    # 변환 단계
    if args.mode in ['convert', 'all']:
        print("\n" + "=" * 80)
        print("데이터셋 변환 단계")
        print("=" * 80)
        
        if args.dataset in ['voca', 'both']:
            print("\nVOCA 데이터 변환 중...")
            convert_voca_to_unitalker_format(
                args.voca_root,
                args.output_voca,
                split_ratio
            )
        
        if args.dataset in ['multiface', 'both']:
            print("\nMultiface 데이터 변환 중...")
            convert_multiface_to_unitalker_format(
                args.multiface_root,
                args.output_multiface,
                split_ratio
            )
    
    print("\n" + "=" * 80)
    print("작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
