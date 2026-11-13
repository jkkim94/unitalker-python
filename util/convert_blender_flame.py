#!/usr/bin/env python3
"""
Blender GLB에서 추출한 FLAME 파라미터를 FLAME Head Tracker 형식으로 변환

사용법:
    python convert_blender_flame.py --input blender_params.json --output flame_params.npz
"""

import numpy as np
import json
import argparse
import sys


def convert_blender_pose_to_flame(blender_pose, method='auto'):
    """
    Blender GLB의 18개 pose를 FLAME 형식으로 변환
    
    Args:
        blender_pose: (18,) 또는 (1, 18) - Blender GLB pose 파라미터
        method: 'method1' (global_translation 포함), 'method2' (head_pose 포함), 
                'method3' (모든 회전), 또는 'auto'
    
    Returns:
        dict: FLAME 형식 파라미터
    """
    blender_pose = np.array(blender_pose).flatten()
    
    if len(blender_pose) != 18:
        raise ValueError(f"Pose 파라미터는 18개여야 합니다. 현재: {len(blender_pose)}개")
    
    # 자동 감지: 회전 파라미터는 보통 -π ~ +π 범위, translation은 더 넓은 범위
    if method == 'auto':
        # 각 구간별 범위 분석
        range_0_2 = np.abs(blender_pose[0:3]).max()
        range_3_5 = np.abs(blender_pose[3:6]).max()
        range_6_8 = np.abs(blender_pose[6:9]).max()
        range_9_11 = np.abs(blender_pose[9:12]).max()
        range_12_17 = np.abs(blender_pose[12:18]).max()
        
        rotation_threshold = 3.14  # π
        
        # 3-5 구간이 translation인지 확인 (더 큰 값 범위)
        if range_3_5 > rotation_threshold * 2 and range_3_5 > range_0_2 * 2:
            method = 'method1'  # global_translation 포함
        elif range_0_2 < rotation_threshold and range_3_5 < rotation_threshold:
            # 모든 구간이 회전 범위 내
            method = 'method2'  # head_pose 포함 (기본값)
        else:
            method = 'method1'  # 기본값
    
    if method == 'method1':
        # 방법 1: Global Translation 포함
        # [global_rotation(3), global_translation(3), neck(3), jaw(3), eye(6)]
        return {
            'head_pose': np.zeros((1, 3)),  # 보통 0으로 설정
            'neck_pose': blender_pose[6:9].reshape(1, 3),
            'jaw_pose': blender_pose[9:12].reshape(1, 3),
            'eye_pose': blender_pose[12:18].reshape(1, 6),
            'global_rotation': blender_pose[0:3].reshape(1, 3),  # 참고용
            'global_translation': blender_pose[3:6].reshape(1, 3),  # 참고용
        }
    elif method == 'method2':
        # 방법 2: Head Pose 포함
        # [head(3), neck(3), jaw(3), eye(6), additional(3)]
        return {
            'head_pose': blender_pose[0:3].reshape(1, 3),
            'neck_pose': blender_pose[3:6].reshape(1, 3),
            'jaw_pose': blender_pose[6:9].reshape(1, 3),
            'eye_pose': blender_pose[9:15].reshape(1, 6),
            'additional': blender_pose[15:18].reshape(1, 3),  # 참고용
        }
    elif method == 'method3':
        # 방법 3: 모든 회전 파라미터
        # [global_rotation(3), head(3), neck(3), jaw(3), eye(6)]
        return {
            'head_pose': blender_pose[3:6].reshape(1, 3),
            'neck_pose': blender_pose[6:9].reshape(1, 3),
            'jaw_pose': blender_pose[9:12].reshape(1, 3),
            'eye_pose': blender_pose[12:18].reshape(1, 6),
            'global_rotation': blender_pose[0:3].reshape(1, 3),  # 참고용
        }
    else:
        raise ValueError(f"알 수 없는 방법: {method}")


def convert_blender_to_flame(blender_shape, blender_expression, blender_pose, pose_method='auto'):
    """
    Blender GLB 파라미터를 FLAME Head Tracker 형식으로 변환
    
    Args:
        blender_shape: (300,) - Shape 파라미터
        blender_expression: (100,) - Expression 파라미터
        blender_pose: (18,) - Pose 파라미터
        pose_method: Pose 변환 방법
    
    Returns:
        dict: FLAME Head Tracker 형식 파라미터
    """
    # Shape와 Expression은 그대로 사용
    shape = np.array(blender_shape).reshape(1, 300)
    expression = np.array(blender_expression).reshape(1, 100)
    
    # Pose 변환
    pose_dict = convert_blender_pose_to_flame(blender_pose, method=pose_method)
    
    # FLAME Head Tracker 형식으로 구성
    flame_params = {
        'shape': shape,
        'exp': expression,
        'head_pose': pose_dict['head_pose'],
        'neck_pose': pose_dict['neck_pose'],
        'jaw_pose': pose_dict['jaw_pose'],
        'eye_pose': pose_dict['eye_pose'],
    }
    
    return flame_params, pose_dict


def load_blender_params(input_file):
    """
    Blender에서 추출한 파라미터 로드
    
    지원 형식:
    - JSON: {"shape": [...], "expression": [...], "pose": [...]}
    - NPZ: shape, expression, pose 키
    """
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            data = json.load(f)
        return {
            'shape': np.array(data['shape']),
            'expression': np.array(data['expression']),
            'pose': np.array(data['pose']),
        }
    elif input_file.endswith('.npz'):
        data = np.load(input_file)
        return {
            'shape': data['shape'],
            'expression': data['expression'],
            'pose': data['pose'],
        }
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {input_file}")


def save_flame_params(flame_params, output_file):
    """
    FLAME 파라미터를 NPZ 파일로 저장
    """
    np.savez_compressed(
        output_file,
        shape=flame_params['shape'],
        exp=flame_params['exp'],
        head_pose=flame_params['head_pose'],
        neck_pose=flame_params['neck_pose'],
        jaw_pose=flame_params['jaw_pose'],
        eye_pose=flame_params['eye_pose'],
    )
    print(f"✅ FLAME 파라미터 저장 완료: {output_file}")


def validate_params(blender_params):
    """
    Blender 파라미터 검증
    """
    errors = []
    
    if 'shape' not in blender_params:
        errors.append("Shape 파라미터가 없습니다")
    elif len(blender_params['shape']) != 300:
        errors.append(f"Shape 파라미터는 300개여야 합니다. 현재: {len(blender_params['shape'])}개")
    
    if 'expression' not in blender_params:
        errors.append("Expression 파라미터가 없습니다")
    elif len(blender_params['expression']) != 100:
        errors.append(f"Expression 파라미터는 100개여야 합니다. 현재: {len(blender_params['expression'])}개")
    
    if 'pose' not in blender_params:
        errors.append("Pose 파라미터가 없습니다")
    elif len(blender_params['pose']) != 18:
        errors.append(f"Pose 파라미터는 18개여야 합니다. 현재: {len(blender_params['pose'])}개")
    
    if errors:
        print("❌ 검증 실패:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ 파라미터 검증 통과")
    return True


def print_pose_structure(blender_pose):
    """
    Pose 파라미터 구조 출력
    """
    pose = np.array(blender_pose).flatten()
    print("\n" + "=" * 60)
    print("Pose 파라미터 구조 분석")
    print("=" * 60)
    print(f"총 개수: {len(pose)}개")
    print(f"범위: [{pose.min():.4f}, {pose.max():.4f}]")
    print(f"평균: {pose.mean():.4f}")
    print(f"표준편차: {pose.std():.4f}")
    print("\n구간별 분석:")
    print(f"  [0-2]:   {pose[0:3]}   (범위: [{pose[0:3].min():.4f}, {pose[0:3].max():.4f}])")
    print(f"  [3-5]:   {pose[3:6]}   (범위: [{pose[3:6].min():.4f}, {pose[3:6].max():.4f}])")
    print(f"  [6-8]:   {pose[6:9]}   (범위: [{pose[6:9].min():.4f}, {pose[6:9].max():.4f}])")
    print(f"  [9-11]:  {pose[9:12]}  (범위: [{pose[9:12].min():.4f}, {pose[9:12].max():.4f}])")
    print(f"  [12-17]: {pose[12:18]} (범위: [{pose[12:18].min():.4f}, {pose[12:18].max():.4f}])")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Blender GLB FLAME 파라미터를 FLAME Head Tracker 형식으로 변환'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Blender에서 추출한 파라미터 파일 (JSON 또는 NPZ)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='flame_params.npz',
        help='출력 FLAME 파라미터 파일 (기본값: flame_params.npz)'
    )
    parser.add_argument(
        '--pose-method',
        type=str,
        default='auto',
        choices=['auto', 'method1', 'method2', 'method3'],
        help='Pose 변환 방법 (기본값: auto)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Pose 파라미터 구조 분석만 수행'
    )
    
    args = parser.parse_args()
    
    # Blender 파라미터 로드
    print(f"📂 Blender 파라미터 로드: {args.input}")
    try:
        blender_params = load_blender_params(args.input)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        sys.exit(1)
    
    # 검증
    if not validate_params(blender_params):
        sys.exit(1)
    
    # Pose 구조 분석
    print_pose_structure(blender_params['pose'])
    
    if args.analyze:
        print("\n분석만 수행하고 종료합니다.")
        return
    
    # 변환
    print(f"\n🔄 FLAME 형식으로 변환 중... (방법: {args.pose_method})")
    try:
        flame_params, pose_info = convert_blender_to_flame(
            blender_params['shape'],
            blender_params['expression'],
            blender_params['pose'],
            pose_method=args.pose_method
        )
        
        print("\n✅ 변환 완료!")
        print(f"  Shape: {flame_params['shape'].shape}")
        print(f"  Expression: {flame_params['exp'].shape}")
        print(f"  Head Pose: {flame_params['head_pose'].shape}")
        print(f"  Neck Pose: {flame_params['neck_pose'].shape}")
        print(f"  Jaw Pose: {flame_params['jaw_pose'].shape}")
        print(f"  Eye Pose: {flame_params['eye_pose'].shape}")
        
        # 저장
        save_flame_params(flame_params, args.output)
        
        # 추가 정보 출력
        if 'global_rotation' in pose_info:
            print(f"\n📝 추가 정보:")
            print(f"  Global Rotation: {pose_info['global_rotation']}")
            print(f"  Global Translation: {pose_info['global_translation']}")
            print("  (이 값들은 camera pose로 사용할 수 있습니다)")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
