#!/usr/bin/env python3
"""
Mediapipe Blendshapes to FLAME 변환 테스트 스크립트

이 스크립트는 util 폴더의 mp_2_flame 모듈을 독립적으로 테스트합니다.
"""

import os
import sys
import numpy as np

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mp_2_flame import MP_2_FLAME, compute_head_pose_from_mp_landmarks_3d


def test_mp2flame_conversion():
    """Mediapipe blendshapes를 FLAME 계수로 변환하는 테스트"""
    print("=" * 60)
    print("Mediapipe Blendshapes to FLAME 변환 테스트")
    print("=" * 60)
    
    # 매핑 경로 설정
    mappings_path = os.path.join(os.path.dirname(__file__), 'mappings')
    
    if not os.path.exists(mappings_path):
        print(f"오류: 매핑 폴더를 찾을 수 없습니다: {mappings_path}")
        return False
    
    # MP_2_FLAME 인스턴스 생성
    try:
        mp2flame = MP_2_FLAME(mappings_path=mappings_path)
        print(f"✓ MP_2_FLAME 인스턴스 생성 성공")
    except Exception as e:
        print(f"✗ MP_2_FLAME 인스턴스 생성 실패: {e}")
        return False
    
    # 테스트 데이터 생성: 랜덤 blendshape scores [N, 52]
    # N=1 (단일 샘플)
    print("\n1. 단일 샘플 테스트 (N=1)")
    blendshape_scores_single = np.random.rand(1, 52).astype(np.float32)
    print(f"   입력 shape: {blendshape_scores_single.shape}")
    
    try:
        exp, jaw, eye_pose = mp2flame.convert(blendshape_scores=blendshape_scores_single)
        print(f"   ✓ 변환 성공")
        print(f"   - Expression shape: {exp.shape} (예상: (1, 100))")
        print(f"   - Jaw pose shape: {jaw.shape} (예상: (1, 3))")
        print(f"   - Eye pose shape: {eye_pose.shape} (예상: (1, 6))")
        print(f"   - Expression 범위: [{exp.min():.4f}, {exp.max():.4f}]")
        print(f"   - Jaw pose 범위: [{jaw.min():.4f}, {jaw.max():.4f}]")
        print(f"   - Eye pose 범위: [{eye_pose.min():.4f}, {eye_pose.max():.4f}]")
    except Exception as e:
        print(f"   ✗ 변환 실패: {e}")
        return False
    
    # 테스트 데이터 생성: 배치 샘플 [N, 52]
    # N=5 (배치)
    print("\n2. 배치 샘플 테스트 (N=5)")
    blendshape_scores_batch = np.random.rand(5, 52).astype(np.float32)
    print(f"   입력 shape: {blendshape_scores_batch.shape}")
    
    try:
        exp, jaw, eye_pose = mp2flame.convert(blendshape_scores=blendshape_scores_batch)
        print(f"   ✓ 변환 성공")
        print(f"   - Expression shape: {exp.shape} (예상: (5, 100))")
        print(f"   - Jaw pose shape: {jaw.shape} (예상: (5, 3))")
        print(f"   - Eye pose shape: {eye_pose.shape} (예상: (5, 6))")
    except Exception as e:
        print(f"   ✗ 변환 실패: {e}")
        return False
    
    # 실제 Mediapipe blendshape 값 범위 시뮬레이션 (0~1 사이)
    print("\n3. 실제 Mediapipe 값 범위 테스트 (0~1)")
    blendshape_scores_realistic = np.clip(np.random.rand(1, 52), 0.0, 1.0).astype(np.float32)
    print(f"   입력 shape: {blendshape_scores_realistic.shape}")
    print(f"   입력 범위: [{blendshape_scores_realistic.min():.4f}, {blendshape_scores_realistic.max():.4f}]")
    
    try:
        exp, jaw, eye_pose = mp2flame.convert(blendshape_scores=blendshape_scores_realistic)
        print(f"   ✓ 변환 성공")
        print(f"   - Expression shape: {exp.shape}")
        print(f"   - Jaw pose shape: {jaw.shape}")
        print(f"   - Eye pose shape: {eye_pose.shape}")
    except Exception as e:
        print(f"   ✗ 변환 실패: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("모든 변환 테스트 통과!")
    print("=" * 60)
    return True


def test_head_pose_estimation():
    """Head pose 추정 함수 테스트"""
    print("\n" + "=" * 60)
    print("Head Pose 추정 테스트")
    print("=" * 60)
    
    # 가상의 얼굴 랜덤 랜드마크 생성 [478, 3]
    # Mediapipe는 478개의 3D 랜드마크를 제공
    img_h, img_w = 480, 640
    
    # 랜덤 랜드마크 생성 (정규화된 좌표 시뮬레이션)
    face_landmarks = np.random.rand(478, 3).astype(np.float32)
    # x, y 좌표를 이미지 크기에 맞게 스케일링
    face_landmarks[:, 0] = face_landmarks[:, 0] * img_w
    face_landmarks[:, 1] = face_landmarks[:, 1] * img_h
    # z 좌표는 깊이 정보 (작은 값)
    face_landmarks[:, 2] = (face_landmarks[:, 2] - 0.5) * 0.1
    
    print(f"입력 랜드마크 shape: {face_landmarks.shape}")
    print(f"이미지 크기: {img_w} x {img_h}")
    
    try:
        rotation_vec, translation_vec = compute_head_pose_from_mp_landmarks_3d(
            face_landmarks=face_landmarks,
            img_h=img_h,
            img_w=img_w
        )
        print(f"✓ Head pose 추정 성공")
        print(f"   - Rotation vector shape: {rotation_vec.shape} (예상: (3, 1))")
        print(f"   - Translation vector shape: {translation_vec.shape} (예상: (3, 1))")
        print(f"   - Rotation vector:\n{rotation_vec}")
        print(f"   - Translation vector:\n{translation_vec}")
    except Exception as e:
        print(f"✗ Head pose 추정 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Head pose 추정 테스트 통과!")
    print("=" * 60)
    return True


def main():
    """메인 테스트 함수"""
    print("\n")
    print("Mediapipe Blendshapes to FLAME 변환기 테스트 시작")
    print("=" * 60)
    
    # 변환 테스트
    conversion_success = test_mp2flame_conversion()
    
    # Head pose 추정 테스트
    pose_success = test_head_pose_estimation()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"변환 테스트: {'✓ 통과' if conversion_success else '✗ 실패'}")
    print(f"Head pose 추정 테스트: {'✓ 통과' if pose_success else '✗ 실패'}")
    
    if conversion_success and pose_success:
        print("\n모든 테스트가 성공적으로 완료되었습니다!")
        return 0
    else:
        print("\n일부 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    exit(main())
