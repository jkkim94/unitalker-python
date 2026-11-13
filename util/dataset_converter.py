#!/usr/bin/env python3
"""
데이터셋 변환 스크립트
VOCA 및 Multiface 데이터를 학습 가능한 형태로 변환합니다.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import shutil


def load_voca_vertices(voca_root: str) -> Optional[Dict]:
    """VOCA vertices 데이터 로드"""
    trainingdata_dir = os.path.join(voca_root, "trainingdata")
    
    # data_verts.npy 또는 data_verts.pkl 확인
    verts_npy = os.path.join(trainingdata_dir, "data_verts.npy")
    verts_pkl = os.path.join(trainingdata_dir, "data_verts.pkl")
    
    if os.path.exists(verts_npy):
        print(f"Loading vertices from {verts_npy}...")
        verts = np.load(verts_npy, allow_pickle=True)
        return {"data": verts, "format": "npy"}
    elif os.path.exists(verts_pkl):
        print(f"Loading vertices from {verts_pkl}...")
        with open(verts_pkl, 'rb') as f:
            verts = pickle.load(f)
        return {"data": verts, "format": "pkl"}
    
    return None


def load_voca_mapping(voca_root: str) -> Optional[Dict]:
    """VOCA subj_seq_to_idx 매핑 로드"""
    mapping_file = os.path.join(voca_root, "trainingdata", "subj_seq_to_idx.pkl")
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)
        return mapping
    return None


def extract_multiface_vertices(obj_file: str) -> Optional[np.ndarray]:
    """Multiface .obj 파일에서 vertices 추출"""
    try:
        vertices = []
        with open(obj_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        v = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(v)
        
        if vertices:
            return np.array(vertices, dtype=np.float32)
        return None
    except Exception as e:
        print(f"Error reading {obj_file}: {e}")
        return None


def process_multiface_segment(mesh_dir: str, segment_name: str, verbose: bool = False) -> Optional[np.ndarray]:
    """Multiface 세그먼트의 모든 프레임 vertices 추출"""
    segment_dir = os.path.join(mesh_dir, segment_name)
    
    if not os.path.exists(segment_dir):
        return None
    
    # .obj 파일들을 시간순으로 정렬
    obj_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.obj')])
    
    if not obj_files:
        return None
    
    all_vertices = []
    iterator = tqdm(obj_files, desc=f"Processing {segment_name}") if verbose else obj_files
    
    for obj_file in iterator:
        obj_path = os.path.join(segment_dir, obj_file)
        vertices = extract_multiface_vertices(obj_path)
        
        if vertices is not None:
            all_vertices.append(vertices)
    
    if all_vertices:
        # (T, N, 3) 형태로 변환
        vertices_array = np.array(all_vertices, dtype=np.float32)
        
        # Vertices 수 확인 (6172 또는 7306 등)
        num_vertices = vertices_array.shape[1]
        
        # 일반적인 vertices 수: 6172 (meshtalk) 또는 7306 (다른 형식)
        # 둘 다 허용
        if num_vertices not in [6172, 7306]:
            # 경고만 출력하고 계속 진행
            pass
        
        return vertices_array
    
    return None


def extract_voca_vertices_for_sample(verts_data: np.ndarray, mapping: Dict, speaker: str, sentence: str) -> Optional[np.ndarray]:
    """VOCA 매핑을 사용하여 특정 샘플의 vertices 추출"""
    try:
        if speaker not in mapping:
            return None
        
        speaker_mapping = mapping[speaker]
        if sentence not in speaker_mapping:
            return None
        
        frame_indices = speaker_mapping[sentence]
        # frame_indices는 {0: idx0, 1: idx1, ...} 형태
        if not isinstance(frame_indices, dict):
            return None
        
        indices = sorted(frame_indices.values())
        
        if not indices:
            return None
        
        # vertices 추출 (인덱스 범위 확인)
        max_idx = len(verts_data) - 1
        valid_indices = [idx for idx in indices if 0 <= idx <= max_idx]
        
        if not valid_indices:
            return None
        
        sample_verts = verts_data[valid_indices]
        
        # Shape 확인 및 변환
        # verts_data shape: (123341, 5023, 3)
        if len(sample_verts.shape) == 3:
            # 이미 (T, 5023, 3) 형태
            return sample_verts
        elif len(sample_verts.shape) == 2:
            # (T, 15069) 또는 (T, 5023) 형태
            if sample_verts.shape[1] == 5023 * 3:
                # (T, 15069) -> (T, 5023, 3)
                sample_verts = sample_verts.reshape(-1, 5023, 3)
            elif sample_verts.shape[1] == 5023:
                # (T, 5023) -> (T, 5023, 1) 또는 그대로 유지
                # FLAME은 3D이므로 3차원이어야 함
                pass
        
        return sample_verts
    except Exception as e:
        return None


def convert_voca_to_unitalker_format(
    voca_root: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):
    """VOCA 데이터를 UniTalker 형식으로 변환"""
    print("VOCA 데이터 변환 시작...")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # 데이터 로드
    verts_info = load_voca_vertices(voca_root)
    mapping = load_voca_mapping(voca_root)
    
    if verts_info is None:
        print("ERROR: vertices 데이터를 찾을 수 없습니다.")
        return
    
    if mapping is None:
        print("ERROR: 매핑 파일을 찾을 수 없습니다.")
        return
    
    verts_data = verts_info["data"]
    print(f"Vertices 데이터 shape: {verts_data.shape}")
    
    voca_audio_dir = os.path.join(voca_root, "VOCA_AUDIO")
    speakers = sorted([d for d in os.listdir(voca_audio_dir) 
                      if os.path.isdir(os.path.join(voca_audio_dir, d))])
    
    # 화자 ID 매핑
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(speakers)}
    
    all_samples = []
    total_duration = 0.0
    
    # 각 화자별로 처리
    print(f"  총 {len(speakers)}명의 화자 처리 중...")
    for speaker_idx, speaker in enumerate(speakers):
        print(f"  Processing speaker {speaker_idx+1}/{len(speakers)}: {speaker}")
        speaker_dir = os.path.join(voca_audio_dir, speaker)
        wav_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.wav')])
        
        # 샘플 분할
        n_samples = len(wav_files)
        n_train = int(n_samples * split_ratio[0])
        n_val = int(n_samples * split_ratio[1])
        
        splits = {
            'train': wav_files[:n_train],
            'val': wav_files[n_train:n_train+n_val],
            'test': wav_files[n_train+n_val:]
        }
        
        # 각 샘플 처리
        for split_name, files in splits.items():
            for wav_file in files:
                base_name = os.path.splitext(wav_file)[0]  # sentence01
                sentence_name = base_name  # sentence01
                
                # 오디오 파일 복사
                src_audio = os.path.join(speaker_dir, wav_file)
                dst_audio = os.path.join(output_dir, split_name, f"{speaker}_{base_name}.wav")
                shutil.copy2(src_audio, dst_audio)
                
                # vertices 데이터 추출
                sample_verts = extract_voca_vertices_for_sample(verts_data, mapping, speaker, sentence_name)
                
                if sample_verts is not None:
                    # Shape 확인
                    if len(sample_verts.shape) != 3 or sample_verts.shape[1] != 5023:
                        # Shape이 맞지 않으면 건너뛰기
                        continue
                    
                    # vertices 저장
                    vertices_path = os.path.join(output_dir, split_name, f"{speaker}_{base_name}.npy")
                    np.save(vertices_path, sample_verts)
                    
                    sample_info = {
                        "annot_type": "FLAME_5023_vertices",
                        "annot_path": f"{split_name}/{speaker}_{base_name}.npy",
                        "audio_path": f"{split_name}/{speaker}_{base_name}.wav",
                        "id": speaker_idx,
                        "fps": 60,
                        "dataset": "D1"
                    }
                    all_samples.append(sample_info)
                else:
                    # 매핑에 없는 sentence는 건너뛰기 (조용히)
                    pass
    
    # JSON 파일 생성
    info = {
        "id_list": speakers,
        "data_source": "VOCA",
        "total_duration": total_duration
    }
    
    # train/val/test로 분할
    train_samples = [s for s in all_samples if 'train' in s['annot_path']]
    val_samples = [s for s in all_samples if 'val' in s['annot_path']]
    test_samples = [s for s in all_samples if 'test' in s['annot_path']]
    
    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        json_data = {
            "info": info,
            "data": samples
        }
        
        json_path = os.path.join(output_dir, f"{split_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name}.json 생성 완료: {len(samples)}개 샘플")
    
    print(f"VOCA 변환 완료: {output_dir}")


def convert_multiface_to_unitalker_format(
    multiface_root: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):
    """Multiface 데이터를 UniTalker 형식으로 변환"""
    print("Multiface 데이터 변환 시작...")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    entities = sorted([d for d in os.listdir(multiface_root) 
                      if os.path.isdir(os.path.join(multiface_root, d)) and d.startswith("m--")])
    
    entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
    all_samples = []
    
    # 각 entity별로 처리
    print(f"  총 {len(entities)}개 entity 처리 중...")
    for entity_idx, entity in enumerate(entities):
        print(f"  Processing entity {entity_idx+1}/{len(entities)}: {entity}")
        entity_dir = os.path.join(multiface_root, entity)
        audio_dir = os.path.join(entity_dir, "audio")
        mesh_dir = os.path.join(entity_dir, "tracked_mesh")
        
        if not os.path.exists(audio_dir) or not os.path.exists(mesh_dir):
            print(f"Warning: {entity}에 audio 또는 mesh 디렉토리가 없습니다.")
            continue
        
        # 오디오 파일과 메시 세그먼트 매칭
        wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        mesh_segments = sorted([d for d in os.listdir(mesh_dir) 
                               if os.path.isdir(os.path.join(mesh_dir, d))])
        
        # SEN_으로 시작하는 세그먼트만 처리 (오디오와 매칭 가능한 것들)
        sen_segments = [s for s in mesh_segments if s.startswith('SEN_')]
        
        # 각 SEN 세그먼트별로 처리
        print(f"    {len(sen_segments)}개 SEN 세그먼트 처리 중...")
        for seg_idx, segment in enumerate(sen_segments):
            if seg_idx % 10 == 0 and seg_idx > 0:
                print(f"      진행: {seg_idx}/{len(sen_segments)} 세그먼트 완료")
            # 세그먼트 이름과 매칭되는 오디오 파일 찾기
            # 예: SEN_xxx 세그먼트는 SEN_xxx.wav와 매칭
            matching_wav = None
            segment_base = segment  # SEN_xxx
            for wav_file in wav_files:
                base_name = os.path.splitext(wav_file)[0]
                if base_name == segment_base:
                    matching_wav = wav_file
                    break
            
            if matching_wav is None:
                continue
            
            # vertices 추출 (verbose=False로 tqdm 제거하여 성능 개선)
            vertices = process_multiface_segment(mesh_dir, segment, verbose=False)
            if vertices is None:
                continue
            
            # vertices 수 확인 (6172 또는 7306 등 허용)
            num_vertices = vertices.shape[1] if len(vertices.shape) == 3 else 0
            # 6172 (meshtalk) 또는 7306 (다른 형식) 모두 허용
            
            # 파일명 생성
            base_name = os.path.splitext(matching_wav)[0]
            output_base = f"id_{entity_idx:02d}_audio_id_{len(all_samples):02d}_{base_name}"
            
            # 샘플 분할 결정 (화자별로 균등 분할)
            n_samples_per_entity = len(sen_segments)
            sample_idx = len([s for s in all_samples if s['id'] == entity_idx])
            entity_split_ratio = sample_idx / n_samples_per_entity if n_samples_per_entity > 0 else 0
            
            if entity_split_ratio < split_ratio[0]:
                split_name = 'train'
            elif entity_split_ratio < split_ratio[0] + split_ratio[1]:
                split_name = 'val'
            else:
                split_name = 'test'
            
            # 오디오 파일 복사
            src_audio = os.path.join(audio_dir, matching_wav)
            dst_audio = os.path.join(output_dir, split_name, f"{output_base}.wav")
            shutil.copy2(src_audio, dst_audio)
            
            # vertices 저장
            vertices_path = os.path.join(output_dir, split_name, f"{output_base}.npy")
            np.save(vertices_path, vertices)
            
            # vertices 수에 따라 annot_type 결정
            num_vertices = vertices.shape[1] if len(vertices.shape) == 3 else 6172
            if num_vertices == 7306:
                annot_type = "meshtalk_7306_vertices"
            else:
                annot_type = "meshtalk_6172_vertices"
            
            sample_info = {
                "annot_type": annot_type,
                "annot_path": f"{split_name}/{output_base}.npy",
                "audio_path": f"{split_name}/{output_base}.wav",
                "id": entity_idx,
                "fps": 30,
                "dataset": "D2"
            }
            all_samples.append(sample_info)
    
    # JSON 파일 생성
    info = {
        "id_list": [f"id_{i:02d}" for i in range(len(entities))],
        "data_source": "multiface",
        "total_duration": 0.0
    }
    
    # train/val/test로 분할
    train_samples = [s for s in all_samples if 'train' in s['annot_path']]
    val_samples = [s for s in all_samples if 'val' in s['annot_path']]
    test_samples = [s for s in all_samples if 'test' in s['annot_path']]
    
    for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        json_data = {
            "info": info,
            "data": samples
        }
        
        json_path = os.path.join(output_dir, f"{split_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name}.json 생성 완료: {len(samples)}개 샘플")
    
    print(f"Multiface 변환 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 변환")
    parser.add_argument("--dataset", type=str, choices=['voca', 'multiface', 'both'],
                       default='both', help="변환할 데이터셋")
    parser.add_argument("--voca_root", type=str,
                       default="/home/motion/omotion/datasets_raw/VOCA",
                       help="VOCA 데이터셋 루트")
    parser.add_argument("--multiface_root", type=str,
                       default="/home/motion/omotion/datasets_raw/multiface_data",
                       help="Multiface 데이터셋 루트")
    parser.add_argument("--output_voca", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D1_vocaset",
                       help="VOCA 출력 디렉토리")
    parser.add_argument("--output_multiface", type=str,
                       default="/home/motion/omotion/trainFramework/data/unitalker_data/D2_meshtalk",
                       help="Multiface 출력 디렉토리")
    parser.add_argument("--split_ratio", type=float, nargs=3,
                       default=[0.8, 0.1, 0.1], help="train/val/test 분할 비율")
    
    args = parser.parse_args()
    
    split_ratio = tuple(args.split_ratio)
    
    if args.dataset in ['voca', 'both']:
        convert_voca_to_unitalker_format(
            args.voca_root,
            args.output_voca,
            split_ratio
        )
    
    if args.dataset in ['multiface', 'both']:
        convert_multiface_to_unitalker_format(
            args.multiface_root,
            args.output_multiface,
            split_ratio
        )


if __name__ == "__main__":
    main()
