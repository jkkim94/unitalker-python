#!/usr/bin/env python3
"""
데이터셋 검증 스크립트
VOCA 및 Multiface 데이터가 정상적으로 다운로드되었는지 확인합니다.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy가 설치되지 않았습니다. 일부 검증 기능이 제한됩니다.")

try:
    import wave
    HAS_WAV = True
except ImportError:
    HAS_WAV = False


def validate_audio_file(audio_path: str) -> Dict:
    """오디오 파일 검증"""
    result = {
        "exists": os.path.exists(audio_path),
        "size": 0,
        "valid": False,
        "duration": None,
        "sample_rate": None,
        "error": None
    }
    
    if not result["exists"]:
        result["error"] = "파일이 존재하지 않음"
        return result
    
    try:
        result["size"] = os.path.getsize(audio_path)
        
        if HAS_WAV:
            try:
                with wave.open(audio_path, 'rb') as wav_file:
                    result["sample_rate"] = wav_file.getframerate()
                    result["duration"] = wav_file.getnframes() / result["sample_rate"]
                    result["valid"] = True
            except Exception as e:
                result["error"] = f"WAV 파일 읽기 실패: {e}"
        else:
            # 파일 크기만 확인
            if result["size"] > 0:
                result["valid"] = True
            else:
                result["error"] = "파일 크기가 0"
        
    except Exception as e:
        result["error"] = f"파일 접근 실패: {e}"
    
    return result


def validate_voca_dataset(voca_root: str) -> Dict:
    """VOCA 데이터셋 전체 검증"""
    print("=" * 80)
    print("VOCA 데이터셋 검증")
    print("=" * 80)
    
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    voca_audio_dir = os.path.join(voca_root, "VOCA_AUDIO")
    trainingdata_dir = os.path.join(voca_root, "trainingdata")
    
    # 1. 디렉토리 존재 확인
    if not os.path.exists(voca_audio_dir):
        result["valid"] = False
        result["errors"].append(f"오디오 디렉토리가 없습니다: {voca_audio_dir}")
        return result
    
    if not os.path.exists(trainingdata_dir):
        result["valid"] = False
        result["errors"].append(f"Training data 디렉토리가 없습니다: {trainingdata_dir}")
        return result
    
    print("\n[1] 디렉토리 구조 확인")
    print(f"  ✓ 오디오 디렉토리: {voca_audio_dir}")
    print(f"  ✓ Training data 디렉토리: {trainingdata_dir}")
    
    # 2. 오디오 파일 확인
    print("\n[2] 오디오 파일 검증")
    speakers = sorted([d for d in os.listdir(voca_audio_dir) 
                      if os.path.isdir(os.path.join(voca_audio_dir, d))])
    
    total_audio_files = 0
    valid_audio_files = 0
    invalid_audio_files = []
    audio_durations = []
    
    for speaker in speakers:
        speaker_dir = os.path.join(voca_audio_dir, speaker)
        wav_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.wav')])
        
        for wav_file in wav_files:
            total_audio_files += 1
            audio_path = os.path.join(speaker_dir, wav_file)
            audio_info = validate_audio_file(audio_path)
            
            if audio_info["valid"]:
                valid_audio_files += 1
                if audio_info["duration"]:
                    audio_durations.append(audio_info["duration"])
            else:
                invalid_audio_files.append(f"{speaker}/{wav_file}: {audio_info.get('error', 'unknown')}")
    
    print(f"  총 오디오 파일: {total_audio_files}개")
    print(f"  정상 파일: {valid_audio_files}개")
    print(f"  문제 파일: {len(invalid_audio_files)}개")
    
    if invalid_audio_files:
        result["warnings"].extend(invalid_audio_files[:10])  # 처음 10개만
        if len(invalid_audio_files) > 10:
            result["warnings"].append(f"... 외 {len(invalid_audio_files) - 10}개")
    
    result["statistics"]["audio"] = {
        "total": total_audio_files,
        "valid": valid_audio_files,
        "invalid": len(invalid_audio_files),
        "total_duration": sum(audio_durations) if audio_durations else 0
    }
    
    # 3. Vertices 파일 확인
    print("\n[3] Vertices 파일 검증")
    verts_npy = os.path.join(trainingdata_dir, "data_verts.npy")
    
    if not os.path.exists(verts_npy):
        result["valid"] = False
        result["errors"].append(f"data_verts.npy 파일이 없습니다: {verts_npy}")
    else:
        file_size = os.path.getsize(verts_npy) / 1e9  # GB
        print(f"  ✓ data_verts.npy 존재: {file_size:.2f} GB")
        
        if HAS_NUMPY:
            try:
                # 메모리 맵으로 shape만 확인
                data = np.load(verts_npy, mmap_mode='r')
                print(f"  ✓ Shape: {data.shape}")
                print(f"  ✓ 데이터 타입: {data.dtype}")
                
                result["statistics"]["vertices"] = {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "size_gb": file_size
                }
                
                # 예상 크기와 비교
                expected_size = np.prod(data.shape) * data.dtype.itemsize / 1e9
                if abs(file_size - expected_size) / expected_size > 0.1:
                    result["warnings"].append(f"파일 크기가 예상과 다릅니다: {file_size:.2f} GB (예상: {expected_size:.2f} GB)")
                
            except Exception as e:
                result["errors"].append(f"data_verts.npy 읽기 실패: {e}")
        else:
            result["warnings"].append("numpy가 없어 vertices 파일 내용을 확인할 수 없습니다.")
    
    # 4. 매핑 파일 확인
    print("\n[4] 매핑 파일 검증")
    mapping_file = os.path.join(trainingdata_dir, "subj_seq_to_idx.pkl")
    
    if not os.path.exists(mapping_file):
        result["valid"] = False
        result["errors"].append(f"매핑 파일이 없습니다: {mapping_file}")
    else:
        try:
            with open(mapping_file, 'rb') as f:
                mapping = pickle.load(f)
            
            print(f"  ✓ 매핑 파일 로드 성공")
            print(f"  ✓ 화자 수: {len(mapping)}")
            
            # 매핑 구조 검증
            total_sentences = 0
            total_frames = 0
            missing_sentences = []
            
            for speaker in speakers:
                if speaker not in mapping:
                    result["warnings"].append(f"화자 {speaker}가 매핑에 없습니다.")
                    continue
                
                speaker_mapping = mapping[speaker]
                speaker_dir = os.path.join(voca_audio_dir, speaker)
                wav_files = [os.path.splitext(f)[0] for f in os.listdir(speaker_dir) if f.endswith('.wav')]
                
                for wav_base in wav_files:
                    if wav_base not in speaker_mapping:
                        missing_sentences.append(f"{speaker}/{wav_base}")
                    else:
                        total_sentences += 1
                        frame_dict = speaker_mapping[wav_base]
                        if isinstance(frame_dict, dict):
                            total_frames += len(frame_dict)
            
            print(f"  ✓ 총 문장 수: {total_sentences}")
            print(f"  ✓ 총 프레임 수: {total_frames:,}")
            
            if missing_sentences:
                print(f"  ⚠ 매핑에 없는 문장: {len(missing_sentences)}개")
                result["warnings"].extend(missing_sentences[:10])
                if len(missing_sentences) > 10:
                    result["warnings"].append(f"... 외 {len(missing_sentences) - 10}개")
            
            result["statistics"]["mapping"] = {
                "num_speakers": len(mapping),
                "total_sentences": total_sentences,
                "total_frames": total_frames,
                "missing_sentences": len(missing_sentences)
            }
            
        except Exception as e:
            result["errors"].append(f"매핑 파일 읽기 실패: {e}")
    
    # 5. 오디오-vertices 매칭 검증
    print("\n[5] 오디오-Vertices 매칭 검증")
    if HAS_NUMPY and os.path.exists(verts_npy) and os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'rb') as f:
                mapping = pickle.load(f)
            
            data = np.load(verts_npy, mmap_mode='r')
            max_idx = len(data) - 1
            
            matched_samples = 0
            unmatched_samples = []
            
            for speaker in speakers[:3]:  # 처음 3명만 샘플로 확인
                if speaker not in mapping:
                    continue
                
                speaker_dir = os.path.join(voca_audio_dir, speaker)
                wav_files = [os.path.splitext(f)[0] for f in os.listdir(speaker_dir) if f.endswith('.wav')]
                
                for wav_base in wav_files[:5]:  # 각 화자당 5개만 확인
                    if wav_base in mapping[speaker]:
                        frame_dict = mapping[speaker][wav_base]
                        if isinstance(frame_dict, dict):
                            indices = sorted(frame_dict.values())
                            valid_indices = [idx for idx in indices if 0 <= idx <= max_idx]
                            
                            if len(valid_indices) == len(indices):
                                matched_samples += 1
                            else:
                                unmatched_samples.append(f"{speaker}/{wav_base}: {len(indices) - len(valid_indices)}개 인덱스 범위 초과")
            
            print(f"  ✓ 샘플 검증: {matched_samples}개 정상, {len(unmatched_samples)}개 문제")
            if unmatched_samples:
                result["warnings"].extend(unmatched_samples)
        
        except Exception as e:
            result["warnings"].append(f"매칭 검증 중 오류: {e}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    
    if result["errors"]:
        print(f"\n❌ 오류: {len(result['errors'])}개")
        for error in result["errors"][:5]:
            print(f"  - {error}")
        if len(result["errors"]) > 5:
            print(f"  ... 외 {len(result['errors']) - 5}개")
        result["valid"] = False
    
    if result["warnings"]:
        print(f"\n⚠ 경고: {len(result['warnings'])}개")
        for warning in result["warnings"][:5]:
            print(f"  - {warning}")
        if len(result["warnings"]) > 5:
            print(f"  ... 외 {len(result['warnings']) - 5}개")
    
    if result["valid"] and not result["errors"]:
        print("\n✅ VOCA 데이터셋이 정상적으로 다운로드되었습니다!")
    
    return result


def validate_multiface_dataset(multiface_root: str) -> Dict:
    """Multiface 데이터셋 전체 검증"""
    print("\n" + "=" * 80)
    print("Multiface 데이터셋 검증")
    print("=" * 80)
    
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    # 1. Entity 디렉토리 확인
    print("\n[1] Entity 디렉토리 확인")
    entities = sorted([d for d in os.listdir(multiface_root) 
                      if os.path.isdir(os.path.join(multiface_root, d)) and d.startswith("m--")])
    
    if not entities:
        result["valid"] = False
        result["errors"].append("Entity 디렉토리를 찾을 수 없습니다.")
        return result
    
    print(f"  ✓ Entity 수: {len(entities)}개")
    for entity in entities:
        print(f"    - {entity}")
    
    # 2. 각 Entity별 구조 확인
    print("\n[2] Entity별 구조 검증")
    total_audio = 0
    total_mesh_segments = 0
    total_sen_segments = 0
    matched_pairs = 0
    unmatched_audio = []
    unmatched_mesh = []
    
    entity_stats = {}
    
    for entity in entities:
        entity_dir = os.path.join(multiface_root, entity)
        audio_dir = os.path.join(entity_dir, "audio")
        mesh_dir = os.path.join(entity_dir, "tracked_mesh")
        
        entity_stat = {
            "audio_files": 0,
            "mesh_segments": 0,
            "sen_segments": 0,
            "matched": 0,
            "unmatched_audio": [],
            "unmatched_mesh": []
        }
        
        # 오디오 파일 확인
        if os.path.exists(audio_dir):
            wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
            entity_stat["audio_files"] = len(wav_files)
            total_audio += len(wav_files)
            
            # 오디오 파일 검증
            for wav_file in wav_files:
                audio_path = os.path.join(audio_dir, wav_file)
                audio_info = validate_audio_file(audio_path)
                if not audio_info["valid"]:
                    entity_stat["unmatched_audio"].append(f"{wav_file}: {audio_info.get('error', 'unknown')}")
        else:
            result["warnings"].append(f"{entity}: audio 디렉토리가 없습니다.")
        
        # Mesh 세그먼트 확인
        if os.path.exists(mesh_dir):
            mesh_segments = sorted([d for d in os.listdir(mesh_dir) 
                                   if os.path.isdir(os.path.join(mesh_dir, d))])
            entity_stat["mesh_segments"] = len(mesh_segments)
            total_mesh_segments += len(mesh_segments)
            
            sen_segments = [s for s in mesh_segments if s.startswith('SEN_')]
            entity_stat["sen_segments"] = len(sen_segments)
            total_sen_segments += len(sen_segments)
            
            # 오디오와 mesh 매칭 확인
            if os.path.exists(audio_dir):
                wav_files = [os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith('.wav')]
                
                for sen_seg in sen_segments:
                    if sen_seg in wav_files:
                        entity_stat["matched"] += 1
                        matched_pairs += 1
                    else:
                        entity_stat["unmatched_mesh"].append(sen_seg)
                        unmatched_mesh.append(f"{entity}/{sen_seg}")
                
                for wav_base in wav_files:
                    if wav_base.startswith('SEN_') and wav_base not in sen_segments:
                        unmatched_audio.append(f"{entity}/{wav_base}.wav")
        else:
            result["warnings"].append(f"{entity}: tracked_mesh 디렉토리가 없습니다.")
        
        entity_stats[entity] = entity_stat
    
    print(f"  ✓ 총 오디오 파일: {total_audio}개")
    print(f"  ✓ 총 mesh 세그먼트: {total_mesh_segments}개")
    print(f"  ✓ SEN 세그먼트: {total_sen_segments}개")
    print(f"  ✓ 매칭된 쌍: {matched_pairs}개")
    
    if unmatched_audio:
        print(f"  ⚠ 매칭되지 않은 오디오: {len(unmatched_audio)}개")
        result["warnings"].extend(unmatched_audio[:10])
    
    if unmatched_mesh:
        print(f"  ⚠ 매칭되지 않은 mesh: {len(unmatched_mesh)}개")
        result["warnings"].extend(unmatched_mesh[:10])
    
    result["statistics"]["entities"] = {
        "total": len(entities),
        "total_audio": total_audio,
        "total_mesh_segments": total_mesh_segments,
        "total_sen_segments": total_sen_segments,
        "matched_pairs": matched_pairs,
        "unmatched_audio": len(unmatched_audio),
        "unmatched_mesh": len(unmatched_mesh)
    }
    
    # 3. .obj 파일 샘플 검증
    print("\n[3] .obj 파일 샘플 검증")
    obj_samples_checked = 0
    obj_errors = []
    vertices_counts = defaultdict(int)
    
    for entity in entities[:2]:  # 처음 2개 entity만 샘플로 확인
        entity_dir = os.path.join(multiface_root, entity)
        mesh_dir = os.path.join(entity_dir, "tracked_mesh")
        
        if not os.path.exists(mesh_dir):
            continue
        
        sen_segments = [s for s in os.listdir(mesh_dir) 
                       if os.path.isdir(os.path.join(mesh_dir, s)) and s.startswith('SEN_')]
        
        for segment in sen_segments[:3]:  # 각 entity당 3개 세그먼트만
            segment_dir = os.path.join(mesh_dir, segment)
            obj_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.obj')])
            
            if not obj_files:
                continue
            
            # 첫 번째와 마지막 .obj 파일만 확인
            for obj_file in [obj_files[0], obj_files[-1]]:
                obj_path = os.path.join(segment_dir, obj_file)
                obj_samples_checked += 1
                
                try:
                    vertices = []
                    with open(obj_path, 'r') as f:
                        for line in f:
                            if line.startswith('v '):
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    if vertices:
                        num_vertices = len(vertices)
                        vertices_counts[num_vertices] += 1
                    else:
                        obj_errors.append(f"{entity}/{segment}/{obj_file}: vertices 없음")
                
                except Exception as e:
                    obj_errors.append(f"{entity}/{segment}/{obj_file}: {e}")
    
    print(f"  ✓ 검증한 .obj 파일: {obj_samples_checked}개")
    if vertices_counts:
        print(f"  ✓ Vertices 수 분포:")
        for count, num in sorted(vertices_counts.items()):
            print(f"    - {num}개 vertices: {count}개 파일")
    
    if obj_errors:
        print(f"  ⚠ 문제 파일: {len(obj_errors)}개")
        result["warnings"].extend(obj_errors[:5])
    
    result["statistics"]["obj_files"] = {
        "samples_checked": obj_samples_checked,
        "vertices_counts": dict(vertices_counts),
        "errors": len(obj_errors)
    }
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    
    if result["errors"]:
        print(f"\n❌ 오류: {len(result['errors'])}개")
        for error in result["errors"]:
            print(f"  - {error}")
        result["valid"] = False
    
    if result["warnings"]:
        print(f"\n⚠ 경고: {len(result['warnings'])}개")
        for warning in result["warnings"][:10]:
            print(f"  - {warning}")
        if len(result["warnings"]) > 10:
            print(f"  ... 외 {len(result['warnings']) - 10}개")
    
    if result["valid"] and not result["errors"]:
        print("\n✅ Multiface 데이터셋이 정상적으로 다운로드되었습니다!")
    
    return result


def save_validation_report(voca_result: Dict, multiface_result: Dict, output_path: str):
    """검증 결과를 JSON 파일로 저장"""
    report = {
        "voca": voca_result,
        "multiface": multiface_result,
        "summary": {
            "voca_valid": voca_result.get("valid", False),
            "multiface_valid": multiface_result.get("valid", False),
            "all_valid": voca_result.get("valid", False) and multiface_result.get("valid", False)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n검증 리포트가 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 검증")
    parser.add_argument("--voca_root", type=str,
                       default="/home/motion/omotion/datasets_raw/VOCA",
                       help="VOCA 데이터셋 루트 디렉토리")
    parser.add_argument("--multiface_root", type=str,
                       default="/home/motion/omotion/datasets_raw/multiface_data",
                       help="Multiface 데이터셋 루트 디렉토리")
    parser.add_argument("--dataset", type=str, choices=['voca', 'multiface', 'both'],
                       default='both', help="검증할 데이터셋")
    parser.add_argument("--output", type=str,
                       default="/home/motion/omotion/trainFramework/util/dataset_validation_report.json",
                       help="검증 리포트 출력 경로")
    
    args = parser.parse_args()
    
    voca_result = None
    multiface_result = None
    
    if args.dataset in ['voca', 'both']:
        if os.path.exists(args.voca_root):
            voca_result = validate_voca_dataset(args.voca_root)
        else:
            print(f"Warning: VOCA 디렉토리를 찾을 수 없습니다: {args.voca_root}")
    
    if args.dataset in ['multiface', 'both']:
        if os.path.exists(args.multiface_root):
            multiface_result = validate_multiface_dataset(args.multiface_root)
        else:
            print(f"Warning: Multiface 디렉토리를 찾을 수 없습니다: {args.multiface_root}")
    
    # 리포트 저장
    if voca_result and multiface_result:
        save_validation_report(voca_result, multiface_result, args.output)
    elif voca_result:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"voca": voca_result}, f, indent=2, ensure_ascii=False)
    elif multiface_result:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"multiface": multiface_result}, f, indent=2, ensure_ascii=False)
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("최종 검증 결과")
    print("=" * 80)
    
    if voca_result:
        status = "✅ 정상" if voca_result.get("valid") else "❌ 문제 있음"
        print(f"VOCA: {status}")
        if voca_result.get("errors"):
            print(f"  오류: {len(voca_result['errors'])}개")
        if voca_result.get("warnings"):
            print(f"  경고: {len(voca_result['warnings'])}개")
    
    if multiface_result:
        status = "✅ 정상" if multiface_result.get("valid") else "❌ 문제 있음"
        print(f"Multiface: {status}")
        if multiface_result.get("errors"):
            print(f"  오류: {len(multiface_result['errors'])}개")
        if multiface_result.get("warnings"):
            print(f"  경고: {len(multiface_result['warnings'])}개")


if __name__ == "__main__":
    main()
