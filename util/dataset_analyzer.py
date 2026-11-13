#!/usr/bin/env python3
"""
데이터셋 분석 스크립트
VOCA 및 Multiface 데이터셋의 구조를 분석합니다.
"""

import os
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy가 설치되지 않았습니다. 일부 분석 기능이 제한됩니다.")


def analyze_voca_vertices_file(voca_root: str) -> Optional[Dict]:
    """VOCA vertices 파일 상세 분석"""
    if not HAS_NUMPY:
        return None
    
    trainingdata_dir = os.path.join(voca_root, "trainingdata")
    verts_npy = os.path.join(trainingdata_dir, "data_verts.npy")
    verts_pkl = os.path.join(trainingdata_dir, "data_verts.pkl")
    
    verts_info = {}
    
    if os.path.exists(verts_npy):
        try:
            # 메모리 맵 모드로 로드 (대용량 파일 처리)
            data = np.load(verts_npy, mmap_mode='r')
            verts_info = {
                "file": "data_verts.npy",
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "size_bytes": data.nbytes,
                "size_gb": data.nbytes / 1e9,
                "num_elements": data.size,
                "min_value": float(np.min(data)) if data.size < 1e8 else "too_large",
                "max_value": float(np.max(data)) if data.size < 1e8 else "too_large",
                "mean_value": float(np.mean(data)) if data.size < 1e8 else "too_large",
            }
            
            # Shape 해석
            if len(data.shape) == 1:
                verts_info["interpretation"] = "1D array - flattened vertices"
            elif len(data.shape) == 2:
                if data.shape[1] == 5023 * 3:
                    verts_info["interpretation"] = f"2D array: (frames, {5023} vertices * 3 coords)"
                    verts_info["estimated_frames"] = data.shape[0]
                    verts_info["vertices_per_frame"] = 5023
                else:
                    verts_info["interpretation"] = f"2D array: shape {data.shape}"
            else:
                verts_info["interpretation"] = f"Multi-dimensional: {data.shape}"
            
            # 샘플 데이터 추출 (처음 몇 개만)
            if data.size < 1e6:
                verts_info["sample_data"] = data[:min(5, len(data))].tolist() if len(data.shape) == 1 else data[0, :min(5, data.shape[1])].tolist()
        except Exception as e:
            verts_info["error"] = str(e)
    elif os.path.exists(verts_pkl):
        verts_info = {
            "file": "data_verts.pkl",
            "note": "Pickle 파일 - 직접 로드 필요"
        }
    
    return verts_info if verts_info else None


def analyze_voca_mapping_file(voca_root: str) -> Optional[Dict]:
    """VOCA 매핑 파일 상세 분석"""
    mapping_file = os.path.join(voca_root, "trainingdata", "subj_seq_to_idx.pkl")
    
    if not os.path.exists(mapping_file):
        return None
    
    try:
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)
        
        mapping_info = {
            "file": "subj_seq_to_idx.pkl",
            "num_speakers": len(mapping),
            "speakers": list(mapping.keys()),
            "structure": {}
        }
        
        # 각 화자별 상세 정보
        total_frames = 0
        total_sentences = 0
        frame_ranges = []
        
        for speaker, sentences in mapping.items():
            speaker_info = {
                "num_sentences": len(sentences),
                "sentences": list(sentences.keys()),
                "frame_stats": {}
            }
            
            speaker_frames = []
            for sentence, frame_dict in sentences.items():
                if isinstance(frame_dict, dict):
                    frames = list(frame_dict.values())
                    speaker_frames.extend(frames)
                    speaker_info["frame_stats"][sentence] = {
                        "num_frames": len(frames),
                        "frame_range": [min(frames), max(frames)] if frames else None
                    }
            
            if speaker_frames:
                speaker_info["total_frames"] = len(speaker_frames)
                speaker_info["frame_range"] = [min(speaker_frames), max(speaker_frames)]
                total_frames += len(speaker_frames)
                total_sentences += len(sentences)
                frame_ranges.extend(speaker_frames)
            
            mapping_info["structure"][speaker] = speaker_info
        
        mapping_info["summary"] = {
            "total_sentences": total_sentences,
            "total_frames": total_frames,
            "overall_frame_range": [min(frame_ranges), max(frame_ranges)] if frame_ranges else None,
            "average_frames_per_sentence": total_frames / total_sentences if total_sentences > 0 else 0
        }
        
        return mapping_info
    except Exception as e:
        return {"error": str(e)}


def analyze_voca_dataset(voca_root: str) -> Dict:
    """VOCA 데이터셋 구조 분석"""
    voca_audio_dir = os.path.join(voca_root, "VOCA_AUDIO")
    trainingdata_dir = os.path.join(voca_root, "trainingdata")
    
    result = {
        "dataset": "VOCA",
        "audio_dir": voca_audio_dir,
        "trainingdata_dir": trainingdata_dir,
        "speakers": {},
        "total_samples": 0,
        "files": {},
        "detailed_analysis": {}
    }
    
    # 오디오 파일 분석
    if os.path.exists(voca_audio_dir):
        speakers = [d for d in os.listdir(voca_audio_dir) 
                   if os.path.isdir(os.path.join(voca_audio_dir, d))]
        
        for speaker in speakers:
            speaker_dir = os.path.join(voca_audio_dir, speaker)
            wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            
            result["speakers"][speaker] = {
                "num_samples": len(wav_files),
                "samples": sorted(wav_files)
            }
            result["total_samples"] += len(wav_files)
    
    # trainingdata 파일 확인
    if os.path.exists(trainingdata_dir):
        files = os.listdir(trainingdata_dir)
        result["files"]["trainingdata"] = {
            "files": files,
            "has_verts": "data_verts.npy" in files or "data_verts.pkl" in files,
            "has_audio": "raw_audio_fixed.pkl" in files or "processed_audio_deepspeech.pkl" in files,
            "has_templates": "templates.pkl" in files,
            "has_mapping": "subj_seq_to_idx.pkl" in files
        }
    
    # 상세 분석
    print("  VOCA vertices 파일 분석 중...")
    verts_info = analyze_voca_vertices_file(voca_root)
    if verts_info:
        result["detailed_analysis"]["vertices"] = verts_info
    
    print("  VOCA 매핑 파일 분석 중...")
    mapping_info = analyze_voca_mapping_file(voca_root)
    if mapping_info:
        result["detailed_analysis"]["mapping"] = mapping_info
    
    return result


def analyze_multiface_obj_file(obj_path: str) -> Optional[Dict]:
    """Multiface .obj 파일 상세 분석"""
    if not HAS_NUMPY:
        return None
    
    try:
        vertices = []
        faces = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        v = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(v)
                elif line.startswith('f '):
                    faces.append(line.strip())
        
        if vertices:
            vertices_array = np.array(vertices, dtype=np.float32)
            return {
                "num_vertices": len(vertices),
                "num_faces": len(faces),
                "vertices_shape": list(vertices_array.shape),
                "vertices_range": {
                    "min": [float(x) for x in np.min(vertices_array, axis=0)],
                    "max": [float(x) for x in np.max(vertices_array, axis=0)],
                    "mean": [float(x) for x in np.mean(vertices_array, axis=0)]
                },
                "sample_vertices": vertices[:3] if len(vertices) >= 3 else vertices
            }
    except Exception as e:
        return {"error": str(e)}
    
    return None


def analyze_multiface_dataset(multiface_root: str) -> Dict:
    """Multiface 데이터셋 구조 분석"""
    result = {
        "dataset": "Multiface",
        "entities": {},
        "total_samples": 0,
        "structure": {},
        "detailed_analysis": {}
    }
    
    # entity 디렉토리 찾기 (m--로 시작하는 디렉토리)
    entities = [d for d in os.listdir(multiface_root) 
               if os.path.isdir(os.path.join(multiface_root, d)) and d.startswith("m--")]
    
    # 전체 통계
    total_vertices_samples = []
    total_obj_files = 0
    
    for entity in entities:
        entity_dir = os.path.join(multiface_root, entity)
        
        # 오디오 파일 확인
        audio_dir = os.path.join(entity_dir, "audio")
        wav_files = []
        if os.path.exists(audio_dir):
            wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
        # 메시 파일 확인
        mesh_dir = os.path.join(entity_dir, "tracked_mesh")
        mesh_segments = []
        if os.path.exists(mesh_dir):
            mesh_segments = [d for d in os.listdir(mesh_dir) 
                           if os.path.isdir(os.path.join(mesh_dir, d))]
        
        # 각 세그먼트별 프레임 수 확인 및 상세 분석
        segment_info = {}
        sen_segments = []  # SEN_으로 시작하는 세그먼트
        
        for segment in mesh_segments:
            segment_path = os.path.join(mesh_dir, segment)
            obj_files = sorted([f for f in os.listdir(segment_path) if f.endswith('.obj')])
            segment_info[segment] = {
                "num_frames": len(obj_files),
                "obj_files": obj_files[:5] if len(obj_files) > 5 else obj_files  # 처음 5개만
            }
            total_obj_files += len(obj_files)
            
            # SEN_ 세그먼트만 추적
            if segment.startswith('SEN_'):
                sen_segments.append(segment)
                
                # 첫 번째 .obj 파일 분석 (샘플)
                if obj_files and len(total_vertices_samples) < 3:
                    sample_obj = os.path.join(segment_path, obj_files[0])
                    obj_analysis = analyze_multiface_obj_file(sample_obj)
                    if obj_analysis:
                        total_vertices_samples.append(obj_analysis)
        
        result["entities"][entity] = {
            "num_audio_samples": len(wav_files),
            "audio_samples": sorted(wav_files),
            "num_mesh_segments": len(mesh_segments),
            "num_sen_segments": len(sen_segments),
            "mesh_segments": sorted(mesh_segments),
            "sen_segments": sorted(sen_segments),
            "segment_info": segment_info
        }
        result["total_samples"] += len(wav_files)
    
    # 상세 분석 결과 저장
    if total_vertices_samples:
        result["detailed_analysis"]["obj_file_samples"] = total_vertices_samples
        if HAS_NUMPY:
            num_vertices_list = [s["num_vertices"] for s in total_vertices_samples if "num_vertices" in s]
            if num_vertices_list:
                result["detailed_analysis"]["vertices_statistics"] = {
                    "sample_vertex_counts": num_vertices_list,
                    "typical_vertex_count": max(set(num_vertices_list), key=num_vertices_list.count) if num_vertices_list else None
                }
    
    result["detailed_analysis"]["summary"] = {
        "total_obj_files": total_obj_files,
        "total_entities": len(entities),
        "total_audio_samples": result["total_samples"]
    }
    
    return result


def print_analysis_report(voca_result: Dict, multiface_result: Dict):
    """분석 결과 리포트 출력"""
    print("=" * 80)
    print("데이터셋 분석 리포트 (상세)")
    print("=" * 80)
    
    print("\n[VOCA 데이터셋]")
    print(f"  오디오 디렉토리: {voca_result['audio_dir']}")
    print(f"  Training Data 디렉토리: {voca_result['trainingdata_dir']}")
    print(f"  총 화자 수: {len(voca_result['speakers'])}")
    print(f"  총 샘플 수: {voca_result['total_samples']}")
    
    print("\n  화자별 샘플 수:")
    for speaker, info in sorted(voca_result['speakers'].items()):
        print(f"    {speaker}: {info['num_samples']}개")
    
    if 'trainingdata' in voca_result.get('files', {}):
        files_info = voca_result['files']['trainingdata']
        print("\n  Training Data 파일:")
        print(f"    Vertices 데이터: {'있음' if files_info['has_verts'] else '없음'}")
        print(f"    오디오 데이터: {'있음' if files_info['has_audio'] else '없음'}")
        print(f"    Templates: {'있음' if files_info['has_templates'] else '없음'}")
        print(f"    매핑 파일: {'있음' if files_info['has_mapping'] else '없음'}")
    
    # 상세 분석 결과 출력
    if 'detailed_analysis' in voca_result:
        print("\n  [상세 분석]")
        
        # Vertices 파일 분석
        if 'vertices' in voca_result['detailed_analysis']:
            verts = voca_result['detailed_analysis']['vertices']
            print("\n    Vertices 파일 (data_verts.npy):")
            print(f"      Shape: {verts.get('shape', 'N/A')}")
            print(f"      데이터 타입: {verts.get('dtype', 'N/A')}")
            print(f"      파일 크기: {verts.get('size_gb', 0):.2f} GB")
            print(f"      해석: {verts.get('interpretation', 'N/A')}")
            if 'estimated_frames' in verts:
                print(f"      예상 프레임 수: {verts['estimated_frames']:,}")
            if 'vertices_per_frame' in verts:
                print(f"      프레임당 vertices 수: {verts['vertices_per_frame']:,}")
        
        # 매핑 파일 분석
        if 'mapping' in voca_result['detailed_analysis']:
            mapping = voca_result['detailed_analysis']['mapping']
            print("\n    매핑 파일 (subj_seq_to_idx.pkl):")
            print(f"      화자 수: {mapping.get('num_speakers', 'N/A')}")
            if 'summary' in mapping:
                summary = mapping['summary']
                print(f"      총 문장 수: {summary.get('total_sentences', 'N/A')}")
                print(f"      총 프레임 수: {summary.get('total_frames', 'N/A'):,}")
                print(f"      문장당 평균 프레임 수: {summary.get('average_frames_per_sentence', 0):.1f}")
                if summary.get('overall_frame_range'):
                    print(f"      전체 프레임 범위: {summary['overall_frame_range'][0]:,} ~ {summary['overall_frame_range'][1]:,}")
            
            # 샘플 화자 정보
            if 'structure' in mapping and mapping['structure']:
                sample_speaker = list(mapping['structure'].keys())[0]
                sample_info = mapping['structure'][sample_speaker]
                print(f"\n      샘플 화자 ({sample_speaker}):")
                print(f"        문장 수: {sample_info.get('num_sentences', 'N/A')}")
                print(f"        총 프레임 수: {sample_info.get('total_frames', 'N/A')}")
                if sample_info.get('frame_range'):
                    print(f"        프레임 범위: {sample_info['frame_range'][0]:,} ~ {sample_info['frame_range'][1]:,}")
    
    print("\n[Multiface 데이터셋]")
    print(f"  총 Entity 수: {len(multiface_result['entities'])}")
    print(f"  총 샘플 수: {multiface_result['total_samples']}")
    
    print("\n  Entity별 정보:")
    for entity, info in sorted(multiface_result['entities'].items()):
        print(f"    {entity}:")
        print(f"      오디오 샘플: {info['num_audio_samples']}개")
        print(f"      메시 세그먼트: {info['num_mesh_segments']}개")
        print(f"      SEN 세그먼트: {info.get('num_sen_segments', 0)}개")
        if 'segment_info' in info:
            total_frames = sum(s.get('num_frames', 0) if isinstance(s, dict) else s for s in info['segment_info'].values())
            print(f"      총 프레임 수: {total_frames:,}")
    
    # Multiface 상세 분석
    if 'detailed_analysis' in multiface_result:
        print("\n  [상세 분석]")
        
        if 'summary' in multiface_result['detailed_analysis']:
            summary = multiface_result['detailed_analysis']['summary']
            print(f"\n    전체 통계:")
            print(f"      총 .obj 파일 수: {summary.get('total_obj_files', 'N/A'):,}")
            print(f"      총 Entity 수: {summary.get('total_entities', 'N/A')}")
            print(f"      총 오디오 샘플 수: {summary.get('total_audio_samples', 'N/A')}")
        
        if 'obj_file_samples' in multiface_result['detailed_analysis']:
            print(f"\n    .obj 파일 샘플 분석:")
            for i, sample in enumerate(multiface_result['detailed_analysis']['obj_file_samples'][:3]):
                print(f"      샘플 {i+1}:")
                print(f"        Vertices 수: {sample.get('num_vertices', 'N/A'):,}")
                print(f"        Faces 수: {sample.get('num_faces', 'N/A'):,}")
                if 'vertices_range' in sample:
                    v_range = sample['vertices_range']
                    print(f"        좌표 범위: X[{v_range['min'][0]:.3f}, {v_range['max'][0]:.3f}], "
                          f"Y[{v_range['min'][1]:.3f}, {v_range['max'][1]:.3f}], "
                          f"Z[{v_range['min'][2]:.3f}, {v_range['max'][2]:.3f}]")
        
        if 'vertices_statistics' in multiface_result['detailed_analysis']:
            stats = multiface_result['detailed_analysis']['vertices_statistics']
            print(f"\n    Vertices 통계:")
            print(f"      샘플 vertex 수: {stats.get('sample_vertex_counts', [])}")
            if stats.get('typical_vertex_count'):
                print(f"      일반적인 vertex 수: {stats['typical_vertex_count']:,}")
    
    print("\n" + "=" * 80)


def save_analysis_json(voca_result: Dict, multiface_result: Dict, output_path: str):
    """분석 결과를 JSON 파일로 저장"""
    result = {
        "voca": voca_result,
        "multiface": multiface_result
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n분석 결과가 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 구조 분석")
    parser.add_argument("--voca_root", type=str, 
                       default="/home/motion/omotion/datasets_raw/VOCA",
                       help="VOCA 데이터셋 루트 디렉토리")
    parser.add_argument("--multiface_root", type=str,
                       default="/home/motion/omotion/datasets_raw/multiface_data",
                       help="Multiface 데이터셋 루트 디렉토리")
    parser.add_argument("--output", type=str,
                       default="/home/motion/omotion/trainFramework/util/dataset_analysis.json",
                       help="분석 결과 JSON 출력 경로")
    
    args = parser.parse_args()
    
    print("VOCA 데이터셋 분석 중...")
    voca_result = analyze_voca_dataset(args.voca_root)
    
    print("Multiface 데이터셋 분석 중...")
    multiface_result = analyze_multiface_dataset(args.multiface_root)
    
    print_analysis_report(voca_result, multiface_result)
    save_analysis_json(voca_result, multiface_result, args.output)


if __name__ == "__main__":
    main()
