#!/usr/bin/env python3
"""
Multiface 데이터를 MeshTalk 형식으로 변환하는 스크립트
D2_meshtalk_new (7306 vertices) -> D2_meshtalk (6172 vertices)

MeshTalk의 multiface2meshtalk.py 로직을 기반으로 구현
"""

import os
import sys
import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

# MeshTalk에서 제거할 vertices 인덱스 (7306 -> 6172 변환)
# multiface2meshtalk.py에서 가져온 ignore_vertices 세트
IGNORE_VERTICES = {
    551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570,
    571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590,
    591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610,
    611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,
    631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650,
    651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670,
    671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690,
    691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710,
    711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730,
    731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
    751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770,
    771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 812, 813, 814, 815, 816, 817, 818, 819, 820,
    821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840,
    841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860,
    861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880,
    881, 882, 883, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 1145, 1146, 1147, 1148,
    1149, 1150, 1151, 1152, 1156, 1157, 1158, 1159, 1160, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200,
    1201, 1202, 1203, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1299, 1514, 1657, 1658,
    1659, 1660, 1662, 1663, 1664, 2010, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241,
    2242, 2243, 2244, 2245, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480,
    2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 3102, 3103, 3104, 3105, 3106,
    3107, 3108, 3109, 3163, 3164, 3165, 3166, 3167, 3169, 3170, 3171, 3172, 3173, 3174, 3176, 3177, 3178,
    3179, 3200, 3205, 3206, 3207, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257,
    3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274,
    3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291,
    3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308,
    3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325,
    3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3339, 3343, 3351, 3355, 3356,
    3364, 3365, 3374, 3384, 3385, 3386, 3387, 3388, 3389, 3418, 3419, 3420, 3422, 3423, 3424, 3425, 3426,
    3427, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444,
    3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476,
    3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493,
    3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 4651, 4652, 4653, 4654, 4655,
    4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672,
    4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689,
    4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706,
    4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723,
    4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740,
    4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757,
    4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774,
    4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791,
    4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808,
    4809, 4810, 4811, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4824, 4826, 4827,
    4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4849, 4850, 4851, 4852, 4855, 4856,
    4858, 4859, 4860, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4875, 4876, 4877, 4878, 4879,
    4880, 4881, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895, 4900, 4901, 4902, 4903, 4904, 4905, 4906,
    4907, 4908, 4909, 4910, 4914, 4915, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975,
    4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992,
    4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009,
    5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022, 5023, 5024, 5025, 5026,
    5027, 5028, 5029, 5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5040, 5041, 5042, 5043,
    5044, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060,
    5061, 5062, 5063, 5064, 5065, 5066, 5067, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5076, 5077,
    5078, 5079, 5080, 5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094,
    5095, 5096, 5097, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5106, 5107, 5108, 5109, 5110, 5111,
    5112, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5128,
    5129, 5130, 5131, 5132, 5133, 5134, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 5143, 5144, 5145,
    5146, 5149, 5150, 5153, 5154, 5155, 5156, 5159, 5162, 5163, 5166, 5167, 5170, 5187, 5188, 5189, 5190,
    5191, 5192, 5193, 5194, 5195, 5196, 5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5207,
    5208, 5209, 5210, 5211, 5212, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 5220, 5221, 5222, 5223, 5224,
    5225, 5226, 5227, 5228, 5229, 5230, 5231, 5232, 5233, 5235, 5265, 5266, 5267, 5268, 5269, 5270, 5272,
    5274, 5276, 5537, 5538, 5582, 5583, 5588, 6439, 6440, 6441, 6442, 6443, 6958, 6959, 6960, 6977, 6978,
    6980, 6981, 6982, 6983, 6984, 6985, 6986, 6987, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996,
    6997, 6998, 6999, 7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011
}

# 유효한 vertices 인덱스 (7306개 중에서 ignore_vertices를 제외한 6172개)
VALID_VERTICES = [i for i in range(7306) if i not in IGNORE_VERTICES]


def convert_7306_to_6172(vertices_7306: np.ndarray) -> np.ndarray:
    """
    7306 vertices를 6172 vertices로 변환
    
    Args:
        vertices_7306: (T, 7306, 3) 또는 (7306, 3) 형태의 vertices 배열
    
    Returns:
        vertices_6172: (T, 6172, 3) 또는 (6172, 3) 형태의 vertices 배열
    """
    if len(vertices_7306.shape) == 2:
        # (7306, 3) -> (6172, 3)
        return vertices_7306[VALID_VERTICES, :]
    elif len(vertices_7306.shape) == 3:
        # (T, 7306, 3) -> (T, 6172, 3)
        return vertices_7306[:, VALID_VERTICES, :]
    else:
        raise ValueError(f"Unexpected vertices shape: {vertices_7306.shape}")


def convert_meshtalk_new_to_meshtalk(
    source_dir: str,
    target_dir: str,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    D2_meshtalk_new (7306 vertices)를 D2_meshtalk (6172 vertices)로 변환
    
    Args:
        source_dir: D2_meshtalk_new 디렉토리 경로
        target_dir: D2_meshtalk 출력 디렉토리 경로
        dry_run: True면 실제 변환 없이 검증만 수행
    
    Returns:
        (변환된 샘플 수, 실패한 샘플 수)
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # JSON 파일 로드
    json_files = ['train.json', 'val.json', 'test.json']
    all_samples = []
    
    for json_file in json_files:
        json_path = source_dir / json_file
        if not json_path.exists():
            print(f"Warning: {json_file} not found, skipping...")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for sample in data.get('data', []):
            sample['split'] = json_file.replace('.json', '')
            all_samples.append(sample)
    
    print(f"총 {len(all_samples)}개 샘플 발견")
    
    # 출력 디렉토리 생성
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    # id_template.npy 복사 (있는 경우)
    id_template_src = source_dir / 'id_template.npy'
    id_template_dst = target_dir / 'id_template.npy'
    if id_template_src.exists():
        if not dry_run:
            # id_template도 변환 필요 (7306 -> 6172)
            id_template_7306 = np.load(id_template_src)
            if id_template_7306.shape[0] == 7306:
                id_template_6172 = convert_7306_to_6172(id_template_7306)
                np.save(id_template_dst, id_template_6172)
                print(f"id_template.npy 변환 완료: {id_template_7306.shape} -> {id_template_6172.shape}")
            else:
                shutil.copy2(id_template_src, id_template_dst)
                print(f"id_template.npy 복사 완료 (vertices 수 변경 없음)")
    
    # pca.npz 복사 (있는 경우)
    pca_src = source_dir / 'pca.npz'
    pca_dst = target_dir / 'pca.npz'
    if pca_src.exists():
        if not dry_run:
            # PCA는 6172 vertices로 재계산해야 할 수 있지만, 일단 복사
            # 나중에 필요하면 재계산
            shutil.copy2(pca_src, pca_dst)
            print(f"pca.npz 복사 완료 (재계산 필요할 수 있음)")
    
    converted_count = 0
    failed_count = 0
    converted_samples = []
    
    # 각 샘플 변환
    for sample in tqdm(all_samples, desc="Converting samples"):
        split = sample['split']
        annot_path = sample['annot_path']
        audio_path = sample['audio_path']
        
        # 원본 파일 경로
        src_annot = source_dir / annot_path
        src_audio = source_dir / audio_path
        
        # 대상 파일 경로
        dst_annot = target_dir / annot_path
        dst_audio = target_dir / audio_path
        
        # 파일 존재 확인
        if not src_annot.exists():
            print(f"Warning: Annotation file not found: {src_annot}")
            failed_count += 1
            continue
        
        if not src_audio.exists():
            print(f"Warning: Audio file not found: {src_audio}")
            failed_count += 1
            continue
        
        # Vertices 변환
        try:
            vertices_7306 = np.load(src_annot)
            
            # Shape 확인
            if len(vertices_7306.shape) == 3 and vertices_7306.shape[1] == 7306:
                # (T, 7306, 3) -> (T, 6172, 3)
                vertices_6172 = convert_7306_to_6172(vertices_7306)
            elif len(vertices_7306.shape) == 2 and vertices_7306.shape[0] == 7306:
                # (7306, 3) -> (6172, 3)
                vertices_6172 = convert_7306_to_6172(vertices_7306)
            else:
                print(f"Warning: Unexpected shape {vertices_7306.shape} for {src_annot}")
                failed_count += 1
                continue
            
            # 변환된 vertices 저장
            if not dry_run:
                dst_annot.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst_annot, vertices_6172)
            
            # 오디오 파일 복사
            if not dry_run:
                dst_audio.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_audio, dst_audio)
            
            # 샘플 정보 업데이트 (annot_type 변경)
            converted_sample = sample.copy()
            converted_sample['annot_type'] = 'meshtalk_6172_vertices'
            converted_samples.append(converted_sample)
            converted_count += 1
            
        except Exception as e:
            print(f"Error converting {src_annot}: {e}")
            failed_count += 1
            continue
    
    # JSON 파일 생성
    if not dry_run:
        info = {
            "id_list": [f"id_{i:02d}" for i in range(13)],  # D2는 13명
            "data_source": "multiface",
            "total_duration": 0.0
        }
        
        # Split별로 샘플 분류
        for split in ['train', 'val', 'test']:
            split_samples = [s for s in converted_samples if s['split'] == split]
            
            json_data = {
                "info": info,
                "data": split_samples
            }
            
            json_path = target_dir / f"{split}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"{split}.json 생성 완료: {len(split_samples)}개 샘플")
    
    print(f"\n변환 완료: {converted_count}개 성공, {failed_count}개 실패")
    return converted_count, failed_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multiface 데이터를 MeshTalk 형식으로 변환 (7306 -> 6172 vertices)"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='./data/unitalker_data/D2_meshtalk_new',
        help='소스 디렉토리 (D2_meshtalk_new, 삭제됨 - 경로 지정 필요)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='./data/unitalker_data/D2_meshtalk',
        help='타겟 디렉토리 (D2_meshtalk)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 변환 없이 검증만 수행'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multiface -> MeshTalk 변환 (7306 -> 6172 vertices)")
    print("=" * 60)
    print(f"소스: {args.source}")
    print(f"타겟: {args.target}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    convert_meshtalk_new_to_meshtalk(
        args.source,
        args.target,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
