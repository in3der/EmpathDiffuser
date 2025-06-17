import os
import numpy as np
from tqdm import tqdm
"""
train/0_input_img.npy, train/0_output_img.npy, ... → 병합되어 features2_merged/train/0.npy로 저장하는 코드
변환된 .npy 구조
{
    'input_img_latent': np.ndarray,
    'output_img_latent': np.ndarray,
    'input_clip_feat': np.ndarray,
    'output_clip_feat': np.ndarray,
    'input_text_latent': np.ndarray,
    'output_text_latent': np.ndarray,
}
"""
def merge_npy_files(old_dir, new_dir):
    os.makedirs(new_dir, exist_ok=True)

    # 병합 대상 파일 이름 패턴 추정
    # 예: 0_input_img.npy, 0_output_img.npy ...
    files = [f for f in os.listdir(old_dir) if f.endswith(".npy")]
    idx_set = sorted(set(f.split('_')[0] for f in files if f[0].isdigit()))

    for idx in tqdm(idx_set):
        try:
            data_dict = {
                'input_img_latent': np.load(os.path.join(old_dir, f"{idx}_input_img.npy")),
                'output_img_latent': np.load(os.path.join(old_dir, f"{idx}_output_img.npy")),
                'input_clip_feat': np.load(os.path.join(old_dir, f"{idx}_input_clip_img.npy")),
                'output_clip_feat': np.load(os.path.join(old_dir, f"{idx}_output_clip_img.npy")),
                'input_text_latent': np.load(os.path.join(old_dir, f"{idx}_input_text.npy")),
                'output_text_latent': np.load(os.path.join(old_dir, f"{idx}_output_text.npy")),
            }

            # 저장
            np.save(os.path.join(new_dir, f"{idx}.npy"), data_dict)

        except Exception as e:
            print(f"[오류] {idx}번 예제 병합 중 오류 발생: {e}")

# 경로 설정
old_feature_dir = "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features2/train"  # 또는 val
merged_save_dir = "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features2_dict/train"  # 또는 val
old_feature_dir2 = "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features2/val"  # 또는 val
merged_save_dir2 = "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features2_dict/val"  # 또는 val
merge_npy_files(old_feature_dir, merged_save_dir)
merge_npy_files(old_feature_dir2, merged_save_dir2)
