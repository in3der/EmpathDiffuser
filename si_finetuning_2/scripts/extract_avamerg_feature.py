import sys
import os
# 'libs' 폴더가 있는 상위 디렉토리 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from tqdm import tqdm
import libs.autoencoder
import libs.clip
from si_finetuning_2.finetune_datasets import AvaMERGDatabase
import clip as openai_clip  # OpenAI CLIP


def setup(rank, world_size):
    """Initialize the process group for distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the process group."""
    dist.destroy_process_group()

import torch
def extract_features_dataparallel(args, resolution=512):
    import torch
    """Alternative implementation using DataParallel (simpler but less efficient)."""
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

    # Prepare dataset and paths
    if args.split == "train":
        json_path = "/home/ivpl-d29/dataset/AvaMERG/train_finetune.json"
        save_dir = f'/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features_moments/train'
    elif args.split == "val":
        json_path = "/home/ivpl-d29/dataset/AvaMERG/test_finetune.json"
        save_dir = f'/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/features_moments/val'
    else:
        raise NotImplementedError("Unknown split!")

    dataset = AvaMERGDatabase(json_path=json_path,
                              image_root="/home/ivpl-d29/dataset/AvaMERG/image_v5_0",
                              size=resolution)

    os.makedirs(save_dir, exist_ok=True)

    device = "cuda"

    # Initialize models with DataParallel
    autoencoder = libs.autoencoder.get_model(
        '/home/ivpl-d29/sichoi/Emo/unidiffuser/si_finetuning/assets/stable-diffusion/autoencoder_kl.pth')
    if torch.cuda.device_count() > 1:
        autoencoder = nn.DataParallel(autoencoder)
    autoencoder = autoencoder.to(device)

    clip_text = libs.clip.FrozenCLIPEmbedder().eval()
    if torch.cuda.device_count() > 1:
        clip_text = nn.DataParallel(clip_text)
    clip_text = clip_text.to(device)

    # OpenAI CLIP for image encoding
    clip_img_model, clip_img_preprocess = openai_clip.load("ViT-B/32", device=device, jit=False)
    clip_img_model.eval()
    if torch.cuda.device_count() > 1:
        clip_img_model = nn.DataParallel(clip_img_model)

    # Create DataLoader - batch_size=1로 통일
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(dataloader)):
            input_imgs, input_texts, output_imgs, output_texts = batch_data

            # Process batch
            for i in range(len(input_imgs)):
                idx = batch_idx * len(input_imgs) + i

                def encode_image_autoencoder(image):
                    image_tensor = torch.tensor(image, device=device).unsqueeze(0)
                    if hasattr(autoencoder, 'module'):
                        latent = autoencoder.module(image_tensor, fn='encode')
                    else:
                        latent = autoencoder(image_tensor, fn='encode')
                    return latent.squeeze(0).detach().cpu().numpy()

                def encode_text(text):
                    if hasattr(clip_text, 'module'):
                        latent = clip_text.module.encode([text])[0].detach().cpu().numpy()
                    else:
                        latent = clip_text.encode([text])[0].detach().cpu().numpy()
                    #return latent[:, :, :64]  # Project to (1, 77, 64)
                    return latent

                # def encode_image_clip(image):
                #     clip_tensor = clip_img_preprocess(image).unsqueeze(0).to(device)
                #     if hasattr(clip_img_model, 'module'):
                #         latent = clip_img_model.module.encode_image(clip_tensor)
                #     else:
                #         latent = clip_img_model.encode_image(clip_tensor)
                #     return latent.squeeze(0).detach().cpu().numpy()
                from PIL import Image
                import numpy as np
                import torch

                def encode_image_clip(image):
                    # 이미지가 Tensor인 경우 NumPy로 변환 (CHW → HWC)
                    if isinstance(image, torch.Tensor):
                        image = image.permute(1, 2, 0).cpu().numpy()
                    # 이미지가 NumPy일 경우 PIL로 변환
                    if isinstance(image, np.ndarray):
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image)

                    # CLIP 전처리 및 인코딩
                    clip_tensor = clip_img_preprocess(image).unsqueeze(0).to(device)
                    if hasattr(clip_img_model, 'module'):
                        latent = clip_img_model.module.encode_image(clip_tensor)
                    else:
                        latent = clip_img_model.encode_image(clip_tensor)
                    return latent.squeeze(0).detach().cpu().numpy()

                # Encode all features
                input_img_latent = encode_image_autoencoder(input_imgs[i])
                output_img_latent = encode_image_autoencoder(output_imgs[i])
                input_text_latent = encode_text(input_texts[i])
                output_text_latent = encode_text(output_texts[i])
                input_clip_feat = encode_image_clip(input_imgs[i])
                output_clip_feat = encode_image_clip(output_imgs[i])

                print(f"[{idx}] input_img_latent shape: {input_img_latent.shape}")
                print(f"[{idx}] output_img_latent shape: {output_img_latent.shape}")
                print(f"[{idx}] input_clip_feat shape: {input_clip_feat.shape}")
                print(f"[{idx}] output_clip_feat shape: {output_clip_feat.shape}")
                print(f"[{idx}] input_text_latent shape: {input_text_latent.shape}")
                print(f"[{idx}] output_text_latent shape: {output_text_latent.shape}")

                # Save all features
                np.save(os.path.join(save_dir, f'{idx}_input_img.npy'), input_img_latent)
                np.save(os.path.join(save_dir, f'{idx}_output_img.npy'), output_img_latent)
                np.save(os.path.join(save_dir, f'{idx}_input_clip_img.npy'), input_clip_feat)
                np.save(os.path.join(save_dir, f'{idx}_output_clip_img.npy'), output_clip_feat)
                np.save(os.path.join(save_dir, f'{idx}_input_text.npy'), input_text_latent)
                np.save(os.path.join(save_dir, f'{idx}_output_text.npy'), output_text_latent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--resolution', default=512, type=int)
    args = parser.parse_args()
    print(args)


    extract_features_dataparallel(args, args.resolution)


if __name__ == '__main__':
    main()