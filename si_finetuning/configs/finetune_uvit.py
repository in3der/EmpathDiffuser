import ml_collections
import libs.clip
import sys
import os
sys.path.append(os.path.abspath('/home/ivpl-d29/sichoi/Emo/unidiffuser/si_finetuning'))
import clip
import torch.nn as nn
device = "cuda"

clip_img_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim')
    )

    config.train = d(
        n_steps=500000,
        #batch_size=1024,
        batch_size=2,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_multi_post_ln_v1',
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),  # 64
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),  # 512
        use_checkpoint=True
    )

    # ⭐CLIP 텍스트 인코더 로딩 (고정된 사전학습 모델)
    clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()
    clip_text_model.to(device)

    # ⭐Autoencoder 모델 로딩 (이미지 latent 추출용)
    autoencoder = libs.autoencoder.get_model('/home/ivpl-d29/sichoi/Emo/unidiffuser/models/autoencoder_kl.pth')
    autoencoder.to(device)

    # ⭐CLIP 이미지 인코더 로딩 (OpenAI의 CLIP 모델)
    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    linear_proj = nn.Linear(768, config.text_dim).to(device)  # 768 → 64 projection

    config.dataset = d(
        name='avamerg',
        train_json="/home/ivpl-d29/dataset/AvaMERG/train_finetune.json",
        test_json="/home/ivpl-d29/dataset/AvaMERG/test_finetune.json",
        image_root='/home/ivpl-d29/dataset/AvaMERG/image_v5_0_64',
        #cfg=True,
        p_uncond=0.15,
        #clip_preprocess=clip_preprocess  # ★ 이걸 전달해야 한다
        autoencoder=autoencoder,
        clip_img_model=clip_img_model,
        clip_text_model=clip_text_model,
        linear_proj=linear_proj,
        clip_preprocess=clip_img_model_preprocess,
        device='cuda',
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.7,
        path=''
    )

    return config
