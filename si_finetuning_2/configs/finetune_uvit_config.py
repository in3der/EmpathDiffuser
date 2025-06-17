import ml_collections


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64
    config.data_type = 1
    config.mixed_precision = "fp16"
    config.fp16 = True
    config.ema_rate = 0.9999  # EMA 설정 추가

    config.emotion_weight = 0.1  # 감정 손실 가중치
    config.num_emotions = 7  # 감정 카테고리 수 (실제 데이터에 따라 조정)


    config.train = d(
        n_steps=500000,
        batch_size=64,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        #save_interval=50000,   # ckpt 저장 주기
        save_interval=2000,
        grad_clip=1.0,  # 그래디언트 클리핑 추가
        ema_rate=0.9999,
    )

    config.optimizer = d(
        name='adamw',
        # lr=0.0002,
        lr=3e-5,  # 250604
        weight_decay=0.03,
        #betas=(0.99, 0.99),
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        #warmup_steps=5000
        warmup_steps=500
    )

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim')
    )

    config.nnet = d(
        name='uvit_multi_post_ln_v1',
        # pretrained_path='/home/ivpl-d29/sichoi/Emo/unidiffuser/models/uvit_v1.pth',
        pretrained_path='/home/sichoi/unidiffuser/models/uvit_v1.pth',
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
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )

    config.dataset = ml_collections.ConfigDict({
        'name': 'avamerg',
        # 'train_json': "/home/ivpl-d29/dataset/AvaMERG/train_finetune.json",
        # 'test_json': "/home/ivpl-d29/dataset/AvaMERG/test_finetune.json",
        # 'image_root': '/home/ivpl-d29/dataset/AvaMERG/image_v5_0',  # 256x256
        'train_json': "/mnt/dataset/AvaMERG/train_finetune.json",
        'test_json': "/mnt/dataset/AvaMERG/test_finetune.json",
        'image_root': '/home/ivpl-d29/dataset/AvaMERG/image_v5_0',  # 256x256
        # 현재 npy 데이터 구성
        #         # [idx] input_img_latent shape: (4, 64, 64)
        #         # [idx] output_img_latent shape: (4, 64, 64)
        #         # [idx] input_clip_feat shape: (512,)
        #         # [idx] output_clip_feat shape: (512,)
        #         # [idx] input_text_latent shape: (77, 64)
        #         # [idx] output_text_latent shape: (77, 64)
        # 'train_npy_root': "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/feature64/train/",
        # 'test_npy_root': "/home/ivpl-d29/dataset/AvaMERG/si_finetuning_2/assets/feature64/val/",
        'train_npy_root': "/mnt/dataset/AvaMERG/si_finetuning_2/assets/feature64/train/",
        'test_npy_root': "/mnt/dataset/AvaMERG/si_finetuning_2/assets/feature64/val/",
        'p_uncond': 0.15,
    })

    #config.sample = d(
    #    sample_steps=50,
    #    n_samples=50000,
    #    mini_batch_size=50,
    #    algorithm='dpm_solver',
    #    cfg=True,
    #    scale=0.7,
    #    path=''
    #)

    return config