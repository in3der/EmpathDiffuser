import ml_collections
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
import clip
from PIL import Image
import time


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder):
    import einops
    import numpy as np
    import torch
    import utils  # 유틸 함수에 center_crop 함수가 있어야 함
    from PIL import Image

    resolution = config.z_shape[-1] * 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_img_feature(image):
        image = np.array(image).astype(np.uint8)
        image = utils.center_crop(resolution, resolution, image)

        clip_img_feature = clip_img_model.encode_image(
            clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        )

        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> 1 c h w')
        image = torch.tensor(image, device=device)

        moments = autoencoder.encode_moments(image)
        return clip_img_feature, moments

    # 초기값 (fallback 용도)
    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2]).to(device)
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim).to(device)

    if config.mode in ['t2i', 't2i2t']:
        prompts = [config.prompt] * config.n_samples
        contexts = clip_text_model.encode(prompts)

    elif config.mode in ['i2t', 'i2t2i']:
        image = Image.open(config.img).convert('RGB')
        clip_img, img_context = get_img_feature(image)
        img_contexts = img_context.repeat(config.n_samples, 1, 1, 1)
        clip_imgs = clip_img.repeat(config.n_samples, 1, 1)

    elif config.mode == 'joint':
        prompts = [config.prompt] * config.n_samples
        contexts = clip_text_model.encode(prompts)

        image = Image.open(config.img).convert('RGB')
        clip_img, img_context = get_img_feature(image)

        img_contexts = img_context.repeat(config.n_samples, 1, 1, 1)
        clip_imgs = clip_img.repeat(config.n_samples, 1, 1)

    return contexts, img_contexts, clip_imgs





def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):        # 재현 가능한 결과를 위해 랜덤 시드 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(config):
    # PyTorch의 CUDNN 최적화 설정 (성능 향상을 위한 선택적 설정)
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True  # 입력 크기가 고정일 때 성능 최적화
        torch.backends.cudnn.deterministic = False  # 결과 재현성보다 속도 우선

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.seed)

    # 설정을 불변으로 고정 (수정 방지)
    config = ml_collections.FrozenConfigDict(config)

    # 로깅 초기화
    utils.set_logger(log_level='info')

    # 스케줄 함수로 beta 값 초기화 (diffusion noise level 조절용)
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)  # 타임스텝 개수

    # ⭐메인 모델 로딩 (U-ViT)
    nnet = utils.get_nnet(**config.nnet)    # libs. name='uvit_multi_post_ln_v1',
    logging.info(f'load nnet from {config.nnet_path}')

    # U-ViT에 저장된 모델 가중치 로드
    #nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    # checkpoint에서 'model' 키만 꺼냄
    checkpoint = torch.load(config.nnet_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)


    USE_LINEAR_PROJ = True  # ⭐ 이 값을 False로 바꾸면 linear_proj 안 씀

    def strip_prefix_if_present(state_dict, prefix="uvit."):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    # prefix strip (자동 판단)
    if any(k.startswith('uvit.') for k in state_dict.keys()):
        state_dict = strip_prefix_if_present(state_dict, prefix='uvit.')

    # linear_proj 따로 pop
    linear_proj_state = {
        'weight': state_dict.pop('linear_proj.weight', None),
        'bias': state_dict.pop('linear_proj.bias', None)
    }


    # token_embedding 크기 맞추기 (학습 시 사용했던 embedding 구조와 동일하게 맞춰줌)
    # old_token_emb = state_dict.get('token_embedding.weight')  # 체크포인트에서 기존 token_embedding.weight 불러오기
    # print("🥹🥹🥹🥹🥹🥹🥹🥹🥹🥹🥹", old_token_emb.shape)
    # if old_token_emb is not None:
    #     new_shape = nnet.state_dict()['token_embedding.weight'].shape  # 현재 모델의 token_embedding 기대 shape 확인
    #     print("new_shape : ", new_shape)
    #
    #     # 새로운 embedding weight 텐서 초기화 (모델이 기대하는 shape으로)
    #     new_token_emb = torch.zeros(new_shape)
    #
    #     # 가능한 범위 내에서 기존 임베딩 복사 (예: old = [2, 1536], new = [3, 1536] → 2개 복사)
    #     n_copy = min(old_token_emb.shape[0], new_shape[0])
    #     new_token_emb[:n_copy] = old_token_emb[:n_copy]
    #
    #     # 복사 후 남은 부분은 랜덤 초기화 (학습 시 방식과 동일, std=0.02)
    #     if new_shape[0] > n_copy:
    #         new_token_emb[n_copy:] = torch.randn(new_shape[0] - n_copy, new_shape[1]) * 0.02
    #
    #     # 수정된 token_embedding으로 state_dict 갱신
    #     state_dict['token_embedding.weight'] = new_token_emb
    #     print("new_token_emb : ", new_token_emb)
    # token개수 3으로 맞춘 버전
    # token_embedding 크기 맞추기 (checkpoint에서는 [3, 1536], 현재 모델은 [2, 1536]을 기대할 때)
    # token_embedding 크기 확인 (필요 시 resize)
    old_token_emb = state_dict.get('token_embedding.weight')
    expected_shape = nnet.token_embedding.weight.shape

    if old_token_emb is not None and old_token_emb.shape != expected_shape:
        print(f"token_embedding 크기 mismatch: {old_token_emb.shape} -> {expected_shape}")
        new_token_emb = torch.zeros(expected_shape)
        n_copy = min(old_token_emb.shape[0], expected_shape[0])
        new_token_emb[:n_copy] = old_token_emb[:n_copy]
        if expected_shape[0] > n_copy:
            new_token_emb[n_copy:] = torch.randn(expected_shape[0] - n_copy, expected_shape[1]) * 0.02
        state_dict['token_embedding.weight'] = new_token_emb

    # 모델에 로드
    try:
        nnet.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("Error loading state dict:", e)

    # # prefix 문제 처리 (필요 시)
    # def strip_prefix_if_present(state_dict, prefix="uvit."):
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith(prefix):
    #             new_key = k[len(prefix):]
    #         else:
    #             new_key = k
    #         new_state_dict[new_key] = v
    #     return new_state_dict
    #
    # state_dict = strip_prefix_if_present(state_dict, prefix="uvit.")
    #
    # # linear_proj weight/bias는 따로 분리 (nnet에 포함시키면 안됨)
    # linear_proj_state = {
    #     'weight': state_dict.pop('linear_proj.weight', None),
    #     'bias': state_dict.pop('linear_proj.bias', None)
    # }
    #
    # # token_embedding 크기 맞추기
    # old_token_emb = state_dict.get('token_embedding.weight')
    # if old_token_emb is not None:
    #     new_shape = nnet.state_dict()['token_embedding.weight'].shape
    #
    #     def resize_token_embedding(old_tensor, new_shape, mean_resizing=True):
    #         if old_tensor.shape == new_shape:
    #             return old_tensor
    #         if mean_resizing:
    #             mean = old_tensor.mean(dim=0, keepdim=True)
    #             return mean.repeat(new_shape[0], 1)
    #         else:
    #             return torch.zeros(new_shape)
    #
    #     state_dict['token_embedding.weight'] = resize_token_embedding(old_token_emb, new_shape)
    #
    # # U-ViT에 state_dict 로드
    # try:
    #     nnet.load_state_dict(state_dict, strict=True)
    # except RuntimeError as e:
    #     print("Error loading state dict:", e)

    nnet.to(device)
    nnet.eval()

    # ⭐ linear_proj 선택적으로 사용
    if USE_LINEAR_PROJ:
        linear_proj = torch.nn.Linear(768, 64)
        if linear_proj_state['weight'] is not None and linear_proj_state['bias'] is not None:
            linear_proj.load_state_dict(linear_proj_state)
            print("✅ linear_proj loaded")
        else:
            print("⚠️ linear_proj weights not found, using randomly initialized layer")
        linear_proj.to(device)
        linear_proj.eval()
    else:
        linear_proj = None
        print("ℹ️ linear_proj disabled")

    ## 필요한 경우 shape mismatch 처리도 가능
    #nnet.load_state_dict(state_dict, strict=False)
#
    ## ⭐텍스트 임베딩 프로젝션 (학습 시 사용했던 것과 동일한 구조)
    #linear_proj = torch.nn.Linear(768, 64)
    #linear_proj.load_state_dict({  # 체크포인트에서 weight 불러오기
    #    'weight': state_dict['linear_proj.weight'],
    #    'bias': state_dict['linear_proj.bias']
    #})
    #linear_proj.to(device)
    #linear_proj.eval()
#
    #nnet.to(device)
    #nnet.eval()

    # 캡션 디코더 사용 여부 결정
    # 조건: 텍스트 차원이 작거나, 텍스트→이미지(T2I) 모드가 아닌 경우
    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
    if use_caption_decoder:
        # 캡션 디코더 초기화 (이미지 → 텍스트 등에서 사용) - 텍스트 생성하는 디코더
        from libs.caption_decoder import CaptionDecoder
        caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    else:
        caption_decoder = None  # 사용 안함

    # ⭐CLIP 텍스트 인코더 로딩 (고정된 사전학습 모델)
    clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()
    clip_text_model.to(device)

    # ⭐Autoencoder 모델 로딩 (이미지 latent 추출용)
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    # ⭐CLIP 이미지 인코더 로딩 (OpenAI의 CLIP 모델)
    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # 빈 텍스트(공백 문자열)를 인코딩한 context (예: unconditional generation 시 사용)
    empty_context = clip_text_model.encode([''])[0]


    def split(x):   # 벡터 분해 함수
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)      # x 벡터를 latent 이미지 z 와 clip 이미지 임베딩으로 분할
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)      # 벡터를 이미지 텐서로 변형: [B, C*H*W] → [B, C, H, W]
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)       # clip 임베딩을 [B, L*D] → [B, L, D] 로 재구성 (L=1)
        return z, clip_img


    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')     # 이미지 latent z 를 [B, C, H, W] → [B, C*H*W]로 flatten
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')       # clip 임베딩을 [B, L, D] → [B, L*D]로 flatten
        return torch.concat([z, clip_img], dim=-1)      # 두 텐서를 하나의 벡터로 합치기


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        텍스트 → 이미지 변환을 위한 조건부 모델 함수
        Classifier-Free Guidance (CFG) 방식으로 조건부 + 무조건부 출력을 혼합하여 생성 성능 향상
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)  # 입력을 분할 (latent 이미지, clip 임베딩) , x는 noisy한 image latent + img clip embedding

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device) # 텍스트 타임스텝 (text는 고정되어 있으므로 0으로 설정)

        z_out, clip_img_out, text_out = nnet(   # 조건부 (text-conditioned) 모델 출력
            z, clip_img,    # latent img, clip 임베딩을 입력으로
            text=text,
            t_img=timesteps,
            t_text=t_text,
            data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type
        )
        x_out = combine(z_out, clip_img_out)  # [B, latent + clip_emb]로 다시 결합

        # guidance scale이 0이면 조건부 출력만 사용
        if config.sample.scale == 0.:
            return x_out

        # 무조건부 출력 계산 방식 선택
        if config.sample.t2i_cfg_mode == 'empty_token':
            # 방법 1: 비어있는 텍스트 임베딩 사용 (기존 CFG 방식)
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)

            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(
                z, clip_img,
                text=_empty_context,
                t_img=timesteps,
                t_text=t_text,
                data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type
            )
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)

        elif config.sample.t2i_cfg_mode == 'true_uncond':
            # 방법 2: 학습된 무조건부 모델 사용
            text_N = torch.randn_like(text)  # 가우시안 노이즈로 텍스트 대체
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(
                z, clip_img,
                text=text_N,
                t_img=timesteps,
                t_text=torch.ones_like(timesteps) * N,  # 최종 timestep 사용
                data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type
            )
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)

        else:
            raise NotImplementedError  # 정의되지 않은 방식

        # CFG를 위한 최종 출력 혼합
        return x_out + config.sample.scale * (x_out - x_out_uncond)


    def i_nnet(x, timesteps):
        """
        이미지 기반으로 텍스트 임베딩(latent) 생성
        """
        # 입력 x를 이미지 latent(z) 와 clip 이미지 임베딩으로 분할
        z, clip_img = split(x)

        # 무작위 텍스트 임베딩 생성: 이미지에서 텍스트를 예측하도록 학습
        text = torch.randn(x.size(0), 77, config.text_dim, device=device)

        # 텍스트 예측용 타임스텝 (최종 단계인 N 사용)
        t_text = torch.ones_like(timesteps) * N

        # nnet: 이미지 latent, clip 임베딩, 랜덤 텍스트를 조건으로 예측 수행
        z_out, clip_img_out, text_out = nnet(
            z, clip_img, text=text,
            t_img=timesteps,
            t_text=t_text,
            data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type
        )

        # z와 clip_img를 다시 하나의 벡터로 결합하여 출력
        x_out = combine(z_out, clip_img_out)
        return x_out

    def t_nnet(x, timesteps):
        """
        text latent로부터 이미지 복원
        """
        # 무작위 이미지 latent 와 clip 이미지 임베딩 생성
        z = torch.randn(x.size(0), *config.z_shape, device=device)
        clip_img = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)

        # nnet 호출: 텍스트 임베딩(x)을 조건으로 이미지 복원 예측
        z_out, clip_img_out, text_out = nnet(
            z, clip_img, text=x,
            t_img=torch.ones_like(timesteps) * N,  # 이미지는 최종 timestep
            t_text=timesteps,  # 텍스트는 현재 timestep
            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type
        )

        return text_out  # 텍스트 예측 결과만 반환

    def i2t_nnet(x, timesteps, z, clip_img):
        """
        이미지 latent와 clip 임베딩을 조건으로, 텍스트 latent를 예측
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        # 이미지 쪽은 timestep 0에서 시작
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        # 조건부 예측: 이미지(z + clip_img)를 조건으로 텍스트(x) 복원
        z_out, clip_img_out, text_out = nnet(
            z, clip_img, text=x,
            t_img=t_img,
            t_text=timesteps,
            data_type=torch.zeros_like(t_img, device=device, dtype=torch.int) + config.data_type
        )

        # scale이 0이면 unconditional 없이 종료
        if config.sample.scale == 0.:
            return text_out

        # Unconditional 예측용: 무작위 z, clip_img로 학습된 진짜 unconditional 모델 사용
        z_N = torch.randn_like(z)
        clip_img_N = torch.randn_like(clip_img)

        z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(
            z_N, clip_img_N, text=x,
            t_img=torch.ones_like(timesteps) * N,
            t_text=timesteps,
            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type
        )

        # Classifier-free Guidance
        return text_out + config.sample.scale * (text_out - text_out_uncond)

    def split_joint(x):
        """
        하나의 결합 벡터를 z (이미지 latent), clip 이미지 임베딩, 텍스트 latent로 분리
        """
        # config.z_shape = (C, H, W)
        C, H, W = config.z_shape
        z_dim = C * H * W

        # x는 z + clip_img + text가 모두 concat된 상태
        z, clip_img, text = x.split(
            [z_dim, config.clip_img_dim, 77 * config.text_dim],
            dim=1
        )

        # 차원 복원
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=config.text_dim)

        return z, clip_img, text

    def combine_joint(z, clip_img, text):
        """z, clip 이미지 임베딩, 텍스트 latent를 하나의 벡터로 결합"""
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, clip_img, text], dim=-1)

    def joint_nnet(x, timesteps):
        """텍스트 + 이미지 정보가 모두 포함된 latent 벡터를 입력으로 받아, 세 가지 정보(z, clip_img, text)를 모두 예측하고 Classifier-Free Guidance로 조정"""
        # z, clip 이미지, 텍스트 latent로 분리
        z, clip_img, text = split_joint(x)

        # 조건부 예측
        z_out, clip_img_out, text_out = nnet(
            z, clip_img, text=text,
            t_img=timesteps,
            t_text=timesteps,
            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type
        )

        # 결과를 다시 하나의 벡터로 결합
        x_out = combine_joint(z_out, clip_img_out, text_out)

        # scale = 0이면 guidance 없이 return
        if config.sample.scale == 0.:
            return x_out

        # --- 아래부터는 Classifier-Free Guidance (CFG) 처리 --- #

        # 조건 없는 latent 샘플 생성 (text 유지, z/clip_img 랜덤)
        z_noise = torch.randn(x.size(0), *config.z_shape, device=device)
        clip_img_noise = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)
        text_noise = torch.randn(x.size(0), 77, config.text_dim, device=device)

        # 첫 번째 Uncond: 텍스트만 condition, z/clip은 랜덤 (text 유지)
        _, _, text_out_uncond = nnet(
            z_noise, clip_img_noise, text=text,
            t_img=torch.ones_like(timesteps) * N,
            t_text=timesteps,
            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type
        )

        # 두 번째 Uncond: z/clip은 유지, 텍스트는 랜덤
        z_out_uncond, clip_img_out_uncond, _ = nnet(
            z, clip_img, text=text_noise,
            t_img=timesteps,
            t_text=torch.ones_like(timesteps) * N,
            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type
        )

        # 두 개의 Uncond 출력을 결합
        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        # 최종: CFG로 guidance 적용
        return x_out + config.sample.scale * (x_out - x_out_uncond)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        """
        Autoencoder를 사용해 이미지 배치를 latent로 인코딩
        AMP(auto mixed precision) 적용으로 성능 최적화
        """
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        """
        latent 공간의 벡터를 이미지로 복원
        AMP(auto mixed precision) 적용
        """
        return autoencoder.decode(_batch)

    logging.info(config.sample)
    logging.info(f'N={N}')
    """텍스트/이미지 기반 입력에 따라 필요한 context 벡터들을 생성:
    contexts: 텍스트 CLIP 임베딩 (77 x 768)
    img_contexts: 이미지의 VAE 인코딩된 moment
    clip_imgs: CLIP 이미지 임베딩"""
    contexts, img_contexts, clip_imgs = prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder)

    contexts = contexts  # the clip embedding of conditioned texts
    """contexts_low_dim: 네트워크 입력용으로 차원을 줄인 텍스트 context
    caption_decoder가 활성화된 경우, CLIP 텍스트 임베딩을 text_dim에 맞춰 변환"""
    #contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet
    # linear project 불러왔기 때문에 수정함
    if linear_proj is not None:
        contexts_low_dim = linear_proj(contexts)  # [B, 77, 768] → [B, 77, 64]

    img_contexts = img_contexts  # img_contexts is the autoencoder moment
    z_img = autoencoder.sample(img_contexts)
    clip_imgs = clip_imgs  # the clip embedding of conditioned image

    if config.mode in ['t2i', 't2i2t']:
        _n_samples = contexts_low_dim.size(0)   # 텍스트 기반이면 텍스트 context 개수 기준
    elif config.mode in ['i2t', 'i2t2i']:
        _n_samples = img_contexts.size(0)   # 이미지 기반이면 이미지 context 기준
    else:
        _n_samples = config.n_samples   # 아니면 기본 설정값(config.n_samples) 사용


    def sample_fn(mode, **kwargs):
        # 초기 노이즈 텐서 생성: 샘플 수만큼 z latent, clip 이미지 임베딩, 텍스트 임베딩 초기화
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)  # VAE latent 공간 노이즈
        _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)  # CLIP 이미지 임베딩 노이즈
        _text_init = torch.randn(_n_samples, 77, config.text_dim, device=device)  # 텍스트 임베딩 노이즈 (77은 CLIP 토큰 수)
        # 선택 모드에 따라 초기 입력 벡터를 구성
        if mode == 'joint':
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        elif mode in ['t2i', 'i']:
            _x_init = combine(_z_init, _clip_img_init)
        elif mode in ['i2t', 't']:
            _x_init = _text_init
        # 베타 스케줄 기반 노이즈 스케줄러 설정 (DPM-Solver에서 사용)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        # 디퓨전 모델 함수 정의: 시간 t에 따라 적절한 네트워크 함수 호출
        def model_fn(x, t_continuous):
            t = t_continuous * N  # 연속 시간값을 정수 timestep으로 변환
            if mode == 'joint':
                return joint_nnet(x, t)
            elif mode == 't2i':
                return t2i_nnet(x, t, **kwargs)
            elif mode == 'i2t':
                return i2t_nnet(x, t, **kwargs)
            elif mode == 'i':
                return i_nnet(x, t)
            elif mode == 't':
                return t_nnet(x, t)

        # DPM-Solver 객체 생성: 고속 샘플링을 위한 디퓨전 샘플러
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                start_time = time.time()
                """
                x = dpm_solver.sample(
                    _x_init,                               # 초기 노이즈
                    steps=config.sample.sample_steps,     # 샘플링 스텝 수
                    eps=1. / N,                            # 초기 시간
                    T=1.                                   # 최종 시간
                )
                """
                x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

        os.makedirs(config.output_path, exist_ok=True)
        if mode == 'joint':
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
        elif mode in ['t2i', 'i']:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ['i2t', 't']:
            return x

    def watermarking(save_path):
        img_pre = Image.open(save_path)
        img_pos = utils.add_water(img_pre)
        img_pos.save(save_path)

    if config.mode in ['joint']:
        _z, _clip_img, _text = sample_fn(config.mode)  # 샘플링 함수 호출해서 latent 변수들 생성
        samples = unpreprocess(decode(_z))  # VAE 디코딩 후 이미지 형태로 변환
        prompts = caption_decoder.generate_captions(_text)  # 생성된 텍스트 임베딩으로 캡션 생성

        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)
        # 생성된 프롬프트들을 텍스트 파일로 저장
        with open(os.path.join(config.output_path, config.mode, 'prompts.txt'), 'w') as f:
            print('\n'.join(prompts), file=f)

        # 이미지 파일들을 하나씩 저장하고 워터마크 추가
        for idx, sample in enumerate(samples):
            save_path = os.path.join(config.output_path, config.mode, f'{idx}.png')
            save_image(sample, save_path)
            watermarking(save_path)

    elif config.mode in ['t2i', 'i', 'i2t2i']:
        if config.mode == 't2i':
            _z, _clip_img = sample_fn(config.mode, text=contexts_low_dim)  # conditioned on the text embedding
        elif config.mode == 'i':
            _z, _clip_img = sample_fn(config.mode)
        elif config.mode == 'i2t2i':
            # 이미지->텍스트->이미지: 먼저 텍스트 생성 후 그걸 기반으로 이미지 생성
            _text = sample_fn('i2t', z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
            _z, _clip_img = sample_fn('t2i', text=_text)

        samples = unpreprocess(decode(_z))  # latent decode 후 이미지 변환
        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)

        # 개별 이미지 저장 및 워터마크 추가
        for idx, sample in enumerate(samples):
            save_path = os.path.join(config.output_path, config.mode, f'{idx}.png')
            save_image(sample, save_path)
            watermarking(save_path)
        # save a grid of generated images # 생성된 이미지들을 모아서 그리드 이미지 생성 및 저장 (워터마크 포함)
        samples_pos = []
        for idx, sample in enumerate(samples):
            sample_pil = standard_transforms.ToPILImage()(sample)
            sample_pil = utils.add_water(sample_pil)
            sample = standard_transforms.ToTensor()(sample_pil)
            samples_pos.append(sample)
        samples = make_grid(samples_pos, config.nrow)
        save_path = os.path.join(config.output_path, config.mode, f'grid.png')
        save_image(samples, save_path)


    elif config.mode in ['i2t', 't', 't2i2t']:
        if config.mode == 'i2t':
            _text = sample_fn(config.mode, z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
        elif config.mode == 't':
            _text = sample_fn(config.mode)
        elif config.mode == 't2i2t':
            _z, _clip_img = sample_fn('t2i', text=contexts_low_dim)
            _text = sample_fn('i2t', z=_z, clip_img=_clip_img)

        # 텍스트 임베딩으로부터 실제 문장 생성
        samples = caption_decoder.generate_captions(_text)
        logging.info(samples)
        os.makedirs(os.path.join(config.output_path, config.mode), exist_ok=True)
        with open(os.path.join(config.output_path, config.mode, f'{config.mode}.txt'), 'w') as f:
            print('\n'.join(samples), file=f)

    print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
    print(f'\nresults are saved in {os.path.join(config.output_path, config.mode)} :)')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/sample_unidiffuser_v1.py", "Configuration.", lock_config=False)
#flags.DEFINE_string("nnet_path", "models/uvit_v1.pth", "The nnet to evaluate.")
flags.DEFINE_string("nnet_path", "/home/ivpl-d29/sichoi/Emo/unidiffuser/si_finetuning_2/checkpoint_step_4000.pth", "The nnet to evaluate.")
flags.DEFINE_string("output_path", "out", "dir to write results to")
flags.DEFINE_string("prompt", "an elephant under the sea", "the prompt for text-to-image generation and text variation")
flags.DEFINE_string("img", "assets/space.jpg", "the image path for image-to-text generation and image variation")
flags.DEFINE_integer("n_samples", 1, "the number of samples to generate")
flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", None,
                    "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t\n"
                    "t2i: text to image\n"
                    "i2t: image to text\n"
                    "joint: joint generation of text and image\n"
                    "i: only generate image\n"
                    "t: only generate text\n"
                    "i2t2i: image variation, first image to text, then text to image\n"
                    "t2i2t: text variation, first text to image, the image to text\n"
                    )


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode

    evaluate(config)


if __name__ == "__main__":
    app.run(main)
