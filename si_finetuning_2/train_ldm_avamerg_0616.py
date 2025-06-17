"""
U-ViT 멀티모달 파인튜닝 스크립트
===========================================

사전 훈련된 U-ViT 모델을 멀티모달 데이터셋으로 파인튜닝하기 위한
전체 파이프라인 (NPY 데이터셋 지원)

0616 감정 예측 accuracy metrics 추가
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import ml_collections
import accelerate
from torchvision.utils import make_grid, save_image

# 프로젝트 내부 모듈
from libs.uvit_multi_post_ln_v1 import UViT
import torch.multiprocessing as mp


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

def build_emotion_vocab() -> Tuple[Dict[str, int], Dict[str, str]]:
    """감정 계층 구조를 정의하고 어휘를 구성합니다."""
    emotion_group = {
        'happy': ['excited', 'proud', 'grateful', 'hopeful', 'confident',
                  'anticipating', 'joyful', 'prepared', 'content', 'caring', 'trusting', 'faithful'],
        'disgusted': ['disgusted'],
        'fear': ['apprehensive', 'afraid', 'terrified'],
        'contempt': ['jealous'],
        'angry': ['furious', 'annoyed', 'angry'],
        'surprised': ['surprised', 'impressed', 'devastated'],
        'sad': ['ashamed', 'sentimental', 'disappointed', 'embarrassed',
                'nostalgic', 'anxious', 'guilty', 'lonely', 'sad']
    }

    fine2coarse = {}
    coarse2idx = {}

    for coarse, fine_list in emotion_group.items():
        coarse2idx[coarse] = len(coarse2idx)
        for fine in fine_list:
            fine2coarse[fine] = coarse

    # fine_emotion → index 매핑
    vocab = {fine: coarse2idx[fine2coarse[fine]] for fine in fine2coarse}
    return vocab, fine2coarse


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """2D sin-cos positional embedding 생성"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Grid로부터 2D positional embedding 생성"""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D positional embedding 생성"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class NPYDataset(Dataset):
    """NPY 파일로부터 멀티모달 데이터를 로드하는 데이터셋"""

    def __init__(self, npy_root: str):
        """
        감정 라벨을 포함한 NPY 데이터셋 초기화

        Args:
            npy_root: NPY 파일들이 저장된 루트 디렉토리
        """
        self.npy_root = Path(npy_root)
        self.emotion_vocab, self.fine2coarse = build_emotion_vocab()

        # NPY 파일 목록 수집
        self.npy_files = list(self.npy_root.glob("*.npy"))
        if not self.npy_files:
            raise ValueError(f"NPY 파일을 찾을 수 없습니다: {npy_root}")

        logging.info(f"총 {len(self.npy_files)}개의 NPY 파일 발견")
        logging.info(f"감정 어휘 크기: {len(self.emotion_vocab)}")

        # 첫 번째 파일로 데이터 형태 확인
        sample_data = np.load(self.npy_files[0], allow_pickle=True).item()
        self._log_data_shapes(sample_data)

    def _log_data_shapes(self, sample_data: dict):
        """샘플 데이터의 형태 로깅"""
        for key, value in sample_data.items():
            if isinstance(value, np.ndarray):
                logging.info(f"[샘플] {key} shape: {value.shape}")
            elif isinstance(value, dict) and key == 'chain_of_empathy':
                logging.info(f"[샘플] chain_of_empathy keys: {list(value.keys())}")

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """데이터 아이템 반환"""
        try:
            # NPY 파일 로드
            data = np.load(self.npy_files[idx], allow_pickle=True).item()

            # 기본 데이터 추출
            img_latent = torch.from_numpy(data['output_img_latent']).float()
            clip_feat = torch.from_numpy(data['output_clip_feat']).float()
            text_latent = torch.from_numpy(data['output_text_latent']).float()

            # 감정 라벨 추출 (중요: torch.long 타입으로 변환)
            emotion_label = 0  # 기본값
            if 'chain_of_empathy' in data:
                chain_data = data['chain_of_empathy']
                if isinstance(chain_data, dict) and 'speaker_emotion' in chain_data:
                    speaker_emotion = chain_data['speaker_emotion']
                    emotion_label = self.emotion_vocab.get(speaker_emotion, 0)

            # 정규화
            img_latent = (img_latent - img_latent.mean()) / img_latent.std()
            clip_feat = F.normalize(clip_feat, dim=-1)
            text_latent = F.normalize(text_latent, dim=-1)

            # 감정 라벨을 torch.long 타입으로 명시적 변환
            return img_latent, clip_feat, text_latent, torch.tensor(emotion_label, dtype=torch.long)

        except Exception as e:
            logging.error(f"데이터 로드 실패 (파일: {self.npy_files[idx]}): {e}")
            return self.__getitem__(0)


class NoiseScheduler:
    """확산 모델을 위한 노이즈 스케줄러"""
    def __init__(self, linear_start: float = 0.00085,
                 linear_end: float = 0.0120,
                 n_timestep: int = 1000):
        """노이즈 스케줄러 초기화"""
        self._betas = self._create_beta_schedule(linear_start, linear_end, n_timestep)
        self.betas = np.append(0., self._betas)
        self.alphas = 1. - self.betas
        self.N = len(self._betas)

        # 스킵 알파와 베타 계산
        self.skip_alphas, self.skip_betas = self._compute_skip_values()
        self.cum_alphas = self.skip_alphas[0]
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def _create_beta_schedule(self, linear_start: float, linear_end: float,
                              n_timestep: int) -> np.ndarray:
        """베타 스케줄 생성"""
        return (torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                               n_timestep, dtype=torch.float64) ** 2).numpy()

    def _compute_skip_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """스킵 알파와 베타 값 계산"""
        N = len(self.betas) - 1
        skip_alphas = np.ones([N + 1, N + 1], dtype=self.betas.dtype)

        for s in range(N + 1):
            skip_alphas[s, s + 1:] = self.alphas[s + 1:].cumprod()

        skip_betas = np.zeros([N + 1, N + 1], dtype=self.betas.dtype)
        for t in range(N + 1):
            prod = self.betas[1: t + 1] * skip_alphas[1: t + 1, t]
            skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]

        return skip_alphas, skip_betas

    def sample_multimodal(self, x_img: torch.Tensor, x_clip: torch.Tensor, y_text: torch.Tensor) -> Tuple:
        batch_size = len(x_img)

        n_img = np.random.choice(list(range(1, self.N + 1)), (batch_size,))
        n_clip = np.random.choice(list(range(1, self.N + 1)), (batch_size,))
        n_text = np.random.choice(list(range(1, self.N + 1)), (batch_size,))

        eps_img = torch.randn_like(x_img)
        eps_clip = torch.randn_like(x_clip)
        eps_text = torch.randn_like(y_text)

        xn_img = self._apply_noise(x_img, eps_img, n_img)
        xn_clip = self._apply_noise(x_clip, eps_clip, n_clip)
        yn_text = self._apply_noise(y_text, eps_text, n_text)

        return (torch.tensor(n_img, device=x_img.device),
                torch.tensor(n_clip, device=x_clip.device),
                torch.tensor(n_text, device=y_text.device)), \
            (eps_img, eps_clip, eps_text), \
            (xn_img, xn_clip, yn_text)

    def _apply_noise(self, x: torch.Tensor, eps: torch.Tensor,
                     n: np.ndarray) -> torch.Tensor:
        """텐서에 노이즈 적용"""
        alpha_cumprod = torch.from_numpy(self.cum_alphas[n]).type_as(x)
        beta_cumprod = torch.from_numpy(self.cum_betas[n]).type_as(x)

        # 브로드캐스팅을 위한 차원 조정
        extra_dims = (1,) * (x.dim() - 1)
        alpha_cumprod = alpha_cumprod.view(-1, *extra_dims)
        beta_cumprod = beta_cumprod.view(-1, *extra_dims)

        return (alpha_cumprod ** 0.5) * x + (beta_cumprod ** 0.5) * eps


class MultimodalLossWithEmotion:
    """감정 분류 손실이 추가된 멀티모달 U-ViT 손실 함수 클래스"""
    @staticmethod
    def _mean_squared_error(pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """평균 제곱 오차 계산"""
        if target is None:
            return pred.pow(2).mean()
        return (pred - target).pow(2).mean()

    @staticmethod
    def compute_loss(img_latent: torch.Tensor,
                     clip_feat: torch.Tensor,
                     text_latent: torch.Tensor,
                     emotion_labels: torch.Tensor,
                     model: nn.Module,
                     scheduler,
                     img_channels: int,
                     emotion_loss_weight: float = 0.1,
                     **kwargs):
        """확산 손실과 감정 분류 손실을 결합한 총 손실 계산 (개선된 버전)"""

        def validate_tensor_dimensions(tensor, expected_shape_desc, expected_dims):
            """텐서 차원 검증 유틸리티"""
            actual_dims = tensor.dim()
            actual_shape = tensor.shape

            if actual_dims != expected_dims:
                raise ValueError(f"{expected_shape_desc} 차원 오류: "
                                 f"예상 {expected_dims}차원, 실제 {actual_dims}차원 "
                                 f"(shape: {actual_shape})")

            return True

        # 입력 검증
        batch_size = img_latent.size(0)
        validate_tensor_dimensions(emotion_labels, "감정 라벨", 1)
        assert emotion_labels.dtype == torch.long, f"감정 라벨 타입 오류: {emotion_labels.dtype}"

        # 멀티모달 노이즈 샘플링
        (n_img, n_clip, n_text), (eps_img, eps_clip, eps_text), (
            x_img_noised, x_clip_noised, x_text_noised) = scheduler.sample_multimodal(
            img_latent, clip_feat, text_latent)

        # data_type 생성
        data_type = torch.randint(0, 3, (batch_size,), device=img_latent.device)

        # 모델 예측 (감정 예측 포함)
        pred_img, pred_clip, pred_text, emotion_logits = model(
            img=x_img_noised,
            clip_img=x_clip_noised,
            text=x_text_noised,
            t_img=n_img,
            t_text=n_text,
            data_type=data_type,
            return_emotion=True
        )

        # 감정 로짓 형태 강화 검증
        validate_tensor_dimensions(emotion_logits, "감정 로짓", 2)
        expected_shape = (batch_size, 7)  # 7개 감정 클래스
        if emotion_logits.shape != expected_shape:
            raise ValueError(f"감정 로짓 형태 불일치: 예상 {expected_shape}, 실제 {emotion_logits.shape}")

        # 확산 손실 계산
        loss_img = MultimodalLossWithEmotion._mean_squared_error(eps_img - pred_img)
        loss_clip = MultimodalLossWithEmotion._mean_squared_error(eps_clip - pred_clip) * 0.05
        loss_text = MultimodalLossWithEmotion._mean_squared_error(eps_text - pred_text)

        # 감정 분류 손실 계산
        loss_emotion = F.cross_entropy(emotion_logits, emotion_labels)

        return loss_img, loss_clip, loss_text, loss_emotion


class UViTWithEmotion(nn.Module):
    def __init__(self, uvit_model, num_emotions=7, embed_dim=1536):
        super().__init__()
        self.uvit = uvit_model
        self.num_emotions = num_emotions
        self.embed_dim = embed_dim

        # CLS 토큰 방식의 감정 토큰 (학습 가능한 파라미터)
        self.emotion_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 감정 분류를 위한 헤드
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_emotions)
        )

        # [EMO] 토큰을 위한 학습 가능한 파라미터 (1차원으로 수정)
        self.emotion_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.emotion_pos_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # 특징 변환 레이어들
        self.img_projection = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.clip_projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.text_projection = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        # 강화된 감정 분류기
        self.emotion_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_emotions)  # 최종: [B, 7]
        )

        # 감정 토큰용 위치 임베딩
        self.emotion_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def forward(self, img: torch.Tensor, clip_img: torch.Tensor,
                text: torch.Tensor, t_img: torch.Tensor,
                t_text: torch.Tensor, data_type: torch.Tensor,
                return_emotion: bool = True) -> Tuple:
        """감정 CLS 토큰을 포함한 순전파"""

        # 기본 U-ViT 순전파를 통해 중간 특징 추출
        # U-ViT 내부에서 토큰화된 특징들을 가져와야 함
        batch_size = img.size(0)

        if return_emotion:
            # U-ViT의 내부 forward를 수정하여 중간 특징을 가져와야 함
            # 여기서는 간접적으로 기본 예측을 통해 특징을 추출
            with torch.no_grad():
                pred_img, pred_clip, pred_text = self.uvit(img, clip_img, text, t_img, t_text, data_type)

            # 예측 결과로부터 특징 추출 (실제로는 U-ViT 내부 특징을 사용해야 함)
            # 이미지 특징 처리
            if pred_img.dim() == 4:  # [B, C, H, W]
                img_feat = pred_img.flatten(2).mean(dim=2)  # [B, C]
            else:
                img_feat = pred_img.mean(dim=1) if pred_img.dim() > 2 else pred_img

            # 텍스트 특징 처리
            if pred_text.dim() == 3:  # [B, seq_len, dim]
                text_feat = pred_text.mean(dim=1)  # [B, dim]
            else:
                text_feat = pred_text

            # CLIP 특징은 이미 [B, dim] 형태
            clip_feat = pred_clip

            # 특징들을 embed_dim으로 투영
            if img_feat.size(-1) != self.embed_dim:
                img_feat = F.adaptive_avg_pool1d(img_feat.unsqueeze(1), self.embed_dim).squeeze(1)
            if text_feat.size(-1) != self.embed_dim:
                text_feat = F.adaptive_avg_pool1d(text_feat.unsqueeze(1), self.embed_dim).squeeze(1)
            if clip_feat.size(-1) != self.embed_dim:
                # 예: clip_feat가 4D일 경우 [B, C, H, W]
                if clip_feat.dim() == 4:
                    clip_feat = clip_feat.flatten(2).mean(dim=2)  # [B, C]
                elif clip_feat.dim() == 3:
                    clip_feat = clip_feat.mean(dim=1)  # [B, D]

                # 이 시점에서 clip_feat: [B, D]
                clip_feat = F.adaptive_avg_pool1d(clip_feat.unsqueeze(1), self.embed_dim).squeeze(1)

            # CLS 토큰 생성 및 멀티모달 특징 결합
            emotion_cls = self.emotion_cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]

            # 멀티모달 특징을 토큰 형태로 변환
            multimodal_feat = torch.stack([img_feat, clip_feat, text_feat], dim=1)  # [B, 3, embed_dim]

            # CLS 토큰과 멀티모달 특징 결합
            combined_tokens = torch.cat([emotion_cls, multimodal_feat], dim=1)  # [B, 4, embed_dim]

            # 간단한 self-attention을 통한 feature mixture
            # 여기서는 CLS 토큰에 다른 토큰들의 정보를 집약
            cls_output = combined_tokens[:, 0]  # CLS 토큰만 추출 [B, embed_dim]

            # 다른 토큰들과의 상호작용을 위한 weighted average
            attention_weights = F.softmax(torch.matmul(cls_output.unsqueeze(1),
                                                       combined_tokens[:, 1:].transpose(-2, -1)), dim=-1)
            attended_feat = torch.matmul(attention_weights, combined_tokens[:, 1:]).squeeze(1)

            # 최종 CLS token feature
            final_cls_feat = cls_output + attended_feat

            # 감정 분류
            emotion_logits = self.emotion_head(final_cls_feat)  # [B, num_emotions]

            # 실제 예측값도 다시 계산 (감정 정보를 포함하여)
            pred_img, pred_clip, pred_text = self.uvit(img, clip_img, text, t_img, t_text, data_type)

            return pred_img, pred_clip, pred_text, emotion_logits
        else:
            # 감정 예측 없이 기본 U-ViT만 실행
            return self.uvit(img, clip_img, text, t_img, t_text, data_type)


class ModelManagerWithEmotion:
    """감정 예측 기능이 추가된 모델 로딩 및 관리 클래스"""

    @staticmethod
    def load_uvit_model_with_emotion(model_path: str, config: ml_collections.ConfigDict,
                                     num_emotions: int = 7) -> nn.Module:
        """감정 예측 기능이 추가된 U-ViT 모델 로드"""
        from libs.uvit_multi_post_ln_v1 import UViT

        # 기본 U-ViT 모델 생성
        uvit = UViT(
            img_size=config.nnet.img_size,
            in_chans=config.nnet.in_chans,
            patch_size=config.nnet.patch_size,
            embed_dim=config.nnet.embed_dim,
            depth=config.nnet.depth,
            num_heads=config.nnet.num_heads,
            mlp_ratio=config.nnet.mlp_ratio,
            qkv_bias=config.nnet.qkv_bias,
            text_dim=config.nnet.text_dim,
            num_text_tokens=config.nnet.num_text_tokens,
            clip_img_dim=config.nnet.clip_img_dim,
            use_checkpoint=config.nnet.use_checkpoint
        )

        # 사전 훈련된 가중치 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)

            # token_embedding.weight 크기 문제 처리
            if 'token_embedding.weight' in state_dict:
                old_emb = state_dict['token_embedding.weight']
                expected_shape = uvit.token_embedding.weight.shape
                if old_emb.shape != expected_shape:
                    logging.info(f"token_embedding.weight 크기 조정: {old_emb.shape} -> {expected_shape}")
                    new_emb = torch.zeros(expected_shape)
                    n_copy = min(old_emb.shape[0], expected_shape[0])
                    new_emb[:n_copy] = old_emb[:n_copy]
                    if expected_shape[0] > n_copy:
                        new_emb[n_copy:] = torch.randn(expected_shape[0] - n_copy, expected_shape[1]) * 0.02
                    state_dict['token_embedding.weight'] = new_emb

            uvit.load_state_dict(state_dict, strict=False)
            logging.info(f"기본 U-ViT 모델 로드 완료: {model_path}")
        else:
            logging.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 감정 예측 기능 추가
        emotion_model = UViTWithEmotion(uvit, num_emotions=num_emotions, embed_dim=config.nnet.embed_dim)

        # 선택적 모델 고정
        ModelManagerWithEmotion._freeze_model_layers(emotion_model)

        return emotion_model


    @staticmethod
    def _freeze_model_layers(model: nn.Module) -> None:
        """감정 예측 기능을 고려한 선택적 레이어 고정"""
        total_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            if 'projection' in name and param.requires_grad:
                if param.grad is not None:
                    logging.debug(f"{name}: grad_norm={param.grad.norm().item():.6f}")
            total_params += param.numel()

            # 학습할 레이어 선택 (감정 관련 레이어 추가)
            if any(keyword in name for keyword in [
                'decoder_pred', 'clip_img_out', 'text_out',
                'mid_block',
                'out_blocks.12', 'out_blocks.13', 'out_blocks.14',
                'in_blocks.12', 'in_blocks.13', 'in_blocks.14',
                'token_embedding',
                'emotion_token', 'emotion_classifier', 'emotion_pos_embed',  # 감정 관련 파라미터
                'img_projection', 'clip_projection', 'text_projection'  # 투영 레이어 추가
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        freeze_ratio = frozen_params / total_params * 100
        logging.info(f"감정 기능 포함 모델 고정 완료: {frozen_params}/{total_params} "
                     f"파라미터 ({freeze_ratio:.1f}%)")


class TrainingManagerWithEmotion:
    """감정 예측 기능이 추가된 훈련 과정 관리 클래스"""

    def __init__(self, config: ml_collections.ConfigDict):
        """훈련 매니저 초기화"""
        self.config = config
        self.accelerator = accelerate.Accelerator(mixed_precision=config.mixed_precision)
        self.device = self.accelerator.device
        accelerate.utils.set_seed(config.seed, device_specific=True)
        logging.info(f'Process {self.accelerator.process_index} using device: {self.device}')
        self.scheduler = NoiseScheduler()
        self.setup_logging()


    def setup_logging(self) -> None:
        """로깅 설정"""
        if self.accelerator.is_main_process:
            # wandb 초기화
            wandb.init(
                dir=os.path.join('workdir'),
                project=f'uvit_emotion_{self.config.dataset.name}_0616',
                config=self.config.to_dict(),
                name='finetune_emotion_uvit',
                job_type='finetune',
                mode='online'
            )

            # 로거 설정
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('workdir/training_emotion.log'),
                    logging.StreamHandler()
                ]
            )
            logging.info("감정 예측 U-ViT 훈련 설정:")
            logging.info(self.config)
        else:
            logging.basicConfig(level=logging.ERROR)

    def setup_data_loader(self) -> DataLoader:
        """감정 라벨을 포함한 데이터 로더 설정"""
        try:
            # 감정 기능이 추가된 NPY 데이터셋 생성
            dataset = NPYDataset(npy_root=self.config.dataset.train_npy_root)
            return DataLoader(
                dataset,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
        except Exception as e:
            logging.error(f"데이터 로더 설정 중 오류 발생: {e}")
            raise

    def train_step(self, batch: Tuple, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """감정 손실과 accuracy가 포함된 훈련 스텝"""
        metrics = {}
        optimizer.zero_grad()

        try:
            img_latents, clip_features, text_features, emotion_labels = batch

            # GPU로 이동
            img_latents = img_latents.to(self.device)
            clip_features = clip_features.to(self.device)
            text_features = text_features.to(self.device)
            emotion_labels = emotion_labels.to(self.device)

            # 디버깅: 텐서 형태 및 타입 로깅
            logging.debug(f"img_latents shape: {img_latents.shape}, dtype: {img_latents.dtype}")
            logging.debug(f"clip_features shape: {clip_features.shape}, dtype: {clip_features.dtype}")
            logging.debug(f"text_features shape: {text_features.shape}, dtype: {text_features.dtype}")
            logging.debug(f"emotion_labels shape: {emotion_labels.shape}, dtype: {emotion_labels.dtype}")
            logging.debug(f"emotion_labels values: {emotion_labels}")

            # 멀티모달 + 감정 손실 계산
            with self.accelerator.autocast():
                loss_img, loss_clip, loss_text, loss_emotion = MultimodalLossWithEmotion.compute_loss(
                    img_latent=img_latents,
                    clip_feat=clip_features,
                    text_latent=text_features,
                    emotion_labels=emotion_labels,
                    model=model,
                    scheduler=self.scheduler,
                    img_channels=img_latents.shape[1],
                    emotion_loss_weight=self.config.get('emotion_loss_weight', 0.1)
                )

            # 총 손실 계산
            emotion_weight = self.config.get('emotion_loss_weight', 0.1)
            total_loss = loss_img + loss_clip + loss_text + emotion_weight * loss_emotion

            # ========== 감정 예측 Accuracy 계산 추가 ==========
            with torch.no_grad():
                # 노이즈 샘플링 (accuracy 계산용)
                (n_img, n_clip, n_text), (eps_img, eps_clip, eps_text), (
                    x_img_noised, x_clip_noised, x_text_noised) = self.scheduler.sample_multimodal(
                    img_latents, clip_features, text_features)

                # data_type 생성
                batch_size = img_latents.size(0)
                data_type = torch.randint(0, 3, (batch_size,), device=img_latents.device)

                # 감정 예측만 수행
                _, _, _, emotion_logits = model(
                    img=x_img_noised,
                    clip_img=x_clip_noised,
                    text=x_text_noised,
                    t_img=n_img,
                    t_text=n_text,
                    data_type=data_type,
                    return_emotion=True
                )

                # Accuracy 계산
                predicted_emotions = torch.argmax(emotion_logits, dim=1)
                correct_predictions = (predicted_emotions == emotion_labels).sum().item()
                total_predictions = emotion_labels.size(0)
                emotion_accuracy = correct_predictions / total_predictions

                # 감정별 accuracy 계산 (선택사항)
                emotion_names = ['happy', 'disgusted', 'fear', 'contempt', 'angry', 'surprised', 'sad']
                per_emotion_acc = {}
                for emotion_idx in range(7):
                    mask = (emotion_labels == emotion_idx)
                    if mask.sum() > 0:  # 해당 감정이 배치에 존재하는 경우만
                        emotion_correct = (predicted_emotions[mask] == emotion_labels[mask]).sum().item()
                        emotion_total = mask.sum().item()
                        per_emotion_acc[f'acc_{emotion_names[emotion_idx]}'] = emotion_correct / emotion_total
                    else:
                        per_emotion_acc[f'acc_{emotion_names[emotion_idx]}'] = 0.0

                # 메트릭에 추가
                metrics.update(per_emotion_acc)

                # Top-2 Accuracy도 계산 (추가 메트릭)
                _, top2_predictions = torch.topk(emotion_logits, 2, dim=1)
                top2_correct = torch.any(top2_predictions == emotion_labels.unsqueeze(1), dim=1).sum().item()
                emotion_top2_accuracy = top2_correct / total_predictions

            # 역전파
            self.accelerator.backward(total_loss.mean())

            # 메트릭 수집 (accuracy 포함)
            metrics['loss'] = self.accelerator.gather(total_loss.detach()).mean().item()
            metrics['loss_img'] = self.accelerator.gather(loss_img.detach()).mean().item()
            metrics['loss_clip'] = self.accelerator.gather(loss_clip.detach()).mean().item()
            metrics['loss_text'] = self.accelerator.gather(loss_text.detach()).mean().item()
            metrics['loss_emotion'] = self.accelerator.gather(loss_emotion.detach()).mean().item()

            # ========== Accuracy 메트릭 추가 ==========
            metrics['emotion_accuracy'] = emotion_accuracy
            metrics['emotion_top2_accuracy'] = emotion_top2_accuracy

            # 분산 환경에서 accuracy 평균 계산
            if self.accelerator.num_processes > 1:
                accuracy_tensor = torch.tensor(emotion_accuracy, device=self.device)
                top2_accuracy_tensor = torch.tensor(emotion_top2_accuracy, device=self.device)

                metrics['emotion_accuracy'] = self.accelerator.gather(accuracy_tensor).mean().item()
                metrics['emotion_top2_accuracy'] = self.accelerator.gather(top2_accuracy_tensor).mean().item()

            # 그래디언트 클리핑
            if hasattr(self.config.train, 'grad_clip') and self.config.train.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    self.config.train.grad_clip
                )

            optimizer.step()

        except Exception as e:
            logging.error(f"훈련 스텝 중 오류 발생: {e}")
            # 디버깅 정보 추가 출력
            if 'emotion_labels' in locals():
                logging.error(f"emotion_labels shape: {emotion_labels.shape}")
                logging.error(f"emotion_labels dtype: {emotion_labels.dtype}")
            raise

        return metrics

    @staticmethod
    def update_ema(ema_model: nn.Module, model: nn.Module,
                   decay: float = 0.9999) -> None:
        """지수 이동 평균(EMA) 모델 업데이트"""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def save_checkpoint(self, model: nn.Module, ema_model: nn.Module,
                        optimizer: torch.optim.Optimizer, step: int) -> None:
        """체크포인트 저장"""
        if self.accelerator.is_main_process:
            os.makedirs('ckpt', exist_ok=True)

            def remove_prefix(state_dict):
                return {
                    (k.replace("module.", "") if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }

            model_state = remove_prefix(self.accelerator.unwrap_model(model).state_dict())
            ema_state = remove_prefix(self.accelerator.unwrap_model(ema_model).state_dict())

            checkpoint = {
                'model': model_state,
                'ema_model': ema_state,
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': self.config.to_dict(),
                'emotion_vocab': build_emotion_vocab()[0]  # 감정 어휘 저장
            }

            save_path = f'ckpt/checkpoint_emotion_step_{step}.pth'
            torch.save(checkpoint, save_path)
            logging.info(f"감정 예측 체크포인트 저장 완료: {save_path}")

    def evaluate_emotion_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        """감정 예측 정확도 평가"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                img_latents, clip_features, text_features, emotion_labels = batch
                img_latents = img_latents.to(self.device)
                clip_features = clip_features.to(self.device)
                text_features = text_features.to(self.device)
                emotion_labels = emotion_labels.to(self.device)

                # 감정 예측만 수행
                emotion_logits = model.predict_emotion_only(img_latents, clip_features, text_features)
                predicted = torch.argmax(emotion_logits, dim=1)

                total += emotion_labels.size(0)
                correct += (predicted == emotion_labels).sum().item()

        model.train()
        accuracy = correct / total
        return accuracy


def main():
    """감정 예측 기능이 추가된 U-ViT 훈련 메인 함수"""
    try:
        # 설정 로드
        logging.info("설정 로드 중...")
        from configs.finetune_uvit_config import get_config
        config = get_config()

        # 감정 관련 설정 추가
        config.emotion_loss_weight = 0.1  # 감정 손실 가중치
        config.num_emotions = 7  # 감정 클래스 수

        # 훈련 매니저 초기화
        logging.info("훈련 매니저 초기화 중...")
        trainer = TrainingManagerWithEmotion(config)

        # 모델 및 데이터 설정
        logging.info("데이터 로더 및 모델 설정 중...")
        data_loader = trainer.setup_data_loader()
        model = ModelManagerWithEmotion.load_uvit_model_with_emotion(
            config.nnet.pretrained_path, config, num_emotions=config.num_emotions)

        # EMA 모델 생성
        logging.info("EMA 모델 생성 중...")
        ema_model = ModelManagerWithEmotion.load_uvit_model_with_emotion(
            config.nnet.pretrained_path, config, num_emotions=config.num_emotions)
        ema_model.eval()

        # 옵티마이저 설정
        logging.info("옵티마이저 설정 중...")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas
        )

        # 학습률 스케줄러
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / config.lr_scheduler.warmup_steps, 1.0)
        )

        # 분산 훈련 준비
        logging.info("분산 훈련 준비 중...")
        model, ema_model, optimizer, data_loader, lr_scheduler = trainer.accelerator.prepare(
            model, ema_model, optimizer, data_loader, lr_scheduler
        )

        # 훈련 시작
        logging.info("감정 예측 기능이 추가된 U-ViT 파인튜닝 시작")
        step = 0
        model.train()

        while step < config.train.n_steps:
            pbar = tqdm(data_loader, desc=f'Step {step}/{config.train.n_steps}',
                        disable=not trainer.accelerator.is_main_process)

            for batch in pbar:
                if step >= config.train.n_steps:
                    break

                # 훈련 스텝 수행
                metrics = trainer.train_step(batch, model, optimizer)

                # EMA 업데이트
                TrainingManagerWithEmotion.update_ema(ema_model, model, config.get('ema_rate', 0.9999))

                # 학습률 업데이트
                lr_scheduler.step()

                # 로깅
                if trainer.accelerator.is_main_process and step % config.train.log_interval == 0:
                    metrics['lr'] = optimizer.param_groups[0]['lr']
                    metrics['step'] = step

                    logging.info(f"Step {step}: Loss={metrics['loss']:.6f}, "
                                 f"loss_img={metrics['loss_img']:.6f}, "
                                 f"loss_clip={metrics['loss_clip']:.4f}, "
                                 f"loss_text={metrics['loss_text']:.6f}, "
                                 f"loss_emotion={metrics['loss_emotion']:.6f}, "
                                 f"emotion_acc={metrics['emotion_accuracy']:.3f}, "
                                 f"emotion_top2_acc={metrics['emotion_top2_accuracy']:.3f}, "
                                 f"LR={metrics['lr']:.7f}")
                    wandb.log(metrics, step=step)

                # 체크포인트 저장
                if step % config.train.save_interval == 0 and step > 0:
                    trainer.save_checkpoint(model, ema_model, optimizer, step)

                step += 1

        # 최종 체크포인트 저장
        trainer.save_checkpoint(model, ema_model, optimizer, step)
        logging.info("감정 예측 기능이 추가된 U-ViT 훈련 완료!")

    except Exception as e:
        logging.error(f"훈련 중 오류 발생: {e}")
        raise



if __name__ == "__main__":
    import torch
    import logging
    import sys
    import os

    logging.basicConfig(level=logging.INFO)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        logging.info(f"Process {rank} started.")
    else:
        logging.info("Single-process start.")

    logging.info("감정 예측 U-ViT 메인 함수 진입")
    sys.stdout.flush()
    main()

