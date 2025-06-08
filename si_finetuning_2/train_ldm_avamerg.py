"""
U-ViT 멀티모달 파인튜닝 스크립트
================================

이 모듈은 사전 훈련된 U-ViT 모델을 멀티모달 데이터셋으로 파인튜닝하기 위한
전체 파이프라인을 제공합니다.

주요 기능:
- 멀티모달 손실 함수 (LUViT)
- DPM Solver를 이용한 샘플링
- EMA 모델 업데이트
- FID 평가 메트릭

이거는 기존 코드 정리하고 npy 데이터셋 적용 전 

"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import ml_collections
import accelerate
from torchvision.utils import make_grid, save_image

# 프로젝트 내부 모듈
from libs.uvit_multi_post_ln_v1 import UViT
from finetune_datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from tools.fid_score import calculate_fid_given_paths
import utils
import libs.autoencoder


@dataclass
class TrainingConfig:
    """훈련 설정을 위한 데이터클래스"""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    n_steps: int = 10000
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 2000
    grad_clip: float = 1.0
    ema_rate: float = 0.9999


class NoiseScheduler:
    """
    확산 모델을 위한 노이즈 스케줄러

    Stable Diffusion 베타 스케줄을 사용하여 노이즈를 관리합니다.
    """

    def __init__(self, linear_start: float = 0.00085,
                 linear_end: float = 0.0120,
                 n_timestep: int = 1000):
        """
        노이즈 스케줄러 초기화

        Args:
            linear_start: 선형 시작값
            linear_end: 선형 끝값  
            n_timestep: 타임스텝 개수
        """
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

    def sample_multimodal(self, x0: torch.Tensor, y0: torch.Tensor) -> Tuple:
        """
        멀티모달 샘플링 수행

        Args:
            x0: 이미지+CLIP 잠재 변수 [B, C1+C2, H, W]
            y0: 텍스트 특징 [B, T, D]

        Returns:
            타임스텝, 노이즈, 노이즈가 추가된 데이터
        """
        batch_size = len(x0)

        # 각 모달리티에 대해 독립적인 타임스텝 샘플링
        n_x = np.random.choice(list(range(1, self.N + 1)), (batch_size,))
        n_y = np.random.choice(list(range(1, self.N + 1)), (batch_size,))

        # 가우시안 노이즈 생성
        eps_x = torch.randn_like(x0)
        eps_y = torch.randn_like(y0)

        # 노이즈 추가
        xn = self._apply_noise(x0, eps_x, n_x)
        yn = self._apply_noise(y0, eps_y, n_y)

        return (torch.tensor(n_x, device=x0.device),
                torch.tensor(n_y, device=y0.device)), (eps_x, eps_y), (xn, yn)

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


class MultimodalLoss:
    """멀티모달 U-ViT 손실 함수 클래스"""

    @staticmethod
    def compute_loss(x0: torch.Tensor, y0: torch.Tensor,
                     model: nn.Module, scheduler: NoiseScheduler,
                     img_channels: int, **kwargs) -> torch.Tensor:
        """
        멀티모달 U-ViT 손실 계산

        Loss = E_{x0,y0,ε_x,ε_y,t_x,t_y} ||ε_θ(x_t^x, y_t^y, t_x, t_y) - [ε_x, ε_y]||_2^2

        Args:
            x0: 이미지+CLIP 잠재 변수 [B, C1+C2, H, W]
            y0: 텍스트 특징 [B, T, D]
            model: U-ViT 모델
            scheduler: 노이즈 스케줄러
            img_channels: 이미지 채널 수

        Returns:
            계산된 손실값
        """
        # 멀티모달 노이즈 샘플링
        (n_x, n_y), (eps_x, eps_y), (xn, yn) = scheduler.sample_multimodal(x0, y0)

        # 데이터 타입 무작위 선택 (0: 이미지, 1: 텍스트, 2: 멀티모달)
        data_type = torch.randint(0, 3, (x0.size(0),), device=x0.device)

        # 모델 예측
        pred_img_clip, _, pred_text = model(
            img=xn[:, :img_channels],  # 이미지 부분 추출
            clip_img=xn[:, img_channels:],  # CLIP 부분 추출
            text=yn,  # 노이즈가 추가된 텍스트
            t_img=n_x,
            t_text=n_y,
            data_type=data_type
        )

        # 손실 계산 - 모델이 각 모달리티에 추가된 노이즈를 예측해야 함
        loss_img_clip = MultimodalLoss._mean_squared_error(
            eps_x - torch.cat([pred_img_clip, pred_text * 0], dim=1)
        )
        loss_text = MultimodalLoss._mean_squared_error(eps_y - pred_text)

        return loss_img_clip + loss_text

    @staticmethod
    def _mean_squared_error(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
        """평균 제곱 오차 계산"""
        return tensor.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class ModelManager:
    """모델 로딩 및 관리를 위한 클래스"""

    @staticmethod
    def load_uvit_model(model_path: str, config: ml_collections.ConfigDict) -> nn.Module:
        """
        사전 훈련된 U-ViT 모델 로드

        Args:
            model_path: 모델 체크포인트 경로
            config: 모델 설정

        Returns:
            로드된 U-ViT 모델
        """
        # U-ViT 모델 생성
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
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        uvit.load_state_dict(state_dict)

        # 선택적 모델 고정 (메모리 절약)
        ModelManager._freeze_model_layers(uvit)

        return ModelManager.UViTWrapper(uvit)

    @staticmethod
    def _freeze_model_layers(model: nn.Module) -> None:
        """
        메모리 절약을 위한 선택적 레이어 고정

        출력 레이어와 마지막 몇 개 레이어만 학습 가능하게 설정
        """
        total_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()

            # 출력 레이어와 마지막 3개 블록만 학습
            if any(keyword in name for keyword in [
                'decoder_pred', 'clip_img_out', 'text_out',
                'blocks.11', 'blocks.10', 'blocks.9'
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        freeze_ratio = frozen_params / total_params * 100
        logging.info(f"모델 고정 완료: {frozen_params}/{total_params} "
                     f"파라미터 ({freeze_ratio:.1f}%)")

    class UViTWrapper(nn.Module):
        """U-ViT 모델을 위한 래퍼 클래스"""

        def __init__(self, uvit_model: nn.Module):
            super().__init__()
            self.uvit = uvit_model
            self.num_text_tokens = uvit_model.num_text_tokens
            self.embed_dim = uvit_model.embed_dim

        def forward(self, img: torch.Tensor, clip_img: torch.Tensor,
                    text: torch.Tensor, t_img: torch.Tensor,
                    t_text: torch.Tensor, data_type: torch.Tensor) -> Tuple:
            """모델 순전파"""
            return self.uvit(img, clip_img, text, t_img, t_text, data_type)


class TrainingManager:
    """훈련 과정 관리 클래스"""

    def __init__(self, config: ml_collections.ConfigDict):
        """
        훈련 매니저 초기화

        Args:
            config: 훈련 설정
        """
        self.config = config
        self.accelerator = accelerate.Accelerator()
        self.device = self.accelerator.device
        self.scheduler = NoiseScheduler()
        self.setup_logging()

    def setup_logging(self) -> None:
        """로깅 설정"""
        if self.accelerator.is_main_process:
            wandb.init(
                dir=os.path.abspath(self.config.workdir),
                project=f'uvit_finetune_{self.config.dataset.name}',
                config=self.config.to_dict(),
                name=self.config.hparams,
                job_type='finetune',
                mode='online'
            )
            utils.set_logger(
                log_level='info',
                fname=os.path.join(self.config.workdir, 'output.log')
            )
            logging.info("훈련 설정:")
            logging.info(self.config)
        else:
            utils.set_logger(log_level='error')

    def setup_data_loader(self) -> DataLoader:
        """데이터 로더 설정"""
        try:
            # 필요한 모델들 로드
            autoencoder, clip_img_model, clip_preprocess, clip_text_model, linear_proj = \
                utils.load_models(self.config)

            # 데이터셋 설정에 모델들 추가
            dataset_config = self.config.dataset.copy()
            dataset_config.update({
                'autoencoder': autoencoder,
                'clip_img_model': clip_img_model,
                'clip_text_model': clip_text_model,
                'clip_preprocess': clip_preprocess,
                'linear_proj': linear_proj,
                'device': self.device
            })

            # 데이터셋 생성
            dataset = get_dataset(**dataset_config)
            assert os.path.exists(dataset.fid_stat), "FID 통계 파일이 존재하지 않습니다"

            train_dataset = dataset.get_split(split='train', labeled=True)

            return DataLoader(
                train_dataset,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,
                pin_memory=False,
                persistent_workers=False,
            )

        except Exception as e:
            logging.error(f"데이터 로더 설정 중 오류 발생: {e}")
            raise

    def train_step(self, batch: Tuple, model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   autoencoder: nn.Module) -> Dict[str, float]:
        """
        단일 훈련 스텝 수행

        Args:
            batch: 배치 데이터
            model: 훈련 모델
            optimizer: 옵티마이저
            autoencoder: 오토인코더 모델

        Returns:
            훈련 메트릭
        """
        metrics = {}
        optimizer.zero_grad()

        try:
            if len(batch) != 3:
                raise ValueError(f"배치는 3개 요소를 가져야 합니다. 실제: {len(batch)}")

            images, clip_features, text_features = batch
            images = images.to(self.device)
            clip_features = clip_features.to(self.device)
            text_features = text_features.to(self.device)

            # 이미지를 잠재 공간으로 인코딩
            with torch.amp.autocast('cuda'):
                z_img = autoencoder.encode(images)

            # CLIP 특징을 이미지 잠재 변수와 같은 차원으로 변환
            clip_features = self._reshape_clip_features(clip_features, z_img)

            # 이미지 잠재 변수와 CLIP 특징 결합
            x0 = torch.cat([z_img, clip_features], dim=1)
            y0 = text_features

            # 멀티모달 손실 계산
            loss = MultimodalLoss.compute_loss(
                x0, y0, model, self.scheduler,
                img_channels=z_img.shape[1],
                text=text_features
            )

            # 역전파
            self.accelerator.backward(loss.mean())

            # 그래디언트 클리핑
            if self.config.train.get('grad_clip', 0) > 0:
                self.accelerator.clip_grad_norm_(
                    model.parameters(),
                    self.config.train.grad_clip
                )

            optimizer.step()

            metrics['loss'] = self.accelerator.gather(loss.detach()).mean().item()

        except Exception as e:
            logging.error(f"훈련 스텝 중 오류 발생: {e}")
            raise

        return metrics

    def _reshape_clip_features(self, clip_features: torch.Tensor,
                               z_img: torch.Tensor) -> torch.Tensor:
        """CLIP 특징을 이미지 잠재 변수와 같은 공간 차원으로 변환"""
        # [B, D] → [B, D, 1, 1] → [B, D, H, W]
        clip_features = clip_features.unsqueeze(-1).unsqueeze(-1)
        clip_features = clip_features.expand(-1, -1, z_img.shape[2], z_img.shape[3])
        return clip_features

    @staticmethod
    def update_ema(ema_model: nn.Module, model: nn.Module,
                   decay: float = 0.9999) -> None:
        """
        지수 이동 평균(EMA) 모델 업데이트

        Args:
            ema_model: EMA 모델
            model: 현재 모델
            decay: 감쇠율
        """
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    """메인 함수"""
    try:
        # 설정 로드
        from configs import finetune_uvit_config
        config = finetune_uvit_config.get_config()

        # 훈련 매니저 초기화
        trainer = TrainingManager(config)

        # 모델 및 데이터 설정
        data_loader = trainer.setup_data_loader()
        model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
        model.to(trainer.device)

        # EMA 모델 생성
        ema_model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
        ema_model.to(trainer.device)
        ema_model.eval()

        # 옵티마이저 설정
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(
            list(trainable_params),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )

        # 학습률 스케줄러
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / config.lr_scheduler.warmup_steps, 1.0)
        )

        # 분산 훈련 준비
        model, ema_model, optimizer, data_loader = trainer.accelerator.prepare(
            model, ema_model, optimizer, data_loader
        )

        # 훈련 시작
        logging.info("U-ViT 파인튜닝 시작")
        step = 0

        while step < config.train.n_steps:
            for batch in data_loader:
                if step >= config.train.n_steps:
                    break

                model.train()

                # 훈련 스텝 수행
                metrics = trainer.train_step(batch, model, optimizer, None)  # autoencoder 필요

                # EMA 업데이트
                TrainingManager.update_ema(ema_model, model, config.get('ema_rate', 0.9999))

                # 학습률 업데이트
                lr_scheduler.step()

                # 로깅
                if trainer.accelerator.is_main_process and step % config.train.log_interval == 0:
                    metrics['lr'] = optimizer.param_groups[0]['lr']
                    logging.info(f"Step {step}: {metrics}")
                    wandb.log(metrics, step=step)

                step += 1

        logging.info("훈련 완료!")

    except Exception as e:
        logging.error(f"훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
