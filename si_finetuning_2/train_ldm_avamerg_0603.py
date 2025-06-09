"""
U-ViT 멀티모달 파인튜닝 스크립트
===========================================

사전 훈련된 U-ViT 모델을 멀티모달 데이터셋으로 파인튜닝하기 위한
전체 파이프라인 (NPY 데이터셋 지원)
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import copy

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
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from tools.fid_score import calculate_fid_given_paths
import utils
import libs.autoencoder
import torch.multiprocessing as mp


class NPYDataset(Dataset):
    """NPY 파일로부터 멀티모달 데이터를 로드하는 데이터셋"""

    def __init__(self, npy_root: str, linear_proj: nn.Module = None):
        """
        NPY 데이터셋 초기화

        Args:
            npy_root: NPY 파일들이 저장된 루트 디렉토리
            linear_proj: 텍스트 특징 차원 변환용 선형 투영 모델
        """
        self.npy_root = Path(npy_root)
        self.linear_proj = linear_proj

        # NPY 파일 목록 수집
        self.npy_files = list(self.npy_root.glob("*.npy"))
        if not self.npy_files:
            raise ValueError(f"NPY 파일을 찾을 수 없습니다: {npy_root}")

        logging.info(f"총 {len(self.npy_files)}개의 NPY 파일 발견")

        # 첫 번째 파일로 데이터 형태 확인
        sample_data = np.load(self.npy_files[0], allow_pickle=True).item()
        self._log_data_shapes(sample_data)

    def _log_data_shapes(self, sample_data: dict):
        """샘플 데이터의 형태 로깅"""
        for key, value in sample_data.items():
            if isinstance(value, np.ndarray):
                logging.info(f"[샘플] {key} shape: {value.shape}")

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        데이터 아이템 반환

        Returns:
            img_latent: 이미지 잠재 변수 [4, 64, 64]
            clip_feat: CLIP 이미지 특징 [512]
            text_latent: 텍스트 잠재 변수 [77, 64] (linear projection 적용 후)
        """
        try:
            # NPY 파일 로드
            data = np.load(self.npy_files[idx], allow_pickle=True).item()

            # 데이터 추출 (output 데이터 사용)
            img_latent = torch.from_numpy(data['output_img_latent']).float()  # [4, 64, 64]
            clip_feat = torch.from_numpy(data['output_clip_feat']).float()    # [512]
            text_latent = torch.from_numpy(data['output_text_latent']).float() # [77, 768]

            # image latent: z-score 정규화
            img_latent = (img_latent - img_latent.mean()) / img_latent.std()
            # clip_feat, text_latent: L2 정규화
            clip_feat = F.normalize(clip_feat, dim=-1)
            text_latent = F.normalize(text_latent, dim=-1)


            # # 텍스트 특징 차원 변환 (768 -> 64)
            # if self.linear_proj is not None:
            #     self.linear_proj = self.linear_proj.cpu()
            #     with torch.no_grad():
            #         text_latent = self.linear_proj(text_latent.cpu()).cpu()

            return img_latent, clip_feat, text_latent

        except Exception as e:
            logging.error(f"데이터 로드 실패 (파일: {self.npy_files[idx]}): {e}")
            # 에러 발생 시 첫 번째 파일로 대체
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
        # print("🦖🦖🦖🦖🦖🦖🦖🦖🦖xn_img:", xn_img.shape)
        # print("🦖🦖🦖🦖🦖🦖🦖🦖🦖xn_clip:", xn_clip.shape)
        # print("🦖🦖🦖🦖🦖🦖🦖🦖🦖xn_text:", yn_text.shape)

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


class MultimodalLoss:
    """멀티모달 U-ViT 손실 함수 클래스"""

    @staticmethod
    def compute_loss(img_latent: torch.Tensor,
                     clip_feat: torch.Tensor,
                     text_latent: torch.Tensor,
                     model: nn.Module,
                     scheduler: NoiseScheduler,
                     img_channels: int,
                     **kwargs):
        # 텍스트 특징 차원 변환: [batch, 77, 768] -> [batch, 77, 64]
        if text_latent.shape[-1] == 768:
            # 안전한 차원 변환
            original_shape = text_latent.shape  # [batch, 77, 768]
            text_flat = text_latent.view(-1, 768)  # [batch*77, 768]
            text_projected = model.linear_proj(text_flat)  # [batch*77, 64]
            text_latent = text_projected.view(original_shape[0], original_shape[1], -1)  # [batch, 77, 64]







        # 멀티모달 노이즈 샘플링
        (n_img, n_clip, n_text), (eps_img, eps_clip, eps_text), (
        x_img_noised, x_clip_noised, x_text_noised) = scheduler.sample_multimodal(img_latent, clip_feat, text_latent)

        # data_type: 어떤 modality로 학습할지 결정- randint로 무작위 선택 (0=img only, 1=text only, 2=multimodal)
        data_type = torch.randint(0, 3, (img_latent.size(0),), device=img_latent.device)

        # 모델 예측
        pred_img, pred_clip, pred_text = model(
            img=x_img_noised,
            clip_img=x_clip_noised,
            text=x_text_noised,
            t_img=n_img,
            t_text=n_text,
            data_type=data_type
        )

        loss_img = MultimodalLoss._mean_squared_error(eps_img - pred_img)
        loss_clip = MultimodalLoss._mean_squared_error(eps_clip - pred_clip)*0.05
        loss_text = MultimodalLoss._mean_squared_error(eps_text - pred_text)

        return loss_img, loss_clip, loss_text

    @staticmethod
    def _mean_squared_error(tensor: torch.Tensor, start_dim: int = 1) -> torch.Tensor:
        """평균 제곱 오차 계산"""
        return tensor.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class ModelManager:
    """모델 로딩 및 관리를 위한 클래스"""

    @staticmethod
    def load_uvit_model(model_path: str, config: ml_collections.ConfigDict) -> nn.Module:
        """사전 훈련된 U-ViT 모델 로드"""
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
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)

            # token_embedding.weight 크기 문제 처리 -
            if 'token_embedding.weight' in state_dict:
                old_emb = state_dict['token_embedding.weight']      # torch.Size([2, 1536]) - error 발생
                expected_shape = uvit.token_embedding.weight.shape  # torch.Size([3, 1536])
                if old_emb.shape != expected_shape:
                    print(f"token_embedding.weight 크기 mismatch: checkpoint {old_emb.shape} -> model {expected_shape}")
                    # 새 임베딩 텐서 초기화
                    new_emb = torch.zeros(expected_shape)
                    # 기존 임베딩 weight 복사 (가능한 만큼)
                    n_copy = min(old_emb.shape[0], expected_shape[0])
                    new_emb[:n_copy] = old_emb[:n_copy]
                    # 나머지 임베딩은 랜덤 초기화 (표준편차 0.02)
                    if expected_shape[0] > n_copy:
                        new_emb[n_copy:] = torch.randn(expected_shape[0] - n_copy, expected_shape[1]) * 0.02

                    state_dict['token_embedding.weight'] = new_emb

            uvit.load_state_dict(state_dict, strict=False)
            logging.info(f"모델 로드 완료: {model_path}")
        else:
            logging.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 선택적 모델 고정
        ModelManager._freeze_model_layers(uvit)

        return ModelManager.UViTWrapper(uvit)

    @staticmethod
    def _freeze_model_layers(model: nn.Module) -> None:
        """메모리 절약, finetuning을 위한 선택적 레이어 고정"""
        total_params = 0
        frozen_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()

            # # 출력 레이어와 마지막 3개 블록만 학습
            # if any(keyword in name for keyword in [
            #     'decoder_pred', 'clip_img_out', 'text_out',
            #     'blocks.29', 'blocks.28', 'blocks.27'  # depth=30이므로 29, 28, 27
            # ]):
            # 학습할 레이어 선택
            if any(keyword in name for keyword in [
                'decoder_pred', 'clip_img_out', 'text_out',
                'mid_block',
                'out_blocks.13', 'out_blocks.14',
                'in_blocks.13', 'in_blocks.14',
                'token_embedding'
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_params += param.numel()

            # layer grad 고정 확인
            if param.requires_grad:
                if param.grad is not None:
                    print(f"[✅] {name} has grad with mean={param.grad.abs().mean():.6f}")
                else:
                    print(f"[❌] {name} has NO grad!")

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
            self.linear_proj = nn.Linear(768, 64)   # 학습 가능한 text linear proj

        def forward(self, img: torch.Tensor, clip_img: torch.Tensor,
                    text: torch.Tensor, t_img: torch.Tensor,
                    t_text: torch.Tensor, data_type: torch.Tensor) -> Tuple:
            """모델 순전파"""
            return self.uvit(img, clip_img, text, t_img, t_text, data_type)


class TrainingManager:
    """훈련 과정 관리 클래스"""

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
                project=f'uvit_finetune_{self.config.dataset.name}',
                config=self.config.to_dict(),
                name='finetune_avamerg',
                job_type='finetune',
                mode='online'
            )

            # 로거 설정
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('workdir/training.log'),
                    logging.StreamHandler()
                ]
            )
            logging.info("훈련 설정:")
            logging.info(self.config)
        else:
            logging.basicConfig(level=logging.ERROR)

    def setup_data_loader(self) -> DataLoader:
        """데이터 로더 설정"""
        try:
            # Linear projection 모델 생성 (768 -> 64 차원 변환)
            linear_proj = nn.Linear(768, 64).to(self.device)

            # NPY 데이터셋 생성
            dataset = NPYDataset(
                npy_root=self.config.dataset.train_npy_root,
                # linear_proj=linear_proj
                linear_proj=None  # UViTWrapper에서 정의했으므로 여기는 제거함
            )

            return DataLoader(
                dataset,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

        except Exception as e:
            logging.error(f"데이터 로더 설정 중 오류 발생: {e}")
            raise

    def train_step(self, batch: Tuple, model: nn.Module,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """단일 훈련 스텝 수행"""
        metrics = {}
        optimizer.zero_grad()

        try:
            img_latents, clip_features, text_features = batch
            img_latents = img_latents.to(self.device)
            clip_features = clip_features.to(self.device)
            text_features = text_features.to(self.device)

            # 멀티모달 손실 계산
            with self.accelerator.autocast():
                loss_img, loss_clip, loss_text = MultimodalLoss.compute_loss(
                    img_latent=img_latents,
                    clip_feat=clip_features,
                    text_latent=text_features,
                    model=model,
                    scheduler=self.scheduler,
                    img_channels=img_latents.shape[1]
                )
            loss = loss_img + loss_clip + loss_text

            # 역전파
            self.accelerator.backward(loss.mean())

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        print(f"[⭐⭐⭐✅] {name} has grad with mean={param.grad.abs().mean():.6f}")
                    else:
                        print(f"[⭐⭐⭐❌] {name} has NO grad!")




            # ✅ wandb용 metrics 정리
            metrics['loss'] = self.accelerator.gather(loss.detach()).mean().item()
            metrics['loss_img'] = self.accelerator.gather(loss_img.detach()).mean().item()
            metrics['loss_clip'] = self.accelerator.gather(loss_clip.detach()).mean().item()
            metrics['loss_text'] = self.accelerator.gather(loss_text.detach()).mean().item()

            # 그래디언트 클리핑
            if hasattr(self.config.train, 'grad_clip') and self.config.train.grad_clip > 0:
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
                               img_latents: torch.Tensor) -> torch.Tensor:
        """CLIP 특징을 이미지 잠재 변수와 같은 공간 차원으로 변환"""
        # [B, D] → [B, D, 1, 1] → [B, D, H, W]
        clip_features = clip_features.unsqueeze(-1).unsqueeze(-1)
        clip_features = clip_features.expand(-1, -1, img_latents.shape[2], img_latents.shape[3])
        return clip_features

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
            checkpoint = {
                'model': self.accelerator.unwrap_model(model).state_dict(),
                'ema_model': self.accelerator.unwrap_model(ema_model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': self.config.to_dict()
            }
            save_path = f'checkpoint_step_{step}.pth'
            torch.save(checkpoint, save_path)
            logging.info(f"체크포인트 저장 완료: {save_path}")


def main():
    """메인 함수"""
    try:
        # 설정 로드 (import 경로 수정 필요할 수 있음)
        logging.info(f"get_config 진입")
        from configs.finetune_uvit_config import get_config
        config = get_config()
        logging.info(f"TrainingManager 진입")
        # 훈련 매니저 초기화
        trainer = TrainingManager(config)
        logging.info(f"data_loader, load_uvit_model 진입")
        # 모델 및 데이터 설정
        data_loader = trainer.setup_data_loader()
        model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
        logging.info(f"ema_model 진입")
        # EMA 모델 생성
        ema_model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
        ema_model.eval()
        logging.info(f"옵티마이저 설정, trainable_params 진입")
        # 옵티마이저 설정
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        #print(f"⭐⭐⭐⭐Trainable parameters: {trainable_params}")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas
        )
        logging.info(f"lr_scheduler 진입")
        # 학습률 스케줄러
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / config.lr_scheduler.warmup_steps, 1.0)
        )
        logging.info(f"분산학습준비 진입")
        # 분산 훈련 준비
        model, ema_model, optimizer, data_loader, lr_scheduler = trainer.accelerator.prepare(
            model, ema_model, optimizer, data_loader, lr_scheduler
        )

        # 훈련 시작
        logging.info("U-ViT 파인튜닝 시작")
        step = 0
        model.train()

        while step < config.train.n_steps:
            for batch in data_loader:
                if step >= config.train.n_steps:
                    break

                # 훈련 스텝 수행
                metrics = trainer.train_step(batch, model, optimizer)

                # EMA 업데이트
                TrainingManager.update_ema(ema_model, model, config.get('ema_rate', 0.9999))

                # 학습률 업데이트
                lr_scheduler.step()

                # 로깅
                if trainer.accelerator.is_main_process and step % config.train.log_interval == 0:
                    metrics['lr'] = optimizer.param_groups[0]['lr']
                    metrics['step'] = step
                    logging.info(f"Step {step}: Loss={metrics['loss']:.6f}, loss_img={metrics['loss_img']:.4f}, "
                                 f"loss_clip={metrics['loss_clip']:.4f}, loss_text={metrics['loss_text']:.4f}, "
                                 f"LR={metrics['lr']:.7f}")
                    wandb.log(metrics, step=step)

                # 체크포인트 저장
                if step % config.train.save_interval == 0 and step > 0:
                    trainer.save_checkpoint(model, ema_model, optimizer, step)

                step += 1

        # 최종 체크포인트 저장
        trainer.save_checkpoint(model, ema_model, optimizer, step)
        logging.info("훈련 완료!")

    except Exception as e:
        logging.error(f"훈련 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
            logging.FileHandler("train.log")  # 파일 로그
        ]
    )
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        logging.info(f"Process {rank} started.")
    else:
        logging.info("Single-process start.")
    logging.info("mp 진입")
    import sys
    sys.stdout.flush()
    mp.set_start_method("spawn", force=True)
    logging.info("main 진입")
    main()
