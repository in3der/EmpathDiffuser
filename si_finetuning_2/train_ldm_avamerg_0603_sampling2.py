import torch
import torch.nn as nn
import wandb
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import ml_collections
import accelerate
import torch.multiprocessing as mp
from dataclasses import dataclass

from libs.uvit_multi_post_ln_v1 import UViT
from tools.fid_score import calculate_fid_given_paths
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from training_components import NPYDataset, NoiseScheduler, MultimodalLoss, ModelManager, WandbSampler
import os


class TrainingManager:
    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config
        self.accelerator = accelerate.Accelerator(mixed_precision=config.mixed_precision)
        self.device = self.accelerator.device
        self.scheduler = NoiseScheduler()
        self.wandb_sampler = WandbSampler(autoencoder=None)
        self.setup_logging()

    def setup_logging(self):
        if self.accelerator.is_main_process:
            wandb.init(
                dir=os.path.join('workdir'),
                project=f'uvit_finetune_{self.config.dataset.name}',
                config=self.config.to_dict(),
                name='finetune_avamerg_npy',
                job_type='finetune',
                mode='online'
            )
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('workdir/training.log'),
                    logging.StreamHandler()
                ]
            )

    def setup_data_loader(self) -> DataLoader:
        linear_proj = nn.Linear(768, 64).to(self.device)
        dataset = NPYDataset(self.config.dataset.train_npy_root, linear_proj)
        return DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_step(self, batch, model, optimizer):
        optimizer.zero_grad()
        img_latents, clip_features, text_features = [x.to(self.device) for x in batch]

        with self.accelerator.autocast():
            loss = MultimodalLoss.compute_loss(
                img_latent=img_latents,
                clip_feat=clip_features,
                text_latent=text_features,
                model=model,
                scheduler=self.scheduler,
                img_channels=img_latents.shape[1]
            )

        self.accelerator.backward(loss.mean())
        if self.config.train.grad_clip > 0:
            self.accelerator.clip_grad_norm_(model.parameters(), self.config.train.grad_clip)

        optimizer.step()
        return {'loss': self.accelerator.gather(loss.detach()).mean().item()}

    @staticmethod
    def update_ema(ema_model, model, decay):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def save_checkpoint(self, model, ema_model, optimizer, step):
        if self.accelerator.is_main_process:
            checkpoint = {
                'model': self.accelerator.unwrap_model(model).state_dict(),
                'ema_model': self.accelerator.unwrap_model(ema_model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': self.config.to_dict()
            }
            torch.save(checkpoint, f'checkpoint_step_{step}.pth')

    def train(self, model, ema_model, optimizer, scheduler, dataloader):
        model.train()
        step = 0
        pbar = tqdm(total=self.config.train.n_steps, desc="🔥 Training")

        while step < self.config.train.n_steps:
            for batch in dataloader:
                if step >= self.config.train.n_steps:
                    break

                metrics = self.train_step(batch, model, optimizer)
                self.update_ema(ema_model, model, self.config.train.ema_rate)
                scheduler.step()

                if self.accelerator.is_main_process and step % self.config.train.log_interval == 0:
                    metrics['lr'] = optimizer.param_groups[0]['lr']
                    metrics['step'] = step
                    logging.info(f"[Step {step}] Loss: {metrics['loss']:.4f}")
                    wandb.log(metrics, step=step)

                if step % self.config.train.eval_interval == 0:
                    self.wandb_sampler.sample_and_log_images(model, self.scheduler, step, device=self.device)
                    self.sample_ti2ti_step(model, step)

                if step % self.config.train.save_interval == 0 and step > 0:
                    self.save_checkpoint(model, ema_model, optimizer, step)

                step += 1
                pbar.update(1)

        pbar.close()
        self.save_checkpoint(model, ema_model, optimizer, step)

    def sample_ti2ti_step(self, model, step: int):
        from configs.finetune_uvit_config import get_config
        config = get_config()
        model.eval()

        num_samples = self.config.sample.n_samples
        device = self.device
        text_dim = self.config.nnet.text_dim
        clip_dim = self.config.nnet.clip_img_dim
        z_shape = (4, 64, 64)  # AvaMERG latent image shape

        # random text latent → image → text 재구성
        text_latent = torch.randn(num_samples, 77, text_dim, device=device)
        clip_noise = torch.randn(num_samples, clip_dim, device=device)
        img_noise = torch.randn(num_samples, *z_shape, device=device)

        from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
        betas = self.scheduler._betas
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(betas).float().to(device))

        N = len(betas)

        def combine(z, clip_img):
            return torch.cat([z.flatten(1), clip_img], dim=1)

        def split(x):
            z_dim = z_shape[0] * z_shape[1] * z_shape[2]
            z = x[:, :z_dim].view(-1, *z_shape)
            clip = x[:, z_dim:]
            return z, clip

        def t2i_nnet(x, t, text):
            z, clip = split(x)
            t_img = t
            t_text = torch.zeros_like(t)

            pred_z, pred_clip, _ = model(
                img=z, clip_img=clip, text=text,
                t_img=t_img, t_text=t_text,
                data_type=torch.full_like(t_img, 2)
            )
            return combine(pred_z, pred_clip)

        def i2t_nnet(x, t, z, clip):
            t_img = torch.zeros_like(t)
            t_text = t

            _, _, pred_text = model(
                img=z, clip_img=clip, text=x,
                t_img=t_img, t_text=t_text,
                data_type=torch.full_like(t, 2)
            )
            return pred_text

        # text → image
        x_init = combine(img_noise, clip_noise)

        dpm_solver_i = DPM_Solver(
            model_fn=lambda x, t_cont: t2i_nnet(x, t_cont * N, text_latent),
            noise_schedule=noise_schedule,
            predict_x0=True, thresholding=False
        )
        with torch.no_grad():
            x_sampled = dpm_solver_i.sample(x_init, steps=self.config.sample.sample_steps, eps=1. / N, T=1.)
            z_gen, clip_gen = split(x_sampled)

        # image → text
        dpm_solver_t = DPM_Solver(
            model_fn=lambda x, t_cont: i2t_nnet(x, t_cont * N, z_gen, clip_gen),
            noise_schedule=noise_schedule,
            predict_x0=True, thresholding=False
        )
        text_sampled = dpm_solver_t.sample(text_latent, steps=self.config.sample.sample_steps, eps=1. / N, T=1.)

        # wandb logging
        from libs.caption_decoder import CaptionDecoder
        caption_decoder = CaptionDecoder(device=self.device, pretrained_path=config.caption_decoder)

        for idx, cap in enumerate(caption_decoder):
            wandb.log({f"ti2ti/sample_{idx}": cap}, step=step)

        if self.wandb_sampler.autoencoder is not None:
            decoded_imgs = self.wandb_sampler._decode_latents(z_gen)
            wandb_images = [
                wandb.Image(self.wandb_sampler.to_pil(img.clamp(0, 1)), caption=f"Sample {idx}")
                for idx, img in enumerate(decoded_imgs)
            ]
            wandb.log({"ti2ti/generated_images": wandb_images}, step=step)


def main():
    from configs.finetune_uvit_config import get_config
    config = get_config()

    trainer = TrainingManager(config)
    dataloader = trainer.setup_data_loader()
    model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
    ema_model = ModelManager.load_uvit_model(config.nnet.pretrained_path, config)
    ema_model.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr,
                                  weight_decay=config.optimizer.weight_decay,
                                  betas=config.optimizer.betas)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / config.lr_scheduler.warmup_steps, 1.0))

    model, ema_model, optimizer, dataloader, scheduler = trainer.accelerator.prepare(
        model, ema_model, optimizer, dataloader, scheduler
    )

    trainer.train(model, ema_model, optimizer, scheduler, dataloader)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
