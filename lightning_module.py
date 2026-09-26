import torch, pytorch_lightning as pl
from network import RIRNetwork
from loss import compute_loss
from utils import (
    calculate_t60_percentage,
    calculate_c50_error,
    calculate_edt_error,
    if_to_phase,
    reconstruct_sound,
)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import io
import numpy as np
import random
from PIL import Image


class RIRLightning(pl.LightningModule):
    """Lightning wrapper that exactly matches the legacy **train.py** behaviour.

    * Randomly samples **N = cfg.sample_cols** STFT columns (with replacement)
      in ``training_step``.
    * Pulls μ/σ stats from the dataset *once* (``on_train_start``) and fully
      de‑normalises *before* reconstructing audio or computing perceptual metrics.
    * Keeps the twin‑optimiser scheme (Mag‑Net / Phase‑Net) so you can still use
      separate LRs and schedulers.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        n_freq = cfg.n_fft // 2 + 1
        network_kwargs = {
            "L": cfg.L,
            "h": cfg.h,
            "F": n_freq,
            "device": self.device,
            "n_rays": cfg.n_rays,
            "n_occlusion": cfg.n_occlusion,
        }
        self.mag_net = RIRNetwork(**network_kwargs)
        self.phase_net = RIRNetwork(**network_kwargs)

        self.alpha = cfg.alpha
        self.automatic_optimization = False

        self.register_buffer("avg_log_mag", torch.empty(0))
        self.register_buffer("std_log_mag", torch.empty(0))
        self.register_buffer("avg_phase", torch.empty(0))
        self.register_buffer("std_phase", torch.empty(0))

    def on_train_start(self):
        ds = self.trainer.datamodule.train_dataloader().dataset
        with torch.no_grad():
            self.avg_log_mag.resize_(ds.avg_log_mag.shape).copy_(
                torch.tensor(ds.avg_log_mag, device=self.device)
            )
            self.std_log_mag.resize_(ds.std_log_mag.shape).copy_(
                torch.tensor(ds.std_log_mag, device=self.device)
            )
            self.avg_phase.resize_(ds.avg_phase.shape).copy_(
                torch.tensor(ds.avg_phase, device=self.device)
            )
            self.std_phase.resize_(ds.std_phase.shape).copy_(
                torch.tensor(ds.std_phase, device=self.device)
            )

    def on_load_checkpoint(self, checkpoint):
        ds = self.trainer.datamodule.train_dataloader().dataset
        with torch.no_grad():
            self.avg_log_mag = torch.tensor(ds.avg_log_mag, device=self.device)
            self.std_log_mag = torch.tensor(ds.std_log_mag, device=self.device)
            self.avg_phase = torch.tensor(ds.avg_phase, device=self.device)
            self.std_phase = torch.tensor(ds.std_phase, device=self.device)

    def _plot_spectrogram_grid(self, gt_batch, pred_batch, tag_prefix, step):
        n_rows = min(4, gt_batch.size(0))
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 9))
        idxs = random.sample(range(gt_batch.size(0)), k=n_rows)

        for r, idx in enumerate(idxs):
            gt = gt_batch[idx].cpu()
            pd = pred_batch[idx].detach().cpu()
            axs[r, 0].imshow(gt[0], aspect="auto", origin="lower", cmap="viridis")
            axs[r, 0].set_title("GT‑L")
            axs[r, 0].axis("off")
            axs[r, 1].imshow(pd[0], aspect="auto", origin="lower", cmap="viridis")
            axs[r, 1].set_title("Pred‑L")
            axs[r, 1].axis("off")
            # axs[r, 2].imshow(gt[1], aspect="auto", origin="lower", cmap="viridis"); axs[r, 2].set_title("GT‑R"); axs[r, 2].axis("off")
            # axs[r, 3].imshow(pd[1], aspect="auto", origin="lower", cmap="viridis"); axs[r, 3].set_title("Pred‑R"); axs[r, 3].axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = np.array(Image.open(buf)).transpose(2, 0, 1)
        self.logger.experiment.add_image(f"{tag_prefix}/samples", img, step)
        plt.close(fig)

    def _denorm(self, log_mag, phase, indices):
        if self.avg_log_mag.numel() == 0:
            return log_mag, phase

        μm, σm = self.avg_log_mag[..., indices], self.std_log_mag[..., indices]
        μp, σp = self.avg_phase[..., indices], self.std_phase[..., indices]
        denorm_mag = log_mag * (3 * σm + 1e-8) + μm
        denorm_phase = phase * (3 * σp + 1e-8) + μp
        return denorm_mag, denorm_phase

    def _forward_full_stft(self, net, tx, rx, tfeat, rfeat, n_time):
        outs = []
        for t in range(n_time):
            tt = torch.tensor(t, device=self.device)
            mag = net(tx, rx, tfeat, rfeat, tt)
            # pha = net(tx, rx, tfeat, rfeat, tt, torch.tensor(1, device=self.device))
            outs.append(torch.stack([mag], dim=1))
        return torch.stack(outs, dim=-1)

    def _forward_indexed_stft(self, net, tx, rx, orient, tfeat, rfeat, indices):
        outs = []
        for t in indices:
            tt = torch.as_tensor(int(t), device=self.device)
            mag = net(tx, rx, tfeat, rfeat, tt)
            pha = net(tx, rx, tfeat, rfeat, tt)
            outs.append(torch.stack([mag, pha], dim=1))
        return torch.stack(outs, dim=-1)

    def training_step(self, batch, batch_idx):
        log_mag_gt, phase_gt, tx, rx, orient, tfeat, rfeat = batch
        _, _, _, T = log_mag_gt.shape

        idx_full = torch.arange(T, device=self.device)  # 0 … 48
        N = min(getattr(self.hparams, "sample_cols", 16), T)
        mask_idx = torch.randperm(T, device=self.device)[:N]  # shuffled subset

        lm_gt_full, ph_gt_full = (
            log_mag_gt[:, 0, :, :].unsqueeze(1),
            phase_gt[:, 0, :, :].unsqueeze(1),
        )
        lm_gt_mask, ph_gt_mask = lm_gt_full[..., mask_idx], ph_gt_full[..., mask_idx]

        opt_mag, opt_ph = self.optimizers()

        opt_mag.zero_grad()
        lm_pred_full = self._forward_full_stft(self.mag_net, tx, rx, tfeat, rfeat, T)
        loss_mag_spec = compute_loss(lm_pred_full[..., mask_idx], lm_gt_mask)
        dm_pred_full, _ = self._denorm(lm_pred_full, ph_gt_full, idx_full)
        dm_gt_full, dph_gt_f = self._denorm(lm_gt_full, ph_gt_full, idx_full)
        dph_gt_f = if_to_phase(dph_gt_f)

        audio_pred_mag = reconstruct_sound(
            dm_pred_full, dph_gt_f, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )
        audio_gt = reconstruct_sound(
            dm_gt_full, dph_gt_f, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )

        t60_mag = calculate_t60_percentage(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        c50_mag = calculate_c50_error(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        edt_mag = calculate_edt_error(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        loss_mag = loss_mag_spec + self.alpha * t60_mag

        self.manual_backward(loss_mag)
        opt_mag.step()

        opt_ph.zero_grad()
        ph_pred_if_full = self._forward_full_stft(
            self.phase_net, tx, rx, tfeat, rfeat, T
        )
        loss_ph_spec = compute_loss(ph_pred_if_full[..., mask_idx], ph_gt_mask)

        _, dph_pred_f = self._denorm(lm_gt_full, ph_pred_if_full, idx_full)
        dph_pred_f = if_to_phase(dph_pred_f)
        audio_pred_ph = reconstruct_sound(
            dm_gt_full, dph_pred_f, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )

        t60_ph = calculate_t60_percentage(
            audio_pred_ph, audio_gt, sr=self.hparams.sr
        ).mean()
        c50_ph = calculate_c50_error(audio_pred_ph, audio_gt, sr=self.hparams.sr).mean()
        edt_ph = calculate_edt_error(audio_pred_ph, audio_gt, sr=self.hparams.sr).mean()

        loss_ph = loss_ph_spec + self.alpha * t60_ph

        self.manual_backward(loss_ph)
        opt_ph.step()

        self.log_dict(
            {
                "train/loss_mag": loss_mag,
                "train/loss_ph": loss_ph,
                "train/t60_mag": 100 * t60_mag,
                "train/c50_mag": c50_mag,
                "train/edt_mag": edt_mag,
                "train/t60_ph": 100 * t60_ph,
                "train/c50_ph": c50_ph,
                "train/edt_ph": edt_ph,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx % self.hparams.tb_image_freq == 0:
            self._plot_spectrogram_grid(
                dm_gt_full, dm_pred_full, "train", self.global_step
            )

        return loss_mag + loss_ph

    def _shared_step(self, batch, stage):
        """Shared logic for validation_step and test_step."""
        log_mag_gt, phase_gt, tx, rx, orient, tfeat, rfeat = batch
        log_mag_gt = log_mag_gt[:, 0, ...].unsqueeze(1)  # (B, 1, F, T)
        phase_gt = phase_gt[:, 0, ...].unsqueeze(1)  # (B, 1, F, T)
        B, C, F, T = log_mag_gt.shape

        lm_pred = self._forward_full_stft(self.mag_net, tx, rx, tfeat, rfeat, T)
        ph_pred_if = self._forward_full_stft(self.phase_net, tx, rx, tfeat, rfeat, T)

        loss_mag_spec = compute_loss(lm_pred, log_mag_gt)
        loss_ph_spec = compute_loss(ph_pred_if, phase_gt)

        idx_full = torch.arange(T, device=self.device)
        dm_pred, _ = self._denorm(lm_pred, phase_gt, idx_full)
        dm_gt, dph_gt = self._denorm(log_mag_gt, phase_gt, idx_full)
        dph_gt = if_to_phase(dph_gt)
        _, dph_pred = self._denorm(log_mag_gt, ph_pred_if, idx_full)
        dph_pred = if_to_phase(dph_pred)

        audio_pred_mag = reconstruct_sound(
            dm_pred, dph_gt, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )
        audio_pred_ph = reconstruct_sound(
            dm_gt, dph_pred, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )
        audio_gt = reconstruct_sound(
            dm_gt, dph_gt, sr=self.hparams.sr, n_fft=self.hparams.n_fft
        )

        t60_mag = calculate_t60_percentage(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        c50_mag = calculate_c50_error(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        edt_mag = calculate_edt_error(
            audio_pred_mag, audio_gt, sr=self.hparams.sr
        ).mean()
        t60_ph = calculate_t60_percentage(
            audio_pred_ph, audio_gt, sr=self.hparams.sr
        ).mean()
        c50_ph = calculate_c50_error(audio_pred_ph, audio_gt, sr=self.hparams.sr).mean()
        edt_ph = calculate_edt_error(audio_pred_ph, audio_gt, sr=self.hparams.sr).mean()

        self.log_dict(
            {
                f"{stage}/loss_mag": loss_mag_spec,
                f"{stage}/loss_ph": loss_ph_spec,
                f"{stage}/t60_mag": 100 * t60_mag,
                f"{stage}/c50_mag": c50_mag,
                f"{stage}/edt_mag": edt_mag,
                f"{stage}/t60_ph": 100 * t60_ph,
                f"{stage}/c50_ph": c50_ph,
                f"{stage}/edt_ph": edt_ph,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
        )

        return loss_mag_spec + loss_ph_spec

    def validation_step(self, batch, batch_idx):
        """Lightning entry point for validation; delegates to _shared_step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Lightning entry point for test; delegates to _shared_step."""
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt_mag = torch.optim.Adam(
            self.mag_net.parameters(),
            lr=self.hparams.optimizer.lr_mag,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        opt_phase = torch.optim.Adam(
            self.phase_net.parameters(),
            lr=self.hparams.optimizer.lr_phase,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        sch_mag = torch.optim.lr_scheduler.ExponentialLR(
            opt_mag, gamma=self.hparams.optimizer.lr_decay
        )
        sch_phase = torch.optim.lr_scheduler.ExponentialLR(
            opt_phase, gamma=self.hparams.optimizer.lr_decay
        )

        return ([opt_mag, opt_phase], [sch_mag, sch_phase])
