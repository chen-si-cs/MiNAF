from __future__ import annotations
from matplotlib import pyplot as plt
import numpy as np
import hydra, torch, pytorch_lightning as pl

import time
import statistics as _stats

from omegaconf import DictConfig
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from datamodule import SoundDataModule
from lightning_module import RIRLightning
from utils import (
    reconstruct_sound,
    reconstruct_sound_with_random_phase,
    reconstruct_sound_with_griffin_lim,
    calculate_t60_percentage,
    calculate_t60_absolute_error,
    calculate_edt_error,
    calculate_edt_relative_error,
    calculate_c50_error,
    compute_snr,
    compute_psnr,
    if_to_phase,
)
from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"

CASE_NAMES = [
    "1-GTP",  # Pred Mag  + GT Phase
    "2-GTM+PreP",  # GT  Mag   + Pred Phase
    "3-PreP",  # Pred Mag  + Pred Phase
    "4-RanP",  # Pred Mag  + Random Phase
    "5-GLim1",  # Pred Mag  + Griffin-Lim  (compare to GT-GL)
    "5-GLim2",  # Pred Mag  + Griffin-Lim  (compare to GT)
]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    def _stft_to_audio(
        mag: torch.Tensor, pha: torch.Tensor, glim=False
    ) -> torch.Tensor:
        """Wrap `reconstruct_sound` for a *single* sample (C×F×T) → (2,T)."""
        if glim:
            out = reconstruct_sound_with_griffin_lim(
                mag.unsqueeze(0), pha.unsqueeze(0), sr=cfg.sr, n_fft=cfg.n_fft
            )
            return out.squeeze(0)
        else:
            out = reconstruct_sound(
                mag.unsqueeze(0), pha.unsqueeze(0), sr=cfg.sr, n_fft=cfg.n_fft
            )
            return out.squeeze(0)  # drop batch dim → (2, T)

    if not cfg.ckpt_path:
        raise ValueError("Set ckpt_path in conf/config.yaml or on the command line")

    ckpt_path = Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = ckpt_path.parent

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    buf_shape = state_dict["avg_log_mag"].shape

    model = RIRLightning(cfg)

    for name in ("avg_log_mag", "std_log_mag", "avg_phase", "std_phase"):
        getattr(model, name).resize_(buf_shape)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    data = SoundDataModule(cfg)
    data.prepare_data()
    data.setup(stage="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of Parameters: {num_params}")

    gen_times: list[float] = []
    frames_generated: int = 0

    sums = defaultdict(float)
    counts = defaultdict(int)

    loader = data.test_dataloader()
    n_freq = cfg.n_fft // 2 + 1

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            (log_mag_gt, phase_gt, tx, rx, orient, tfeat, rfeat) = [
                b.to(device, non_blocking=True) for b in batch
            ]

            B, C, F, T = log_mag_gt.shape
            log_mag_pred = torch.zeros_like(log_mag_gt)
            phase_pred = torch.zeros_like(phase_gt)

            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

            for t in range(T):
                tt = torch.tensor(t, device=device)
                # channel 0 = L, 1 = R
                for ch in (0, 1):
                    chan_t = torch.tensor(ch, device=device)
                    log_mag_pred[:, ch, :, t] = model.mag_net(tx, rx, tfeat, rfeat, tt)
                    phase_pred[:, ch, :, t] = model.phase_net(tx, rx, tfeat, rfeat, tt)

            if device.type == "cuda":
                torch.cuda.synchronize()
            gen_times.append(time.perf_counter() - _t0)
            frames_generated += int(B * T * 2)

            idx_full = torch.arange(T, device=device)

            # Predicted magnitude & phase
            denorm_mag_pred, denorm_phase_pred = model._denorm(
                log_mag_pred, phase_pred, idx_full
            )

            # Ground-truth magnitude & phase
            denorm_mag_gt, denorm_phase_gt = model._denorm(
                log_mag_gt, phase_gt, idx_full
            )

            # Convert phase from IF → wrapped phase (-π, π]
            denorm_phase_pred = if_to_phase(denorm_phase_pred)
            denorm_phase_gt = if_to_phase(denorm_phase_gt)

            def update(
                case: str, pred_audio: torch.Tensor, gt_audio: torch.Tensor
            ) -> None:
                nonlocal sums, counts
                sums[f"{case}_t60"] += (
                    calculate_t60_percentage(pred_audio, gt_audio, sr=cfg.sr)
                    .mean()
                    .item()
                )
                sums[f"{case}_edt"] += (
                    calculate_edt_error(pred_audio, gt_audio, sr=cfg.sr).mean().item()
                )
                sums[f"{case}_c50"] += (
                    calculate_c50_error(pred_audio, gt_audio, sr=cfg.sr).mean().item()
                )
                sums[f"{case}_snr"] += compute_snr(pred_audio, gt_audio).mean().item()
                sums[f"{case}_psnr"] += compute_psnr(pred_audio, gt_audio).mean().item()

                sums[f"{case}_t60_abs"] += (
                    calculate_t60_absolute_error(pred_audio, gt_audio, sr=cfg.sr)
                    .mean()
                    .item()
                )
                sums[f"{case}_edt_rel"] += (
                    calculate_edt_relative_error(pred_audio, gt_audio, sr=cfg.sr)
                    .mean()
                    .item()
                )
                counts[case] += 1

            # Case 1  (Pred Mag + GT Phase)
            audio_pred = reconstruct_sound(
                denorm_mag_pred, denorm_phase_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            audio_gt = reconstruct_sound(
                denorm_mag_gt, denorm_phase_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            update("c1", audio_pred, audio_gt)

            # Case 2  (GT Mag + Pred Phase)
            audio_pred = reconstruct_sound(
                denorm_mag_gt, denorm_phase_pred, sr=cfg.sr, n_fft=cfg.n_fft
            )
            update("c2", audio_pred, audio_gt)

            # Case 3  (Pred Mag + Pred Phase)
            audio_pred = reconstruct_sound(
                denorm_mag_pred, denorm_phase_pred, sr=cfg.sr, n_fft=cfg.n_fft
            )
            update("c3", audio_pred, audio_gt)
            prep_audio = audio_pred.clone()

            # Case 4  (Pred Mag + Random Phase)
            audio_pred, audio_gt_rnd = reconstruct_sound_with_random_phase(
                denorm_mag_pred, denorm_mag_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            update("c4", audio_pred, audio_gt_rnd)

            # Case 5a (Pred Mag + Griffin-Lim   vs   GT-GL)
            audio_pred, audio_gt_gl = reconstruct_sound_with_griffin_lim(
                denorm_mag_pred, denorm_mag_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            update("c5a", audio_pred, audio_gt_gl)

            # Case 5b (Pred Mag + Griffin-Lim   vs   GT)
            update("c5b", audio_pred, audio_gt)

            sums["spec_mag"] += torch.nn.functional.l1_loss(
                denorm_mag_pred, denorm_mag_gt
            ).item()
            sums["spec_phase"] += torch.nn.functional.l1_loss(
                denorm_phase_pred, denorm_phase_gt
            ).item()
            counts["spec"] += 1

            if "mse_list" not in locals():
                mse_list, cache = [], {}  # ← initialise once

            mse_per_sample = (
                calculate_t60_percentage(prep_audio, audio_gt, sr=cfg.sr)
                .cpu()
                .mean(dim=1)
                + calculate_c50_error(prep_audio, audio_gt, sr=cfg.sr).cpu()
            )

            audio_pred, audio_gt_gl = reconstruct_sound_with_griffin_lim(
                denorm_mag_pred, denorm_mag_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            audio_gt = reconstruct_sound(
                denorm_mag_gt, denorm_phase_gt, sr=cfg.sr, n_fft=cfg.n_fft
            )
            for i in range(B):
                global_idx = len(mse_list)
                mse_list.append(mse_per_sample[i].item())

                cache[global_idx] = dict(
                    mag_pred=denorm_mag_pred[i].cpu(),
                    phase_pred=denorm_phase_pred[i].cpu(),
                    mag_gt=denorm_mag_gt[i].cpu(),
                    phase_gt=denorm_phase_gt[i].cpu(),
                    audio_pairs={
                        "1-GTP": (
                            _stft_to_audio(denorm_mag_pred[i], denorm_phase_gt[i]),
                            _stft_to_audio(denorm_mag_gt[i], denorm_phase_gt[i]),
                        ),
                        "2-GTM+PreP": (
                            _stft_to_audio(denorm_mag_gt[i], denorm_phase_pred[i]),
                            _stft_to_audio(denorm_mag_gt[i], denorm_phase_gt[i]),
                        ),
                        "3-PreP": (
                            _stft_to_audio(denorm_mag_pred[i], denorm_phase_pred[i]),
                            _stft_to_audio(denorm_mag_gt[i], denorm_phase_gt[i]),
                        ),
                        "4-RanP": (
                            _stft_to_audio(
                                denorm_mag_pred[i],
                                denorm_phase_pred[i] * 0
                                + torch.rand_like(denorm_phase_pred[i]),
                            ),
                            _stft_to_audio(
                                denorm_mag_gt[i],
                                denorm_phase_pred[i] * 0
                                + torch.rand_like(denorm_phase_pred[i]),
                            ),
                        ),
                        "5-GLim1": (
                            audio_pred[i],
                            audio_gt_gl[i],
                        ),
                        "5-GLim2": (
                            audio_pred[i],
                            audio_gt[i],
                        ),
                    },
                    phase_gt_wrap=denorm_phase_gt[i].cpu(),
                    phase_pred_wrap=denorm_phase_pred[i].cpu(),
                )

    mse_arr = np.asarray(mse_list)
    idx_first = 0
    idx_worst = int(mse_arr.argmax())
    idx_best = int(mse_arr.argmin())
    sel_indices = [("first", idx_first), ("worst", idx_worst), ("best", idx_best)]

    def plot_comparison_heatmap(sample_key: str, kind: str):
        """Plot GT vs Pred for magnitude or phase (not all 6 cases)."""
        arr_pred = cache[sample_key][f"{kind}_pred"].numpy()
        arr_gt = cache[sample_key][f"{kind}_gt"].numpy()

        fig, axs = plt.subplots(
            nrows=2, ncols=2, figsize=(6, 3.5)
        )  # [L, R] x [GT, Pred]
        vmax = max(arr_pred.max(), arr_gt.max())
        vmin = min(arr_pred.min(), arr_gt.min())
        for ch in range(2):
            axs[0, ch].imshow(
                arr_gt[ch],
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            axs[0, ch].set_title(f"GT - {'L' if ch == 0 else 'R'}", fontsize=8)
            axs[1, ch].imshow(
                arr_pred[ch],
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            axs[1, ch].set_title(f"Pred - {'L' if ch == 0 else 'R'}", fontsize=8)
            for ax in [axs[0, ch], axs[1, ch]]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"{kind.upper()} comparison • sample: {sample_key}", fontsize=10)
        fig.tight_layout()
        fname = output_dir / f"spec_{kind}_{sample_key}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"✔ saved {fname}")

    def plot_waveform_and_schroeder(gt, pred, sr, filename):
        time = np.arange(len(gt[0])) / sr
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        handles_combined = []
        labels_combined = []

        all_signals = [gt[0], gt[1], pred[0], pred[1]]
        max_map = max(np.max(np.abs(signal)) for signal in all_signals)

        for ch, side in enumerate(["Left", "Right"]):
            for i, signal in enumerate([gt[ch], pred[ch]]):
                row = ch * 2 + i
                ax_wave = axs[row]
                label = "Ground Truth" if i == 0 else "Prediction"
                color = "orange" if i == 0 else "purple"

                (line1,) = ax_wave.plot(time, signal, color=color, label=label)
                ax_wave.set_ylabel("Amplitude")
                ax_wave.set_ylim(-max_map, max_map)

                ax_sch = ax_wave.twinx()
                edc = 10 * np.log10(
                    np.maximum(np.cumsum(signal[::-1] ** 2)[::-1], 1e-12)
                )
                edc -= edc[0]

                (line2,) = ax_sch.plot(
                    time, edc, color=color, linestyle="--", label=f"EDC-{label}"
                )
                ax_sch.set_ylabel("Energy Decay [dB]")

                def mark(ax1, ax2, edc_curve, label_prefix, color, marker, db):
                    idx = np.argmax(edc_curve < -db)
                    t = time[idx]
                    y_edc = edc_curve[idx]
                    ax1.axvline(x=t, color=color, linestyle=":")
                    (pt,) = ax2.plot(
                        t,
                        y_edc,
                        marker=marker,
                        color=color,
                        label=f"{label_prefix} {db}dB",
                    )
                    return pt

                if row == 0:
                    handles_combined.extend([line1, line2])
                    labels_combined.extend(["GT", "EDC-GT"])
                elif row == 1:
                    handles_combined.extend([line1, line2])
                    labels_combined.extend(["Pred", "EDC-Pred"])

                for db_val, color_val, marker_val in zip(
                    [0, 5, 35], ["red", "blue", "green"], ["o", "^", "s"]
                ):
                    pt = mark(
                        ax_wave, ax_sch, edc, label, color_val, marker_val, db_val
                    )
                    if ch == 0 and i == 0:  # Only mark for GT Left
                        handles_combined.append(pt)
                        labels_combined.append(f"{db_val}dB")

                ax_wave.set_title(f"{side} Channel - {label}")
                ax_wave.grid(True)

        axs[-1].set_xlabel("Time [s]")
        fig.legend(handles_combined, labels_combined, loc="upper right")
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(filename)
        plt.close(fig)

    for tag, idx in sel_indices:
        plot_comparison_heatmap(idx, "mag")
        plot_comparison_heatmap(idx, "phase")

        for case in cache[idx]["audio_pairs"]:
            pred_audio, gt_audio = cache[idx]["audio_pairs"][case]
            pred_audio = pred_audio.cpu().numpy()
            gt_audio = gt_audio.cpu().numpy()
            filename = output_dir / f"waveform_{tag}_{case}.png"
            plot_waveform_and_schroeder(gt_audio, pred_audio, cfg.sr, str(filename))

    ################################# report ################################
    def mean(key: str, case: str | None = None):
        if case is None:
            return sums[key] / counts["spec"]
        return sums[f"{case}_{key}"] / counts[case]

    print("\n====================  Test Results  ====================")
    print(f"Spectral L1   (Mag)  : {mean('spec_mag'):8.4f}")
    print(f"Spectral L1 (Phase)  : {mean('spec_phase'):8.4f}\n")

    table_rows = []
    for header, tag in zip(CASE_NAMES, ["c1", "c2", "c3", "c4", "c5a", "c5b"]):
        table_rows.append(
            (
                header,
                f"{mean('t60', tag) * 100:6.2f} %",
                f"{mean('t60_abs', tag):6.6f}",
                f"{mean('c50', tag):8.4f}",
                f"{mean('edt', tag):8.4f}",
                f"{mean('edt_rel', tag) * 100:8.4f}",
                f"{mean('snr', tag):8.4f}",
                f"{mean('psnr', tag):8.4f}",
            )
        )

    col_names = ("Case", "T60 %", "T60_ABS", "C50", "EDT", "EDT %", "SNR", "PSNR")
    col_fmt = "{:<36s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
    print(col_fmt.format(*col_names))
    print("-" * 80)
    for row in table_rows:
        print(col_fmt.format(*row))
    print("==========================================================")

    if gen_times:
        total_gen = sum(gen_times)
        print("\n=================  Generation Timing  =================")
        print(f"Total generation time (mag+phase): {total_gen:.2f} s")
        print(
            f"Avg per batch: {_stats.mean(gen_times):.4f} s | Median: {_stats.median(gen_times):.4f} s"
        )
        if total_gen > 0:
            throughput = frames_generated / total_gen
            print(
                f"Throughput: {throughput:.1f} frames/s "
                f"(frames = B×T×2 channels across all batches)"
            )
        print("=======================================================")


if __name__ == "__main__":
    main()
