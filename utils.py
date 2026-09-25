import json
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio.transforms import GriffinLim


def get_loc(name):
    match = re.match(r"(\d+)_(\d+)_(\d+)\.wav", name)
    if match:
        tx_idx = int(match.group(1))
        rx_idx = int(match.group(2))
        orientation = str(match.group(3))
        return tx_idx, rx_idx, orientation
    else:
        raise ValueError("Filename does not match the expected pattern")


def get_loc_GWA(name):
    match = re.match(r"(L\d+)_(R\d+)\.wav", name)
    if match:
        tx = match.group(1)
        rx_raw = match.group(2)
        rx = "R" + str(int(rx_raw[1:]))
        return tx, rx, "0"
    else:
        raise ValueError(f"Filename does not match expected pattern: {name}")


def read_3d_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                _, x, y, z = parts
                point = (float(x), float(y), float(z))
                points.append(point)
    return points


def read_named_3d_points(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    result = {}

    for group in ("receivers", "sources"):
        if group in data:
            for item in data[group]:
                name = item["name"]
                if name[0] == "S":
                    name = "L" + name[1:]
                xyz = tuple(item["xyz"])
                result[name] = xyz

    return result


def normalize_point(points, idx):
    """
    Normalizes the coordinates of a point at the given index in a list of 3D points.

    Parameters:
    points (list of tuple): List of tuples where each tuple represents a 3D point (x, y, z).
    idx (int): Index of the point to normalize.

    Returns:
    tuple: Normalized coordinates of the point at the given index.
    """

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    point = points[idx]

    normalized_x = (point[0] - min_x) / (max_x - min_x)
    normalized_y = (point[1] - min_y) / (max_y - min_y)
    normalized_z = (point[2] - min_z) / (max_z - min_z)

    return (normalized_x, normalized_y, normalized_z)


def if_to_phase(predicted_if):
    """
    Compute the phase from the predicted instantaneous frequency (IF).

    Parameters:
    - predicted_if: torch.Tensor of shape (batch_size, 2, n_freq, n_time), containing the predicted IF.

    Returns:
    - phase: torch.Tensor of shape (batch_size, 2, n_freq, n_time), containing the computed phase.
    """
    angular_frequency = predicted_if * (2 * torch.pi)

    unwrapped_phase = torch.cumsum(angular_frequency, dim=-1)

    phase = torch.remainder(unwrapped_phase + torch.pi, 2 * torch.pi) - torch.pi

    return phase


def reconstruct_sound(
    log_magnitude_batch, phase_batch, sr: int, n_fft: int = 512
) -> torch.Tensor:
    """
    Reconstructs binaural sound from log-magnitude STFT spectrum.

    Args:
    log_magnitude_batch (torch.Tensor): Log-magnitude STFT spectrum of shape (batch_size, 2, n_freq, n_time).
    phase_batch (torch.Tensor): Phase STFT spectrum of shape (batch_size, 2, n_freq, n_time).
    sr (int): Sample rate of the audio.
    n_fft (int): Number of FFT components. Default is 512.

    Returns:
    torch.Tensor: Reconstructed binaural sound, shape (batch_size, 2, N).
    """

    batch_size, n_channels, n_freq, n_time = log_magnitude_batch.shape

    reconstructed_audio = []

    for b in range(batch_size):
        channels_audio = []
        for c in range(n_channels):
            log_mag = log_magnitude_batch[b, c, :, :]
            phase = phase_batch[b, c, :, :]

            magnitude = torch.exp(log_mag) - 1e-3

            complex_spectrum = magnitude * torch.exp(1j * phase)

            hann_window = torch.hann_window(
                n_fft,
                periodic=True,
                dtype=torch.float32,
                device=log_magnitude_batch.device,
            )
            audio = torch.istft(complex_spectrum, n_fft=n_fft, window=hann_window).to(
                log_magnitude_batch.device
            )

            channels_audio.append(audio)

        reconstructed_audio.append(torch.stack(channels_audio, dim=0))

    reconstructed_audio = torch.stack(reconstructed_audio, dim=0)

    return reconstructed_audio


def reconstruct_sound_with_random_phase(
    pred_log_magnitude_batch, gt_log_magnitude_batch, sr: int, n_fft: int = 512
) -> tuple:
    """
    Reconstructs binaural sound from log-magnitude STFT spectrum using a random phase.

    Args:
    pred_log_magnitude_batch (torch.Tensor): Log-magnitude STFT spectrum of Predicted version (batch_size, 2, n_freq, n_time).
    gt_log_magnitude_batch (torch.Tensor): Log-magnitude STFT spectrum of Ground Truth (batch_size, 2, n_freq, n_time).
    sr (int): Sample rate of the audio.
    n_fft (int): Number of FFT components. Default is 512.

    Returns:
    tuple: Reconstructed binaural sounds for Ground Truth and Predicted, each of shape (batch_size, 2, N).
    """

    def generate_random_phase(shape):
        rp = np.random.uniform(-np.pi, np.pi, shape)
        return torch.from_numpy(rp).float()

    device = gt_log_magnitude_batch.device

    batch_size, n_channels, n_freq, n_time = gt_log_magnitude_batch.shape
    random_phase = generate_random_phase((batch_size, n_channels, n_freq, n_time)).to(
        device
    )

    gt_reconstructed_audio = reconstruct_sound(
        gt_log_magnitude_batch, random_phase, sr, n_fft
    )

    pred_reconstructed_audio = reconstruct_sound(
        pred_log_magnitude_batch, random_phase, sr, n_fft
    )

    return pred_reconstructed_audio, gt_reconstructed_audio


def reconstruct_sound_with_griffin_lim(
    pred_log_magnitude_batch,
    gt_log_magnitude_batch,
    sr: int,
    n_fft: int = 512,
    hop_length: int = None,
    power: int = 1,
) -> tuple:
    """
    Reconstructs binaural sound from log-magnitude STFT spectrum using the Griffin-Lim algorithm.

    Args:
    pred_log_magnitude_batch (torch.Tensor): Log-magnitude STFT spectrum of Predicted version (batch_size, 2, n_freq, n_time).
    gt_log_magnitude_batch (torch.Tensor): Log-magnitude STFT spectrum of Ground Truth (batch_size, 2, n_freq, n_time).
    sr (int): Sample rate of the audio.
    n_fft (int): Number of FFT components. Default is 512.
    hop_length (int): Hop length for STFT. If None, it defaults to n_fft // 4.
    power (int): Power to which the magnitude is raised before applying Griffin-Lim. Default is 1.

    Returns:
    tuple: Reconstructed binaural sounds for Ground Truth and Predicted, each of shape (batch_size, 2, N).
    """
    device = gt_log_magnitude_batch.device

    if hop_length is None:
        hop_length = n_fft // 4

    griffin_lim_transform = GriffinLim(
        n_fft=n_fft, hop_length=hop_length, power=power
    ).to(device)

    batch_size, n_channels, n_freq, n_time = gt_log_magnitude_batch.shape

    gt_reconstructed_audio = []
    pred_reconstructed_audio = []

    for b in range(batch_size):
        gt_channels_audio = []
        pred_channels_audio = []

        for c in range(n_channels):
            gt_log_mag = gt_log_magnitude_batch[b, c, :, :]
            gt_magnitude = torch.exp(gt_log_mag) - 1e-3

            pred_log_mag = pred_log_magnitude_batch[b, c, :, :]
            pred_magnitude = torch.exp(pred_log_mag) - 1e-3

            gt_audio = griffin_lim_transform(gt_magnitude)
            pred_audio = griffin_lim_transform(pred_magnitude)

            gt_channels_audio.append(gt_audio)
            pred_channels_audio.append(pred_audio)

        gt_reconstructed_audio.append(torch.stack(gt_channels_audio, dim=0))
        pred_reconstructed_audio.append(torch.stack(pred_channels_audio, dim=0))

    gt_reconstructed_audio = torch.stack(gt_reconstructed_audio, dim=0)
    pred_reconstructed_audio = torch.stack(pred_reconstructed_audio, dim=0)

    return pred_reconstructed_audio, gt_reconstructed_audio


def compute_t60(audio: torch.Tensor, sr: int):
    """
    Compute the T60 reverberation time for an audio signal.

    Parameters:
        audio (torch.Tensor): Audio signal, shape (N,).
        sr (int): Sample rate.

    Returns:
        rt30 (torch.Tensor): T30 reverberation time.
    """
    power = audio.pow(2)
    energy_decay = torch.flip(torch.cumsum(torch.flip(power, [0]), dim=0), [0])

    decay_db = 10 * torch.log10(energy_decay)
    decay_db = decay_db - decay_db[0]

    try:
        decay_5_idx = torch.where(decay_db <= -5)[0][0]
    except IndexError:
        decay_5_idx = torch.tensor(0)

    try:
        decay_35_idx = torch.where(decay_db <= -35)[0][0]
    except IndexError:
        decay_35_idx = torch.tensor(len(decay_db) - 1)

    rt30 = (decay_35_idx - decay_5_idx).item() / sr

    return rt30


def calculate_t60_percentage(
    predicted_audio_batch: torch.Tensor, gt_audio_batch: torch.Tensor, sr: int
):
    """
    Calculate the T60 percentage of the predicted audio compared to the ground truth audio for a batch of inputs.

    Parameters:
        predicted_audio_batch (torch.Tensor): Predicted binaural audio, shape (batch_size, 2, N).
        gt_audio_batch (torch.Tensor): Ground truth binaural audio, shape (batch_size, 2, N).
        sr (int): Sample rate.

    Returns:
        t60_percentage_batch (torch.Tensor): T60 percentage of the predicted audio compared to the ground truth audio for each batch element.
    """
    batch_size, n_channel, _ = predicted_audio_batch.shape
    t60_percentage_batch = []

    for i in range(batch_size):
        predicted_audio = predicted_audio_batch[i]
        gt_audio = gt_audio_batch[i]

        t60_pred = torch.tensor(
            [compute_t60(predicted_audio[ch], sr) for ch in range(n_channel)]
        )
        t60_gt = torch.tensor(
            [compute_t60(gt_audio[ch], sr) for ch in range(n_channel)]
        )

        score = torch.abs(t60_pred - t60_gt) / torch.abs(t60_gt)
        t60_percentage_batch.append(score)

    return torch.stack(t60_percentage_batch).to(predicted_audio.device)


def calculate_t60_absolute_error(
    predicted_audio_batch: torch.Tensor, gt_audio_batch: torch.Tensor, sr: int
):
    """
    Calculate the absolute error of T60 between the predicted audio and ground truth audio for a batch of inputs.

    Parameters:
        predicted_audio_batch (torch.Tensor): Predicted binaural audio, shape (batch_size, 2, N).
        gt_audio_batch (torch.Tensor): Ground truth binaural audio, shape (batch_size, 2, N).
        sr (int): Sample rate.

    Returns:
        t60_abs_error_batch (torch.Tensor): Absolute T60 error in seconds for each channel of each batch element.
                                            Shape: (batch_size, 2)
    """
    batch_size, n_channel, _ = predicted_audio_batch.shape
    t60_abs_error_batch = []

    for i in range(batch_size):
        predicted_audio = predicted_audio_batch[i]
        gt_audio = gt_audio_batch[i]

        t60_pred = torch.tensor(
            [compute_t60(predicted_audio[ch], sr) for ch in range(n_channel)]
        )
        t60_gt = torch.tensor(
            [compute_t60(gt_audio[ch], sr) for ch in range(n_channel)]
        )

        abs_error = torch.abs(t60_pred - t60_gt)
        t60_abs_error_batch.append(abs_error)

    return torch.stack(t60_abs_error_batch).to(predicted_audio.device)


def compute_edt(audio: torch.Tensor, sr: int):
    """
    Compute the T60 reverberation time for an audio signal.

    Parameters:
        audio (torch.Tensor): Audio signal, shape (N,).
        sr (int): Sample rate.

    Returns:
        rt30 (torch.Tensor): T30 reverberation time.
    """
    power = audio.pow(2)
    energy_decay = torch.flip(torch.cumsum(torch.flip(power, [0]), dim=0), [0])

    decay_db = 10 * torch.log10(energy_decay)
    decay_db -= decay_db[0]

    decay_10_idx = torch.where(decay_db <= -10)[0][0]

    rt30 = decay_10_idx.item() / sr

    return rt30


def calculate_edt_error(
    predicted_audio_batch: torch.Tensor, gt_audio_batch: torch.Tensor, sr: int
):
    """
    Calculate the EDT error (in seconds) between the predicted audio and the ground truth audio for a batch of inputs.

    Parameters:
        predicted_audio_batch (torch.Tensor): Predicted binaural audio, shape (batch_size, 2, N).
        gt_audio_batch (torch.Tensor): Ground truth binaural audio, shape (batch_size, 2, N).
        sr (int): Sample rate.

    Returns:
        edt_error_batch (torch.Tensor): EDT error in seconds for each batch element.
    """
    batch_size, n_channel, _ = predicted_audio_batch.shape
    edt_error_batch = []

    for i in range(batch_size):
        predicted_audio = predicted_audio_batch[i]
        gt_audio = gt_audio_batch[i]

        t60_pred = torch.tensor(
            [compute_t60(predicted_audio[ch], sr) for ch in range(n_channel)]
        )
        t60_gt = torch.tensor(
            [compute_t60(gt_audio[ch], sr) for ch in range(n_channel)]
        )

        score = torch.abs(t60_pred - t60_gt)

        edt_error_batch.append(score)

    return torch.stack(edt_error_batch).to(predicted_audio.device)


def calculate_edt_relative_error(
    predicted_audio_batch: torch.Tensor, gt_audio_batch: torch.Tensor, sr: int
):
    """
    Calculate the EDT error (in seconds) between the predicted audio and the ground truth audio for a batch of inputs.

    Parameters:
        predicted_audio_batch (torch.Tensor): Predicted binaural audio, shape (batch_size, 2, N).
        gt_audio_batch (torch.Tensor): Ground truth binaural audio, shape (batch_size, 2, N).
        sr (int): Sample rate.

    Returns:
        edt_error_batch (torch.Tensor): EDT error in seconds for each batch element.
    """
    batch_size, n_channel, _ = predicted_audio_batch.shape
    edt_error_batch = []

    for i in range(batch_size):
        predicted_audio = predicted_audio_batch[i]
        gt_audio = gt_audio_batch[i]

        t60_pred = torch.tensor(
            [compute_t60(predicted_audio[ch], sr) for ch in range(n_channel)]
        )
        t60_gt = torch.tensor(
            [compute_t60(gt_audio[ch], sr) for ch in range(n_channel)]
        )

        score = torch.abs(t60_pred - t60_gt) / torch.abs(t60_gt)

        edt_error_batch.append(score)

    return torch.stack(edt_error_batch).to(predicted_audio.device)


def calculate_c50_error(
    predicted_audio_batch: torch.Tensor, gt_audio_batch: torch.Tensor, sr: int
) -> torch.Tensor:
    """
    Calculate the C50 error (in dB) between the predicted audio and the ground truth audio for a batch of inputs.

    Parameters:
        predicted_audio_batch (np.ndarray): Predicted binaural audio, shape (batch_size, 2, N).
        gt_audio_batch (np.ndarray): Ground truth binaural audio, shape (batch_size, 2, N).
        sr (int): Sample rate.

    Returns:
        c50_error_batch (np.ndarray): C50 error in dB for each batch element.
    """

    def calculate_c50(audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Calculate the C50 clarity index for a given audio signal.

        Parameters:
            audio (torch.Tensor): Mono audio signal.
            sr (int): Sample rate.

        Returns:
            c50 (torch.Tensor): C50 clarity index in dB.
        """
        early_time = int(0.05 * sr) + 1
        early_energy = torch.sum(audio[:early_time] ** 2)
        late_energy = torch.sum(audio[early_time:] ** 2)
        c50 = 10 * torch.log10(early_energy / late_energy)
        return c50

    batch_size = predicted_audio_batch.shape[0]
    c50_error_batch = []

    for i in range(batch_size):
        predicted_audio = predicted_audio_batch[i]
        gt_audio = gt_audio_batch[i]

        n_channels = predicted_audio.shape[0]
        abs_errors = []

        for ch in range(n_channels):
            c50_pred = calculate_c50(predicted_audio[ch], sr)
            c50_gt = calculate_c50(gt_audio[ch], sr)
            abs_errors.append(torch.abs(c50_pred - c50_gt))

        # Mean across channels
        mean_abs_error = torch.mean(torch.stack(abs_errors))
        c50_error_batch.append(mean_abs_error)

    return torch.stack(c50_error_batch).to(predicted_audio_batch.device)


def compute_snr(pred_rir: torch.Tensor, gt_rir: torch.Tensor) -> torch.Tensor:
    """
    Computes Signal-to-Noise Ratio (SNR) for each batch.

    Parameters:
        gt_rir (torch.Tensor): Ground truth RIR of shape (batch_size, N).
        pred_rir (torch.Tensor): Predicted RIR of shape (batch_size, N).

    Returns:
        torch.Tensor: SNR values of shape (batch_size,).
    """
    signal_power = torch.mean(gt_rir**2, dim=1)
    noise = gt_rir - pred_rir
    noise_power = torch.mean(noise**2, dim=1)
    snr_db = 10 * torch.log10(signal_power / noise_power)
    return snr_db


def compute_psnr(
    pred_rir: torch.Tensor, gt_rir: torch.Tensor, max_value: float = 1.0
) -> torch.Tensor:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) for each batch.

    Parameters:
        gt_rir (torch.Tensor): Ground truth RIR of shape (batch_size, N).
        pred_rir (torch.Tensor): Predicted RIR of shape (batch_size, N).
        max_value (float): Maximum possible value of the signal. Default is 1.0 for normalized RIR.

    Returns:
        torch.Tensor: PSNR values of shape (batch_size,).
    """
    mse = torch.mean((gt_rir - pred_rir) ** 2, dim=1)
    mse = torch.clamp(mse, min=1e-10)
    psnr_db = 10 * torch.log10(max_value**2 / mse)
    return psnr_db


class History:
    def __init__(self):
        self.epochs = []
        self.train_losses_mag = []
        self.val_losses_mag = []
        self.train_losses_phase = []
        self.val_losses_phase = []
        self.val_t60_mag = []
        self.val_c50_mag = []
        self.val_edt_mag = []
        self.val_t60_phase = []
        self.val_c50_phase = []
        self.val_edt_phase = []

    def update(
        self,
        epoch,
        train_loss_mag,
        val_loss_mag,
        train_loss_phase,
        val_loss_phase,
        t60_loss_mag,
        c50_loss_mag,
        edt_loss_mag,
        t60_loss_phase,
        c50_loss_phase,
        edt_loss_phase,
    ):
        self.epochs.append(epoch)
        self.train_losses_mag.append(train_loss_mag)
        self.val_losses_mag.append(val_loss_mag)
        self.train_losses_phase.append(train_loss_phase)
        self.val_losses_phase.append(val_loss_phase)
        self.val_t60_mag.append(t60_loss_mag)
        self.val_c50_mag.append(c50_loss_mag)
        self.val_edt_mag.append(edt_loss_mag)
        self.val_t60_phase.append(t60_loss_phase)
        self.val_c50_phase.append(c50_loss_phase)
        self.val_edt_phase.append(edt_loss_phase)

    def save(self, filepath):
        np.savez(
            filepath,
            epochs=self.epochs,
            train_losses_mag=self.train_losses_mag,
            val_losses_mag=self.val_losses_mag,
            train_losses_phase=self.train_losses_phase,
            val_losses_phase=self.val_losses_phase,
            val_t60_mag=self.val_t60_mag,
            val_c50_mag=self.val_c50_mag,
            val_edt_mag=self.val_edt_mag,
            val_t60_phase=self.val_t60_phase,
            val_c50_phase=self.val_c50_phase,
            val_edt_phase=self.val_edt_phase,
        )

    def load(self, filepath):
        data = np.load(filepath)
        self.epochs = data["epochs"].tolist()
        self.train_losses_mag = data["train_losses_mag"].tolist()
        self.val_losses_mag = data["val_losses_mag"].tolist()
        self.train_losses_phase = data["train_losses_phase"].tolist()
        self.val_losses_phase = data["val_losses_phase"].tolist()
        self.val_t60_mag = data["val_t60_mag"].tolist()
        self.val_c50_mag = data["val_c50_mag"].tolist()
        self.val_edt_mag = data["val_edt_mag"].tolist()
        self.val_t60_phase = data["val_t60_phase"].tolist()
        self.val_c50_phase = data["val_c50_phase"].tolist()
        self.val_edt_phase = data["val_edt_phase"].tolist()


def plot_history(history, filepath):
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    axs[0].plot(history.epochs, history.train_losses_mag, label="Train Loss Mag")
    axs[0].plot(history.epochs, history.val_losses_mag, label="Val Loss Mag")
    axs[0].plot(history.epochs, history.train_losses_phase, label="Train Loss Phase")
    axs[0].plot(history.epochs, history.val_losses_phase, label="Val Loss Phase")
    axs[0].set_title("Train & Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(history.epochs, history.val_t60_mag, label="Val T60 Loss Mag (%)")
    axs[1].plot(history.epochs, history.val_t60_phase, label="Val T60 Loss Phase (%)")
    axs[1].set_title("Validation T60 Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("T60 Loss (%)")
    axs[1].axhline(y=3.18, color="r", linestyle="--")
    axs[1].axhline(y=2.36, color="g", linestyle="--")
    axs[1].axhline(y=2.04, color="m", linestyle="--")
    axs[1].legend()

    axs[2].plot(history.epochs, history.val_c50_mag, label="Val C50 Loss Mag (dB)")
    axs[2].plot(history.epochs, history.val_c50_phase, label="Val C50 Loss Phase (dB)")
    axs[2].set_title("Validation C50 Loss")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("C50 Loss (dB)")
    axs[2].axhline(y=1.06, color="r", linestyle="--")
    axs[2].axhline(y=0.50, color="g", linestyle="--")
    axs[2].axhline(y=0.39, color="m", linestyle="--")
    axs[2].legend()

    axs[3].plot(history.epochs, history.val_edt_mag, label="Val EDT Loss Mag (sec)")
    axs[3].plot(history.epochs, history.val_edt_phase, label="Val EDT Loss Phase (sec)")
    axs[3].set_title("Validation EDT Loss")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("EDT Loss (sec)")
    axs[3].axhline(y=0.031, color="r", linestyle="--")
    axs[3].axhline(y=0.0143, color="g", linestyle="--")
    axs[3].axhline(y=0.011, color="m", linestyle="--")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_schroeder_curve(IR1, IR2, sr, filename):
    energy_decay1 = np.cumsum(IR1[::-1] ** 2)[::-1]
    energy_decay2 = np.cumsum(IR2[::-1] ** 2)[::-1]

    energy_decay_db1 = 10 * np.log10(energy_decay1)
    energy_decay_db1 -= energy_decay_db1[0]
    energy_decay_db2 = 10 * np.log10(energy_decay2)
    energy_decay_db2 -= energy_decay_db2[0]

    time_axis = np.arange(len(IR1)) / sr

    peak_db1 = np.max(energy_decay_db1)
    peak_idx1 = np.argmax(energy_decay_db1)
    peak_db2 = np.max(energy_decay_db2)
    peak_idx2 = np.argmax(energy_decay_db2)

    drop_5dB_idx1 = np.min(np.where(energy_decay_db1 < peak_db1 - 5)[0])
    drop_35dB_idx1 = np.min(np.where(energy_decay_db1 < peak_db1 - 35)[0])
    drop_5dB_idx2 = np.min(np.where(energy_decay_db2 < peak_db2 - 5)[0])
    drop_35dB_idx2 = np.min(np.where(energy_decay_db2 < peak_db2 - 35)[0])

    T60_1 = (drop_35dB_idx1 - drop_5dB_idx1) / sr
    T60_2 = (drop_35dB_idx2 - drop_5dB_idx2) / sr

    plt.figure(figsize=(20, 5))
    plt.plot(time_axis, energy_decay_db1, label="RIR_1")
    plt.plot(time_axis, energy_decay_db2, label="RIR_2")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy Decay (dB)")
    plt.title("Schroeder Curve")
    plt.grid(True)

    plt.plot(time_axis[peak_idx1], peak_db1, "ro", label="Peak 1")
    plt.plot(time_axis[peak_idx2], peak_db2, "r+", label="Peak 2")

    plt.plot(
        time_axis[drop_5dB_idx1], energy_decay_db1[drop_5dB_idx1], "go", label="-5 dB 1"
    )
    plt.plot(
        time_axis[drop_5dB_idx2], energy_decay_db2[drop_5dB_idx2], "g+", label="-5 dB 2"
    )

    plt.plot(
        time_axis[drop_35dB_idx1],
        energy_decay_db1[drop_35dB_idx1],
        "bo",
        label="-35 dB 1",
    )
    plt.plot(
        time_axis[drop_35dB_idx2],
        energy_decay_db2[drop_35dB_idx2],
        "b+",
        label="-35 dB 2",
    )

    plt.text(0, -70, f"T30 for RIR1: {T60_1:.6f}", fontsize=12, color="black")
    plt.text(0, -76, f"T30 for RIR2: {T60_2:.6f}", fontsize=12, color="black")

    plt.legend()
    plt.savefig(filename)
