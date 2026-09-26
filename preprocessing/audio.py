from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import h5py
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


@dataclass(frozen=True)
class AudioPreprocessResult:
    audio_names: list[str]
    min_length: int
    sample_rate: int
    n_fft: int


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(value, dtype=np.float64)
            self.m2 = np.zeros_like(value, dtype=np.float64)
        if value.shape != self.mean.shape:
            raise ValueError(
                f"Spectrum shape changed from {self.mean.shape} to {value.shape}"
            )

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty running statistics")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def list_wav_files(audio_dir: Path) -> list[Path]:
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    audio_files = sorted(audio_dir.glob("*.wav"), key=lambda path: path.name)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in: {audio_dir}")
    return audio_files


def _resampled_length(path: Path, target_sample_rate: int) -> int:
    try:
        sample_rate, data = wavfile.read(path, mmap=True)
    except ValueError:
        sample_rate, data = wavfile.read(path)
    sample_count = data.shape[0]
    return int(np.ceil(sample_count * target_sample_rate / sample_rate))


def find_common_length(audio_files: list[Path], target_sample_rate: int) -> int:
    lengths = [
        _resampled_length(path, target_sample_rate)
        for path in tqdm(audio_files, desc="Inspecting WAV lengths")
    ]
    return min(lengths)


def load_binaural_audio(
    path: Path, target_sample_rate: int, length: int
) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    data = np.asarray(data)

    if data.ndim == 1:
        data = np.stack([data, data], axis=0)
    elif data.ndim == 2 and data.shape[1] in {1, 2}:
        data = data.T
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
    elif data.ndim == 2 and data.shape[0] == 2:
        pass
    else:
        raise ValueError(
            f"Expected mono or stereo WAV data in {path.name}, got {data.shape}"
        )

    data = data.astype(np.float32, copy=False)
    if sample_rate != target_sample_rate:
        data = librosa.resample(
            data,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
            axis=-1,
        )

    if data.shape[-1] < length:
        data = np.pad(data, ((0, 0), (0, length - data.shape[-1])))
    return data[:, :length]


def phase_to_if(phase: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(phase, axis=-1)
    angular_frequency = np.concatenate(
        [unwrapped[..., :1], np.diff(unwrapped, axis=-1)], axis=-1
    )
    return angular_frequency / (2 * np.pi)


def compute_spectra(audio: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = librosa.stft(audio, n_fft=n_fft)
    log_magnitude = np.log(np.abs(spectrum) + 1e-3)
    instantaneous_frequency = phase_to_if(np.angle(spectrum))
    return log_magnitude, instantaneous_frequency


def _h5_kwargs(compression: str | None) -> dict:
    if compression in (None, "none"):
        return {}
    return {"compression": compression}


def _check_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            f"Preprocessing outputs already exist. Use --overwrite to replace them:\n{formatted}"
        )


def preprocess_audio(
    audio_dir: Path,
    output_dir: Path,
    target_sample_rate: int = 22050,
    n_fft: int = 512,
    compression: str | None = None,
    overwrite: bool = False,
) -> AudioPreprocessResult:
    audio_files = list_wav_files(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log_magnitude.h5"
    phase_path = output_dir / "phase_spectrum.h5"
    _check_outputs([log_path, phase_path], overwrite)

    min_length = find_common_length(audio_files, target_sample_rate)
    log_moments = RunningMoments()
    phase_moments = RunningMoments()

    for path in tqdm(audio_files, desc="Computing spectrum statistics"):
        audio = load_binaural_audio(path, target_sample_rate, min_length)
        log_magnitude, instantaneous_frequency = compute_spectra(audio, n_fft)
        log_moments.update(log_magnitude)
        phase_moments.update(instantaneous_frequency)

    log_mean, log_std = log_moments.finalize()
    phase_mean, phase_std = phase_moments.finalize()

    temporary_log_path = output_dir / ".log_magnitude.h5.tmp"
    temporary_phase_path = output_dir / ".phase_spectrum.h5.tmp"
    dataset_kwargs = _h5_kwargs(compression)

    for path in (temporary_log_path, temporary_phase_path):
        if path.exists():
            path.unlink()

    try:
        with (
            h5py.File(temporary_log_path, "w", track_order=True) as log_h5,
            h5py.File(temporary_phase_path, "w", track_order=True) as phase_h5,
        ):
            for handle, mean, std in (
                (log_h5, log_mean, log_std),
                (phase_h5, phase_mean, phase_std),
            ):
                handle.create_dataset("mean", data=mean)
                handle.create_dataset("std", data=std)
                handle.create_dataset("min_len", data=min_length)

            for path in tqdm(audio_files, desc="Writing standardized spectra"):
                audio = load_binaural_audio(path, target_sample_rate, min_length)
                log_magnitude, instantaneous_frequency = compute_spectra(audio, n_fft)
                standardized_log = (log_magnitude - log_mean) / (3 * log_std + 1e-8)
                standardized_phase = (instantaneous_frequency - phase_mean) / (
                    3 * phase_std + 1e-8
                )
                log_h5.create_dataset(
                    path.name,
                    data=standardized_log.astype(np.float32),
                    **dataset_kwargs,
                )
                phase_h5.create_dataset(
                    path.name,
                    data=standardized_phase.astype(np.float32),
                    **dataset_kwargs,
                )

        os.replace(temporary_log_path, log_path)
        os.replace(temporary_phase_path, phase_path)
    finally:
        for path in (temporary_log_path, temporary_phase_path):
            if path.exists():
                path.unlink()

    return AudioPreprocessResult(
        audio_names=[path.name for path in audio_files],
        min_length=min_length,
        sample_rate=target_sample_rate,
        n_fft=n_fft,
    )
