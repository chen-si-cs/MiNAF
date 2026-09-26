from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import sys
import tempfile

import h5py
import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.audio import preprocess_audio
from preprocessing.geometry import extract_features
from preprocessing.pipelines import create_splits
from SoundDataset import SoundDataset


class FakeRayIntersector:
    def intersects_location(
        self, ray_origins, ray_directions, multiple_hits=True
    ):
        ray_indices = np.arange(len(ray_directions))
        distances = 1.0 + ray_indices / max(1, len(ray_indices))
        locations = ray_origins + ray_directions * distances[:, None]
        triangle_indices = np.zeros(len(ray_indices), dtype=np.int64)
        return locations, ray_indices, triangle_indices


class FakeMesh:
    def __init__(self):
        self.ray = FakeRayIntersector()
        self.face_normals = np.array([[0.0, 0.0, 1.0]])


def check_geometry() -> None:
    n_rays = 16
    thresholds = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    features = extract_features(
        FakeMesh(),
        np.zeros(3),
        thresholds,
        n_rays=n_rays,
        neighbor_count=4,
    )
    expected_size = 6 * n_rays + len(thresholds)
    if features.shape != (expected_size,):
        raise AssertionError(f"Expected {(expected_size,)}, got {features.shape}")
    if not np.isfinite(features).all():
        raise AssertionError("Geometry features contain non-finite values")


def write_splits(path: Path, count: int) -> None:
    with path.open("wb") as stream:
        pickle.dump(
            {
                "train": np.arange(count),
                "val": np.array([], dtype=np.int64),
                "test": np.array([], dtype=np.int64),
            },
            stream,
        )


def check_audio_and_loaders() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        audio_dir = root / "soundspaces_audio"
        processed_root = root / "processed"
        scene = "soundspaces_scene"
        output_dir = processed_root / scene
        audio_dir.mkdir()

        time_8k = np.arange(1024) / 8000
        stereo = np.stack(
            [np.sin(2 * np.pi * 220 * time_8k), np.sin(2 * np.pi * 330 * time_8k)],
            axis=1,
        )
        wavfile.write(audio_dir / "0_1_0.wav", 8000, (stereo * 2000).astype(np.int16))

        time_16k = np.arange(2048) / 16000
        mono = np.sin(2 * np.pi * 440 * time_16k)
        wavfile.write(audio_dir / "1_0_90.wav", 16000, (mono * 2000).astype(np.int16))

        result = preprocess_audio(
            audio_dir,
            output_dir,
            target_sample_rate=8000,
            n_fft=64,
        )
        if result.audio_names != ["0_1_0.wav", "1_0_90.wav"]:
            raise AssertionError(f"Unexpected audio order: {result.audio_names}")

        with (
            h5py.File(output_dir / "log_magnitude.h5", "r") as log_h5,
            h5py.File(output_dir / "phase_spectrum.h5", "r") as phase_h5,
        ):
            for name in result.audio_names:
                if log_h5[name].shape != phase_h5[name].shape:
                    raise AssertionError(f"Spectrum shape mismatch for {name}")
                if log_h5[name].shape[0] != 2:
                    raise AssertionError("Mono input was not converted to binaural audio")

        metadata_root = root / "metadata"
        points_dir = metadata_root / scene
        points_dir.mkdir(parents=True)
        (points_dir / "points.txt").write_text(
            "0 0.0 0.0 0.0\n1 1.0 0.0 0.0\n", encoding="utf-8"
        )
        np.save(output_dir / "features.npy", np.zeros((2, 6 * 4 + 2), dtype=np.float32))
        write_splits(output_dir / "split_indices.pkl", len(result.audio_names))
        soundspaces_cfg = SimpleNamespace(
            dataset_type="soundspaces",
            data_root=str(processed_root),
            metadata_root=str(metadata_root),
            geometry_root=str(root / "unused"),
            apt=scene,
            n_rays=4,
            n_occlusion=2,
        )
        soundspaces_sample = SoundDataset(soundspaces_cfg, "train")[0]
        if soundspaces_sample[-1].shape != (26,):
            raise AssertionError("SoundSpaces loader returned the wrong feature shape")

        gwa_scene = "gwa_scene"
        gwa_root = root / "GWA" / gwa_scene
        gwa_audio_dir = gwa_root / "hybrid"
        gwa_audio_dir.mkdir(parents=True)
        wavfile.write(gwa_audio_dir / "L1_R1.wav", 8000, (mono * 2000).astype(np.int16))
        wavfile.write(gwa_audio_dir / "L1_R2.wav", 8000, (mono * 2000).astype(np.int16))
        (gwa_audio_dir / "sim_config.json").write_text(
            json.dumps(
                {
                    "sources": [{"name": "S1", "xyz": [0.0, 0.0, 0.0]}],
                    "receivers": [
                        {"name": "R1", "xyz": [1.0, 0.0, 0.0]},
                        {"name": "R2", "xyz": [0.0, 1.0, 0.0]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        gwa_output = processed_root / gwa_scene
        gwa_result = preprocess_audio(
            gwa_audio_dir,
            gwa_output,
            target_sample_rate=8000,
            n_fft=64,
        )
        feature_size = 6 * 4 + 3
        np.save(
            gwa_output / "features.npy",
            {
                "S1": np.zeros(feature_size, dtype=np.float32),
                "R1": np.zeros(feature_size, dtype=np.float32),
                "R2": np.zeros(feature_size, dtype=np.float32),
            },
        )
        write_splits(gwa_output / "split_indices.pkl", len(gwa_result.audio_names))
        gwa_cfg = SimpleNamespace(
            dataset_type="gwa",
            data_root=str(processed_root),
            metadata_root=str(root / "unused"),
            geometry_root=str(root / "GWA"),
            apt=gwa_scene,
            n_rays=4,
            n_occlusion=3,
        )
        gwa_sample = SoundDataset(gwa_cfg, "train")[0]
        if gwa_sample[-1].shape != (feature_size,):
            raise AssertionError("GWA loader returned the wrong feature shape")

        splits = create_splits(20, val_ratio=0.1, test_ratio=0.2, seed=0)
        if {name: len(values) for name, values in splits.items()} != {
            "train": 14,
            "val": 2,
            "test": 4,
        }:
            raise AssertionError("Unexpected split sizes")


def main() -> None:
    check_geometry()
    check_audio_and_loaders()
    print("Preprocessing check passed.")


if __name__ == "__main__":
    main()
